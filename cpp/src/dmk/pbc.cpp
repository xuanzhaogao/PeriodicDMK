#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/logger.h>
#include <dmk/omp_wrapper.hpp>
#include <dmk/legeexps.hpp>
#include <dmk/pbc.hpp>
#include <dmk/pbc_tree.hpp>
#include <dmk/prolate0_fun.hpp>
#include <dmk/tree.hpp>
#include <dmk/util.hpp>
#include <sctl.hpp>

#include <ducc0/nufft/nufft.h>
#include <ducc0/infra/mav.h>

#if PDMK_HAVE_FINUFFT
#include <finufft.h>
#endif

#include "pdmk/nufft_backend.hpp"

namespace dmk {
namespace pbc {

using pdmk::NufftBackend;
using pdmk::pick_nufft_backend;

// ============================================================
// ducc0 NUFFT dispatch (same Nufft<double,double,double> used by
// PeriodicDMK's WindowPotential). We always execute in double internally
// — if Real=float, positions/charges are cast on the way in/out.
// Grid shape passed to ducc0 is {slowest, ..., fastest} so the natural
// axis order for our (ms, mt, mu) Fourier grid is {mu, mt, ms}, with
// coord packing {z, y, x} per point. This matches
// PeriodicDMK/window_potential.cpp.
// ============================================================

// ============================================================
// Cached ducc0 plan state. Keyed by (sources pointer, ns, rc, eps, umat[9]).
// The same inputs on subsequent calls reuse the plans + coords + weights,
// dropping the per-call plan construction (~50 ms on 1M at 1e-3) and
// coord/weight re-computation.
// ============================================================
template <typename Real, int DIM>
struct NufftCache {
    const Real *sources = nullptr;
    int ns = -1;
    Real rc = 0, eps = 0, beta = 0;
    Real umat_key[DIM * DIM] = {};
    int64_t ms = 0, mt = 0, mu = 0, nk = 0;
    std::vector<double> coords;      // (ns, 3) packed {z,y,x} (DUCC path)
    std::vector<double> weights;     // nk
    std::unique_ptr<ducc0::Nufft<double, double, double>> plan1, plan2;

#if PDMK_HAVE_FINUFFT
    // FINUFFT needs coords in three separate x/y/z arrays. Same numeric values
    // as the DUCC `coords` triples, just unpacked. Kept as owned buffers so
    // finufft_setpts can hold pointers into them across calls.
    std::vector<double> fx, fy, fz;
    finufft_plan fplan1 = nullptr;   // type 1 (sources → grid)
    finufft_plan fplan2 = nullptr;   // type 2 (grid → sources)
    NufftBackend backend = NufftBackend::Ducc;
#endif

    bool matches(const Real *src, int ns_, Real rc_, Real eps_, Real beta_,
                 const Real *umat_, NufftBackend bk) const {
        if (src != sources || ns_ != ns || rc_ != rc || eps_ != eps || beta_ != beta)
            return false;
        for (int i = 0; i < DIM * DIM; ++i)
            if (umat_[i] != umat_key[i]) return false;
#if PDMK_HAVE_FINUFFT
        if (bk != backend) return false;
#else
        (void)bk;
#endif
        return true;
    }

#if PDMK_HAVE_FINUFFT
    ~NufftCache() {
        if (fplan1) { finufft_destroy(fplan1); fplan1 = nullptr; }
        if (fplan2) { finufft_destroy(fplan2); fplan2 = nullptr; }
    }
#endif
};

// ============================================================
// 2D Laplace far-field self-correction: W_0(r=0)
// W_0(0) = (1/V) Σ_{k≠0} W_hat(k)  where W_hat(k) = dfac_2d * fval(|k|rc/beta) / |k|²
// This mirrors the grid construction in far_field_nufft<Real,2>.
// ============================================================
template <typename Real>
static Real compute_far_field_self_correction_2d(const Real *umat, Real rc, Real beta) {
    const Real twopi = Real(2) * Real(M_PI);

    Real vol = umat[0] * umat[1 + 2];  // DIM==2: vol = umat[0]*umat[3]
    Prolate0Fun pf((double)beta, 5000);
    Real psi0 = (Real)pf.eval_val(0.0);

    Real dkmax = beta / rc;
    Real anrm1 = umat[0];
    Real anrm2 = std::sqrt(umat[2] * umat[2] + umat[3] * umat[3]);
    int64_t ms = std::max<int64_t>(2, 2 * (int64_t)std::ceil(dkmax * anrm1 / twopi));
    int64_t mt = std::max<int64_t>(2, 2 * (int64_t)std::ceil(dkmax * anrm2 / twopi));

    // Reciprocal-space row vectors from upper-triangular cell matrix
    Real a11 = Real(1) / umat[0];
    Real a21 = -umat[2] / (umat[0] * umat[3]);
    Real a22 = Real(1) / umat[3];
    // DMK 2D Laplace convention is G(r) = +log(r) (matches
    // laplace_2d_poly_all_pairs in vector_kernels.hpp). Its distributional
    // FT is -2π/k², giving dfac_2d = -2π/(psi0*vol) — analogous to the 3D
    // +1/r kernel's dfac = 4π/(psi0*vol).
    Real dfac_2d = -twopi / (psi0 * vol);

    Real sum = Real(0);
    for (int64_t j = -mt / 2; j <= (mt - 1) / 2; j++) {
        for (int64_t i = -ms / 2; i <= (ms - 1) / 2; i++) {
            if (i == 0 && j == 0) continue;
            Real dk1 = twopi * (a11 * i);
            Real dk2 = twopi * (a21 * i + a22 * j);
            Real dk2sq = dk1 * dk1 + dk2 * dk2;
            if (dk2sq < Real(1e-30)) continue;
            Real dk = std::sqrt(dk2sq);
            Real xval = dk * rc / beta;
            Real fval = (xval <= Real(1)) ? (Real)pf.eval_val(xval) : Real(0);
            sum += dfac_2d * fval / dk2sq;
        }
    }
    return sum;
}

// ============================================================
// Far-field NUFFT evaluation
// ============================================================

template <typename Real, int DIM>
void far_field_nufft(const Real *umat, Real rc, Real eps, Real beta,
                     int ns, const Real *sources, const Real *charge,
                     Real *pot, Real self_correction_value,
                     const Real *bloch) {
    using std::size_t;
    const Real twopi = Real(2) * Real(M_PI);
    const bool qp = (bloch != nullptr);

    if constexpr (DIM == 2) {
        if (qp)
            throw std::runtime_error("far_field_nufft<Real,2>: QP (bloch) not supported in 2D");

        // Inverse cell matrix + area
        Real umatinv[DIM * DIM];
        utri_inv<Real, DIM>(umat, umatinv);
        Real vol = umat[0] * umat[1 + DIM];

        // PSWF + grid sizes
        Prolate0Fun pf((double)beta, 5000);
        Real psi0 = (Real)pf.eval_val(0.0);
        Real dkmax = beta / rc;
        Real anrm1 = umat[0];
        Real anrm2 = std::sqrt(umat[DIM] * umat[DIM] + umat[1 + DIM] * umat[1 + DIM]);
        int64_t ms = std::max<int64_t>(2, 2 * (int64_t)std::ceil(dkmax * anrm1 / twopi));
        int64_t mt = std::max<int64_t>(2, 2 * (int64_t)std::ceil(dkmax * anrm2 / twopi));
        int64_t nk = ms * mt;

        // Coord packing: {t, s} per point (slowest first), matching grid {mt, ms}.
        std::vector<double> coords((size_t)ns * 2);
        for (int j = 0; j < ns; ++j) {
            Real x = sources[j * DIM], y = sources[j * DIM + 1];
            Real xj = twopi * (umatinv[0] * x + umatinv[DIM] * y);   // fast axis (s)
            Real yj = twopi * (umatinv[1 + DIM] * y);                // slow axis (t)
            coords[2 * (size_t)j + 0] = (double)yj;
            coords[2 * (size_t)j + 1] = (double)xj;
        }

        // Diagonal weights: W_hat_2d(k) = dfac_2d * fval(|k|rc/beta) / |k|²
        // with dfac_2d = -2π/(psi0*vol) — DMK's 2D Laplace convention is
        // G(r) = +log(r) (see LaplacePolyEvaluator2D::operator()), so
        // FT[G] = -2π/k², analogous to 3D where G=1/r gives FT=4π/k².
        Real a11 = Real(1) / umat[0];
        Real a21 = -umat[DIM] / (umat[0] * umat[1 + DIM]);
        Real a22 = Real(1) / umat[1 + DIM];
        Real dfac_2d = -twopi / (psi0 * vol);

        std::vector<double> weights((size_t)nk, 0.0);
        int64_t nidx = 0;
        for (int64_t j = -mt / 2; j <= (mt - 1) / 2; j++) {
            for (int64_t i = -ms / 2; i <= (ms - 1) / 2; i++) {
                Real dk1 = twopi * (a11 * i);
                Real dk2 = twopi * (a21 * i + a22 * j);
                Real dk2sq = dk1 * dk1 + dk2 * dk2;
                double w = 0.0;
                if (dk2sq >= Real(1e-30)) {
                    Real dk = std::sqrt(dk2sq);
                    Real xval = dk * rc / beta;
                    Real fval = (xval <= Real(1)) ? (Real)pf.eval_val(xval) : Real(0);
                    w = (double)(dfac_2d * fval / dk2sq);
                }
                weights[(size_t)nidx++] = w;  // k=0 entry stays 0.0 (gauge)
            }
        }

        // NUFFT Type 1 (source → grid), diagonal scaling, Type 2 (grid → source)
        std::vector<std::complex<double>> zcharge_d((size_t)ns);
        std::vector<std::complex<double>> fk_d((size_t)nk);
        std::vector<std::complex<double>> zpot_d((size_t)ns);
        for (int j = 0; j < ns; ++j)
            zcharge_d[(size_t)j] = std::complex<double>((double)charge[j], 0.0);

        {
            using namespace ducc0;
            const std::vector<size_t> shape = {(size_t)mt, (size_t)ms};
            const std::vector<double> periodicity = {2.0 * M_PI, 2.0 * M_PI};
            const double tol_d = (double)eps * 0.25;
            cmav<double, 2> coord_view(coords.data(), std::array<size_t, 2>{(size_t)ns, (size_t)2});

            Nufft<double, double, double> plan1(true, coord_view, shape, tol_d, 0, 1.1, 2.6,
                                                periodicity, false);
            Nufft<double, double, double> plan2(false, coord_view, shape, tol_d, 0, 1.1, 2.6,
                                                periodicity, false);

            cmav<std::complex<double>, 1> pts_cj(zcharge_d.data(), std::array<size_t, 1>{(size_t)ns});
            vfmav<std::complex<double>> uniform_v(fk_d.data(), shape);
            plan1.nu2u(true, 0, pts_cj, uniform_v);
            for (int64_t i = 0; i < nk; ++i)
                fk_d[(size_t)i] *= weights[(size_t)i];
            cfmav<std::complex<double>> uniform_c(fk_d.data(), shape);
            vmav<std::complex<double>, 1> pts_out(zpot_d.data(), std::array<size_t, 1>{(size_t)ns});
            plan2.u2nu(false, 0, uniform_c, pts_out);
        }

        // Match 3D non-QP convention: caller applies self-correction after combining
        // near+far. self_correction_value is unused here.
        (void)self_correction_value;
        for (int j = 0; j < ns; ++j)
            pot[j] += (Real)zpot_d[(size_t)j].real();
        return;
    }

    // QP path does not use the plan cache (first implementation).
    // k-grid size and weights depend on bloch; we just rebuild per call.
    if (qp) {
        Real umatinv[DIM * DIM];
        utri_inv<Real, DIM>(umat, umatinv);
        Real vol = umat[0] * umat[1 + DIM] * umat[2 + 2 * DIM];

        Prolate0Fun pf((double)beta, 5000);
        Real psi0 = (Real)pf.eval_val(0.0);
        Real bnrm2 = bloch[0] * bloch[0] + bloch[1] * bloch[1] + bloch[2] * bloch[2];
        Real dkmax = beta / rc + std::sqrt(bnrm2);
        Real anrm1 = umat[0];
        Real anrm2 = std::sqrt(umat[DIM] * umat[DIM] + umat[1 + DIM] * umat[1 + DIM]);
        Real anrm3 = std::sqrt(umat[2 * DIM] * umat[2 * DIM] + umat[1 + 2 * DIM] * umat[1 + 2 * DIM] +
                               umat[2 + 2 * DIM] * umat[2 + 2 * DIM]);
        int64_t ms = std::max<int64_t>(2, 2 * (int64_t)std::ceil(dkmax * anrm1 / twopi));
        int64_t mt = std::max<int64_t>(2, 2 * (int64_t)std::ceil(dkmax * anrm2 / twopi));
        int64_t mu = std::max<int64_t>(2, 2 * (int64_t)std::ceil(dkmax * anrm3 / twopi));
        int64_t nk = ms * mt * mu;

        // Coords: 2π L^{-1} x_j
        std::vector<double> coords((size_t)ns * 3);
        for (int j = 0; j < ns; ++j) {
            Real x = sources[j * DIM], y = sources[j * DIM + 1], z = sources[j * DIM + 2];
            Real xj = twopi * (umatinv[0] * x + umatinv[DIM] * y + umatinv[2 * DIM] * z);
            Real yj = twopi * (umatinv[1 + DIM] * y + umatinv[1 + 2 * DIM] * z);
            Real zj = twopi * (umatinv[2 + 2 * DIM] * z);
            coords[3 * (size_t)j + 0] = (double)zj;
            coords[3 * (size_t)j + 1] = (double)yj;
            coords[3 * (size_t)j + 2] = (double)xj;
        }

        // Charges phased by e^{-i β·x_j}
        std::vector<std::complex<double>> zcharge_d((size_t)ns);
        for (int j = 0; j < ns; ++j) {
            Real x = sources[j * DIM], y = sources[j * DIM + 1], z = sources[j * DIM + 2];
            Real bdoty = bloch[0] * x + bloch[1] * y + bloch[2] * z;
            zcharge_d[(size_t)j] = std::complex<double>((double)charge[j] * std::cos(-bdoty),
                                                         (double)charge[j] * std::sin(-bdoty));
        }

        // Diagonal weights at shifted wavenumbers 2π L^{-T} m + β
        Real a11 = Real(1) / umat[0];
        Real a21 = -umat[DIM] / (umat[0] * umat[1 + DIM]);
        Real a22 = Real(1) / umat[1 + DIM];
        Real a31 = (umat[DIM] * umat[1 + 2 * DIM] - umat[2 * DIM] * umat[1 + DIM]) /
                   (umat[0] * umat[1 + DIM] * umat[2 + 2 * DIM]);
        Real a32 = -umat[1 + 2 * DIM] / (umat[1 + DIM] * umat[2 + 2 * DIM]);
        Real a33 = Real(1) / umat[2 + 2 * DIM];
        Real dfac = Real(4) * Real(M_PI) / (psi0 * vol);

        std::vector<double> weights((size_t)nk, 0.0);
        int64_t nidx = 0;
        for (int64_t k = -mu / 2; k <= (mu - 1) / 2; k++) {
            for (int64_t j = -mt / 2; j <= (mt - 1) / 2; j++) {
                for (int64_t i = -ms / 2; i <= (ms - 1) / 2; i++) {
                    Real dk1 = twopi * (a11 * i) + bloch[0];
                    Real dk2 = twopi * (a21 * i + a22 * j) + bloch[1];
                    Real dk3 = twopi * (a31 * i + a32 * j + a33 * k) + bloch[2];
                    Real dk2sq = dk1 * dk1 + dk2 * dk2 + dk3 * dk3;
                    double w = 0.0;
                    if (dk2sq >= Real(1e-30)) {
                        Real dk = std::sqrt(dk2sq);
                        Real xval = dk * rc / beta;
                        Real fval = (xval <= Real(1)) ? (Real)pf.eval_val(xval) : Real(0);
                        w = (double)(dfac * fval / dk2sq);
                    }
                    weights[(size_t)nidx++] = w;
                }
            }
        }

        // NUFFT Type 1 (source → grid), diagonal scaling, NUFFT Type 2 (grid → source)
        std::vector<std::complex<double>> fk_d((size_t)nk);
        std::vector<std::complex<double>> zpot_d((size_t)ns);
        {
            using namespace ducc0;
            const std::vector<size_t> shape = {(size_t)mu, (size_t)mt, (size_t)ms};
            const std::vector<double> periodicity = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
            const double tol_d = (double)eps * 0.25;
            cmav<double, 2> coord_view(coords.data(), std::array<size_t, 2>{(size_t)ns, (size_t)3});

            Nufft<double, double, double> plan1(true, coord_view, shape, tol_d, 0, 1.1, 2.6,
                                                periodicity, false);
            Nufft<double, double, double> plan2(false, coord_view, shape, tol_d, 0, 1.1, 2.6,
                                                periodicity, false);

            cmav<std::complex<double>, 1> pts_cj(zcharge_d.data(), std::array<size_t, 1>{(size_t)ns});
            vfmav<std::complex<double>> uniform_v(fk_d.data(), shape);
            plan1.nu2u(true, 0, pts_cj, uniform_v);
            for (int64_t i = 0; i < nk; ++i)
                fk_d[(size_t)i] *= weights[(size_t)i];
            cfmav<std::complex<double>> uniform_c(fk_d.data(), shape);
            vmav<std::complex<double>, 1> pts_out(zpot_d.data(), std::array<size_t, 1>{(size_t)ns});
            plan2.u2nu(false, 0, uniform_c, pts_out);
        }

        // Restore physical-space potential: zpot_nufft(x) = e^{-i β·x} * ψ(x),
        // so ψ(x) = e^{+i β·x} * zpot_nufft(x). Output layout: (Re, Im) per source.
        for (int j = 0; j < ns; ++j) {
            Real x = sources[j * DIM], y = sources[j * DIM + 1], z = sources[j * DIM + 2];
            Real bdoty = bloch[0] * x + bloch[1] * y + bloch[2] * z;
            Real cph = std::cos(bdoty), sph = std::sin(bdoty);
            double zr = zpot_d[(size_t)j].real();
            double zi = zpot_d[(size_t)j].imag();
            pot[2 * j + 0] += (Real)(cph * zr - sph * zi);
            pot[2 * j + 1] += (Real)(sph * zr + cph * zi);
        }
        return;
    }

    // Per-thread plan cache — hit when (sources, ns, rc, eps, beta, umat,
    // backend) all match the previous call. Keeps the plans, the coord
    // array, and the diagonal weight table alive across invocations.
    thread_local NufftCache<Real, DIM> cache;

    const NufftBackend backend = pick_nufft_backend();
    const bool hit = cache.matches(sources, ns, rc, eps, beta, umat, backend);
    if (!hit) {
        // -------- (re)build cache --------
        cache.sources = sources;
        cache.ns = ns;
        cache.rc = rc;
        cache.eps = eps;
        cache.beta = beta;
        for (int i = 0; i < DIM * DIM; ++i) cache.umat_key[i] = umat[i];
#if PDMK_HAVE_FINUFFT
        cache.backend = backend;
#endif

        // Inverse lattice matrix + volume
        Real umatinv[DIM * DIM];
        utri_inv<Real, DIM>(umat, umatinv);
        Real vol = umat[0] * umat[1 + DIM] * umat[2 + 2 * DIM];

        // PSWF + grid sizes
        Prolate0Fun pf((double)beta, 5000);
        Real psi0 = (Real)pf.eval_val(0.0);
        Real dkmax = beta / rc;
        Real anrm1 = umat[0];
        Real anrm2 = std::sqrt(umat[DIM] * umat[DIM] + umat[1 + DIM] * umat[1 + DIM]);
        Real anrm3 = std::sqrt(umat[2 * DIM] * umat[2 * DIM] + umat[1 + 2 * DIM] * umat[1 + 2 * DIM] +
                               umat[2 + 2 * DIM] * umat[2 + 2 * DIM]);
        int64_t ms = std::max<int64_t>(2, 2 * (int64_t)std::ceil(dkmax * anrm1 / twopi));
        int64_t mt = std::max<int64_t>(2, 2 * (int64_t)std::ceil(dkmax * anrm2 / twopi));
        int64_t mu = std::max<int64_t>(2, 2 * (int64_t)std::ceil(dkmax * anrm3 / twopi));
        int64_t nk = ms * mt * mu;
        cache.ms = ms; cache.mt = mt; cache.mu = mu; cache.nk = nk;

        // Coord packing: {z, y, x} per point, matching ducc grid {mu, mt, ms}.
        cache.coords.resize((size_t)ns * 3);
        for (int j = 0; j < ns; ++j) {
            Real x = sources[j * DIM], y = sources[j * DIM + 1], z = sources[j * DIM + 2];
            Real xj = twopi * (umatinv[0] * x + umatinv[DIM] * y + umatinv[2 * DIM] * z);
            Real yj = twopi * (umatinv[1 + DIM] * y + umatinv[1 + 2 * DIM] * z);
            Real zj = twopi * (umatinv[2 + 2 * DIM] * z);
            cache.coords[3 * (size_t)j + 0] = (double)zj;
            cache.coords[3 * (size_t)j + 1] = (double)yj;
            cache.coords[3 * (size_t)j + 2] = (double)xj;
        }

        // Diagonal weights (kernel FT / V), same layout as ducc grid {mu, mt, ms}.
        Real a11 = Real(1) / umat[0];
        Real a21 = -umat[DIM] / (umat[0] * umat[1 + DIM]);
        Real a22 = Real(1) / umat[1 + DIM];
        Real a31 = (umat[DIM] * umat[1 + 2 * DIM] - umat[2 * DIM] * umat[1 + DIM]) /
                   (umat[0] * umat[1 + DIM] * umat[2 + 2 * DIM]);
        Real a32 = -umat[1 + 2 * DIM] / (umat[1 + DIM] * umat[2 + 2 * DIM]);
        Real a33 = Real(1) / umat[2 + 2 * DIM];
        Real dfac = Real(4) * Real(M_PI) / (psi0 * vol);

        cache.weights.assign((size_t)nk, 0.0);
        int64_t n = 0;
        for (int64_t k = -mu / 2; k <= (mu - 1) / 2; k++) {
            for (int64_t j = -mt / 2; j <= (mt - 1) / 2; j++) {
                for (int64_t i = -ms / 2; i <= (ms - 1) / 2; i++) {
                    Real dk1 = twopi * (a11 * i);
                    Real dk2 = twopi * (a21 * i + a22 * j);
                    Real dk3 = twopi * (a31 * i + a32 * j + a33 * k);
                    Real dk2sq = dk1 * dk1 + dk2 * dk2 + dk3 * dk3;
                    double w = 0.0;
                    if (dk2sq >= Real(1e-30)) {
                        Real dk = std::sqrt(dk2sq);
                        Real xval = dk * rc / beta;
                        Real fval = (xval <= Real(1)) ? (Real)pf.eval_val(xval) : Real(0);
                        w = (double)(dfac * fval / dk2sq);
                    }
                    cache.weights[(size_t)n++] = w;
                }
            }
        }

        const double tol_d = (double)eps * 0.25;
        if (backend == NufftBackend::Ducc) {
            // ducc0 plans. Construction is dominated by FFT planning and kernel
            // gridding setup — the same thing PeriodicDMK does once at
            // WindowPotential ctor.
            using namespace ducc0;
            const std::vector<size_t> shape = {(size_t)mu, (size_t)mt, (size_t)ms};
            const std::vector<double> periodicity = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
            cmav<double, 2> coord_view(cache.coords.data(), std::array<size_t,2>{(size_t)ns, (size_t)3});
            cache.plan1 = std::make_unique<Nufft<double, double, double>>(
                /*gridding=*/true, coord_view, shape, tol_d,
                /*nthreads=*/0, /*sigma_min=*/1.1, /*sigma_max=*/2.6, periodicity, /*fft_order=*/false);
            cache.plan2 = std::make_unique<Nufft<double, double, double>>(
                /*gridding=*/false, coord_view, shape, tol_d,
                /*nthreads=*/0, /*sigma_min=*/1.1, /*sigma_max=*/2.6, periodicity, /*fft_order=*/false);
        } else {
#if PDMK_HAVE_FINUFFT
            // FINUFFT wants separate x/y/z arrays. DUCC's `cache.coords` packs
            // {z,y,x} per point; unpack into cache.{fx,fy,fz}.
            cache.fx.resize((size_t)ns);
            cache.fy.resize((size_t)ns);
            cache.fz.resize((size_t)ns);
            for (int j = 0; j < ns; ++j) {
                cache.fz[(size_t)j] = cache.coords[3*(size_t)j + 0];
                cache.fy[(size_t)j] = cache.coords[3*(size_t)j + 1];
                cache.fx[(size_t)j] = cache.coords[3*(size_t)j + 2];
            }
            // Destroy any prior plans before rebuilding.
            if (cache.fplan1) { finufft_destroy(cache.fplan1); cache.fplan1 = nullptr; }
            if (cache.fplan2) { finufft_destroy(cache.fplan2); cache.fplan2 = nullptr; }

            finufft_opts opts;
            finufft_default_opts(&opts);
            opts.nthreads  = 0;       // use OMP default
            opts.modeord   = 0;       // CMCL: indices run k = -M/2 .. M/2-1 (matches DUCC layout)
            opts.upsampfac = 2.0;     // FFT oversampling
            int64_t n_modes[3] = {ms, mt, mu};  // {fastest, ..., slowest} in FINUFFT memory layout

            int ier;
            ier = finufft_makeplan(/*type=*/1, /*dim=*/3, n_modes, /*iflag=*/-1,
                                   /*ntrans=*/1, tol_d, &cache.fplan1, &opts);
            if (ier) throw std::runtime_error("FINUFFT makeplan type1 failed");
            ier = finufft_makeplan(/*type=*/2, /*dim=*/3, n_modes, /*iflag=*/+1,
                                   /*ntrans=*/1, tol_d, &cache.fplan2, &opts);
            if (ier) throw std::runtime_error("FINUFFT makeplan type2 failed");
            ier = finufft_setpts(cache.fplan1, (int64_t)ns,
                                 cache.fx.data(), cache.fy.data(), cache.fz.data(),
                                 0, nullptr, nullptr, nullptr);
            if (ier) throw std::runtime_error("FINUFFT setpts type1 failed");
            ier = finufft_setpts(cache.fplan2, (int64_t)ns,
                                 cache.fx.data(), cache.fy.data(), cache.fz.data(),
                                 0, nullptr, nullptr, nullptr);
            if (ier) throw std::runtime_error("FINUFFT setpts type2 failed");
#else
            throw std::runtime_error("FINUFFT backend requested but PDMK_HAVE_FINUFFT=0");
#endif
        }
    }

    const int64_t nk = cache.nk;

    // -------- per-call work: charge → NUFFT → weights → NUFFT → pot --------
    std::vector<std::complex<double>> zcharge_d((size_t)ns);
    std::vector<std::complex<double>> fk_d((size_t)nk);
    std::vector<std::complex<double>> zpot_d((size_t)ns);
    for (int j = 0; j < ns; ++j) zcharge_d[(size_t)j] = std::complex<double>((double)charge[j], 0.0);

    if (backend == NufftBackend::Ducc) {
        const std::vector<size_t> shape = {(size_t)cache.mu, (size_t)cache.mt, (size_t)cache.ms};
        using namespace ducc0;
        {
            cmav<std::complex<double>, 1> pts_cj(zcharge_d.data(), std::array<size_t,1>{(size_t)ns});
            vfmav<std::complex<double>> uniform(fk_d.data(), shape);
            cache.plan1->nu2u(/*forward=*/true, /*verbosity=*/0, pts_cj, uniform);
        }
        for (int64_t i = 0; i < nk; ++i)
            fk_d[(size_t)i] *= cache.weights[(size_t)i];
        {
            cfmav<std::complex<double>> uniform(fk_d.data(), shape);
            vmav<std::complex<double>, 1> pts_out(zpot_d.data(), std::array<size_t,1>{(size_t)ns});
            cache.plan2->u2nu(/*forward=*/false, /*verbosity=*/0, uniform, pts_out);
        }
    } else {
#if PDMK_HAVE_FINUFFT
        // FINUFFT's type-1 plan stored iflag=-1 (matching DUCC's nu2u
        // forward=true with exp(-i k·x) convention). type-2 uses iflag=+1.
        int ier;
        ier = finufft_execute(cache.fplan1,
                              reinterpret_cast<std::complex<double>*>(zcharge_d.data()),
                              reinterpret_cast<std::complex<double>*>(fk_d.data()));
        if (ier) throw std::runtime_error("FINUFFT execute type1 failed");
        for (int64_t i = 0; i < nk; ++i)
            fk_d[(size_t)i] *= cache.weights[(size_t)i];
        ier = finufft_execute(cache.fplan2,
                              reinterpret_cast<std::complex<double>*>(zpot_d.data()),
                              reinterpret_cast<std::complex<double>*>(fk_d.data()));
        if (ier) throw std::runtime_error("FINUFFT execute type2 failed");
#endif
    }

    for (int i = 0; i < ns; ++i)
        pot[i] += (Real)zpot_d[(size_t)i].real();

    // ============================================
    // Step 4: Self-interaction correction
    // ============================================
    // self_correction_value is W(0) for this kernel/dimension.
    // Subtract it to remove the spurious j=i contribution from the NUFFT.
    // In the 2D case, W(0) = sum of dkernelft (computed from the grid).
    // In the 3D case, W(0) = psi0/(c0*rc).
    // The caller provides this value.
    // NOTE: We do NOT apply it here; the caller (pbcdmk) handles all
    // self-correction in one place after combining near-field + far-field.
}

// ============================================================
// Main PBC-DMK entry point (single-level scheme)
//
// Algorithm (matches Fortran pbcdmk.f):
//   pot = near-field (residual kernel R_0 on rectangular grid)
//       + far-field  (periodized windowed kernel W_0 via NUFFT)
//       - self-correction W_0(0)*q_i
// ============================================================

template <typename Real, int DIM>
void pbcdmk(void *comm, const pdmk_params &params, const Real *avec, Real rc,
            int n_src, const Real *r_src, const Real *charge,
            const Real *normal, const Real *dipole_str, Real *pot,
            const Real *bloch) {
#ifdef DMK_HAVE_MPI
    const auto &sctl_comm = sctl::Comm(MPI_Comm(comm));
#else
    const auto &sctl_comm = sctl::Comm().Self();
#endif
    auto &logger = dmk::get_logger(sctl_comm, params.log_level);

    // rc <= 0 is a sentinel: auto-pick from the K-balance formula. See
    // include/dmk/pbc.hpp single_level_optimal_rc() for the derivation.
    if (rc <= Real(0)) {
        rc = single_level_optimal_rc<Real, DIM>(params, avec, n_src);
        logger->info("PBCDMK auto-selected rc={} (single-level K-balance)", rc);
    }
    logger->info("PBCDMK called: n_src={}, rc={}", n_src, rc);

    const bool qp = (bloch != nullptr);
    if (qp && params.kernel != DMK_LAPLACE)
        throw std::runtime_error("pbcdmk: QP mode currently supports only DMK_LAPLACE");
    const int nd_in = get_kernel_input_dim(DIM, params.kernel);
    const int nd = qp ? 2 : nd_in;  // downstream density count
    if (qp && nd_in != 1)
        throw std::runtime_error("pbcdmk: QP mode currently supports only scalar kernels (nd_in=1)");

    // Step 1: Lattice setup (QR → upper triangular)
    Real umat[DIM * DIM], qmat[DIM * DIM];
    lattice_setup<Real, DIM>(avec, umat, qmat);

    logger->info("umat diagonal: {}, {}{}", umat[0], umat[1 + DIM],
                 DIM == 3 ? (", " + std::to_string(umat[2 + 2 * DIM])) : "");

    // Step 2: Compute inner grid covering the unit cell
    int mgrid[DIM];
    Real xgrid0[DIM];
    compute_grid<Real, DIM>(umat, rc, mgrid, xgrid0);

    // Step 3: Generate periodic images (QP → image charges carry e^{+iβ·R} phase)
    int nsimg = generate_images<Real, DIM>(umat, rc, mgrid, xgrid0, n_src, r_src, nd_in, charge,
                                            nullptr, nullptr, bloch);
    logger->info("Image sources: {} (ratio: {:.2f}x)", nsimg, (double)(n_src + nsimg) / n_src);

    std::vector<Real> srcimg(DIM * std::max(nsimg, 1));
    std::vector<Real> chgimg(nd * std::max(nsimg, 1));
    if (nsimg > 0)
        generate_images<Real, DIM>(umat, rc, mgrid, xgrid0, n_src, r_src, nd_in, charge,
                                    srcimg.data(), chgimg.data(), bloch);

    // Step 4: Concatenate original + image sources.
    // For QP, original sources get (Re, Im) = (q, 0); images already have (Re, Im).
    const int nsall = n_src + nsimg;
    std::vector<Real> srcall(DIM * nsall), chgall(nd * nsall);
    std::memcpy(srcall.data(), r_src, n_src * DIM * sizeof(Real));
    if (qp) {
        for (int i = 0; i < n_src; i++) {
            chgall[i * 2 + 0] = charge[i];
            chgall[i * 2 + 1] = 0;
        }
    } else {
        std::memcpy(chgall.data(), charge, n_src * nd * sizeof(Real));
    }
    if (nsimg > 0) {
        std::memcpy(srcall.data() + n_src * DIM, srcimg.data(), nsimg * DIM * sizeof(Real));
        std::memcpy(chgall.data() + n_src * nd, chgimg.data(), nsimg * nd * sizeof(Real));
    }

    // Step 5: Set up extended rectangular grid
    int mle[DIM];
    Real xelo[DIM];
    for (int i = 0; i < DIM; i++) {
        mle[i] = mgrid[i] + 2;
        xelo[i] = xgrid0[i] - rc;
    }
    constexpr int ncoll_max = (DIM == 2) ? 9 : 27;
    int nboxes = mle[0] * mle[1];
    if constexpr (DIM == 3)
        nboxes *= mle[2];

    std::vector<Real> centers(DIM * nboxes);
    std::vector<int> colleagues(ncoll_max * nboxes);
    rect_grid_setup<Real, DIM>(mgrid, rc, xgrid0, centers.data(), colleagues.data());

    logger->info("Extended grid: {}x{}{}, {} boxes", mle[0], mle[1],
                 DIM == 3 ? ("x" + std::to_string(mle[2])) : "", nboxes);

    // Step 6: Bin sort all sources into grid
    std::vector<int> isrc(nsall), isrcse(2 * nboxes);
    bin_sort<Real, DIM>(mle, rc, xelo, nboxes, nsall, srcall.data(), isrc.data(), isrcse.data());

    // Reorder into sorted arrays
    std::vector<Real> srcsort(DIM * nsall), chgsort(nd * nsall);
    for (int i = 0; i < nsall; i++) {
        const int idx = isrc[i];
        for (int d = 0; d < DIM; d++)
            srcsort[i * DIM + d] = srcall[idx * DIM + d];
        for (int k = 0; k < nd; k++)
            chgsort[i * nd + k] = chgall[idx * nd + k];
    }

    // Inverse permutation for extracting original source potentials
    std::vector<int> isrcinv(nsall);
    for (int i = 0; i < nsall; i++)
        isrcinv[isrc[i]] = i;

    // Step 7: Near-field evaluation via existing direct evaluator
    const int n_digits = std::max(1, (int)std::round(log10(1.0 / params.eps) - 0.1));
    auto evaluator = make_evaluator_aot<Real>(params.kernel, DMK_POTENTIAL, DIM, n_digits, 3);

    // Evaluator parameters: map r ∈ [0, rc] → x ∈ [-1, 1] via x = r*rsc + cen
    Real rsc = Real(2) / rc;
    Real cen = -rc / Real(2);
    if ((params.kernel == DMK_SQRT_LAPLACE && DIM == 3) || (params.kernel == DMK_LAPLACE && DIM == 2)) {
        rsc = Real(2) / (rc * rc);
        cen = Real(-1);
    } else if (params.kernel == DMK_YUKAWA) {
        cen = Real(-1);
    }
    const Real d2max = rc * rc;
    const Real thresh2 = Real(1e-30);

    std::vector<Real> potsort(nd * nsall, Real(0));

    // Workspace for collecting colleague sources. For nd>1 (QP), we run the
    // box loop once per density extracting chg via stride-nd, and the evaluator
    // sees a plain real charge array.
    int max_per_box = nsall / std::max(nboxes, 1) + 100;
    int coll_buf_size = ncoll_max * max_per_box;
    std::vector<Real> coll_src(DIM * coll_buf_size);
    std::vector<Real> coll_chg_k(coll_buf_size);
    std::vector<Real> potsort_k(nsall);

    for (int k = 0; k < nd; k++) {
        std::fill(potsort_k.begin(), potsort_k.end(), Real(0));

        for (int ibox = 0; ibox < nboxes; ibox++) {
            const int ist = isrcse[2 * ibox];
            const int ien = isrcse[2 * ibox + 1];
            if (ist > ien) continue;
            const int n_trg = ien - ist + 1;

            int n_coll_src = 0;
            for (int ic = 0; ic < ncoll_max; ic++) {
                const int jbox = colleagues[ibox * ncoll_max + ic];
                if (jbox < 0) continue;
                const int jst = isrcse[2 * jbox];
                const int jen = isrcse[2 * jbox + 1];
                const int npts = jen - jst + 1;
                if (npts <= 0) continue;

                if (n_coll_src + npts > coll_buf_size) {
                    coll_buf_size = 2 * (n_coll_src + npts);
                    coll_src.resize(DIM * coll_buf_size);
                    coll_chg_k.resize(coll_buf_size);
                }
                std::memcpy(coll_src.data() + n_coll_src * DIM, srcsort.data() + jst * DIM,
                            npts * DIM * sizeof(Real));
                for (int p = 0; p < npts; p++)
                    coll_chg_k[n_coll_src + p] = chgsort[(jst + p) * nd + k];
                n_coll_src += npts;
            }
            if (n_coll_src == 0) continue;

            evaluator(rsc, cen, d2max, thresh2, n_coll_src, coll_src.data(), coll_chg_k.data(), n_trg,
                      srcsort.data() + ist * DIM, potsort_k.data() + ist);
        }

        for (int i = 0; i < nsall; i++)
            potsort[i * nd + k] = potsort_k[i];
    }

    // Extract potentials at original source positions (indices 0..n_src-1 in unsorted order)
    for (int i = 0; i < n_src; i++) {
        const int sidx = isrcinv[i];
        for (int k = 0; k < nd; k++)
            pot[i * nd + k] = potsort[sidx * nd + k];
    }

    logger->info("Near-field evaluation complete");

    // Step 8: Far-field via NUFFT (periodized windowed kernel W_0^{(p)})
    const Real beta = (Real)dmk::util::calc_bandlimiting(params);
    far_field_nufft<Real, DIM>(umat, rc, (Real)params.eps, beta, n_src, r_src, charge, pot, 0, bloch);

    // Step 9: Self-interaction correction — subtract W_0(0)*q_i
    Prolate0Fun pf((double)beta, 5000);
    const Real psi0 = (Real)pf.eval_val(0.0);
    Real sc = 0;
    if (params.kernel == DMK_LAPLACE && DIM == 3) {
        // W_0(0) = psi0 / (c0 * rc) for 3D Laplace
        auto cvals = pf.intvals((double)beta);
        sc = psi0 / ((Real)cvals[0] * rc);
    } else if (params.kernel == DMK_SQRT_LAPLACE && DIM == 2) {
        auto cvals = pf.intvals((double)beta);
        sc = psi0 / ((Real)cvals[0] * rc);
    } else if (params.kernel == DMK_SQRT_LAPLACE && DIM == 3) {
        auto cvals = pf.intvals((double)beta);
        sc = psi0 / (Real(2) * (Real)cvals[1] * rc * rc);
    } else if (params.kernel == DMK_LAPLACE && DIM == 2) {
        sc = compute_far_field_self_correction_2d<Real>(umat, rc, beta);
    }

    // Original source charges: real in non-QP; (q, 0) in QP.
    // Non-QP: pot[i] -= sc * charge[i].
    // QP:     pot[2*i+0] -= sc * charge[i];  pot[2*i+1] -= 0.
    if (qp) {
        for (int i = 0; i < n_src; i++)
            pot[2 * i + 0] -= sc * charge[i];
    } else {
        for (int i = 0; i < n_src; i++)
            pot[i] -= sc * charge[i];
    }

    logger->info("PBCDMK finished");
}

// ============================================================
// Adaptive (multilevel) PBC-DMK
// ============================================================

template <typename Real, int DIM>
void pbcdmk_adaptive(const pdmk_params &params, const Real *avec, Real rc,
                     int n_src, const Real *r_src, const Real *charge, Real *pot,
                     const Real *bloch, PbcAdaptiveTiming *timing) {
    auto logger = spdlog::default_logger();
    logger->info("pbcdmk_adaptive: n_src={}, rc={}", n_src, rc);

    const bool qp = (bloch != nullptr);
    if (qp && params.kernel != DMK_LAPLACE)
        throw std::runtime_error("pbcdmk_adaptive: QP currently supports only DMK_LAPLACE");

    // Match Fortran pbcdmk_adaptive: evaluate at the N original sources treated
    // as targets with ifpgh=0 (no source-side eval, so no work at image sources).
    pdmk_params params_t = params;
    params_t.pgh_src = (dmk_pgh)0;
    if (params_t.pgh_trg == (dmk_pgh)0) params_t.pgh_trg = DMK_POTENTIAL;

    const bool dbg_prof = std::getenv("DMK_PBC_PROFILE") != nullptr;
    auto tA = std::chrono::steady_clock::now();

    // Build combined (original + image) source/charge arrays outside the tree.
    // QP → image charges carry e^{+i β·R_n} phase and nd is promoted to 2.
    const int nd_in = 1;  // Laplace scalar
    std::vector<Real> srcall, chgall;
    int nsall = build_combined_sources<Real, DIM>(avec, rc, n_src, r_src, nd_in, charge,
                                                   srcall, chgall, bloch);
    const int nd_override = qp ? 2 * nd_in : -1;

    dmk::PBCTree<Real, DIM> tree(params_t, avec, rc, nsall, srcall.data(), n_src, r_src,
                                  chgall.data(), nd_override);
    auto tB = std::chrono::steady_clock::now();
    // Hand-unrolled equivalent of tree.eval() with the leaf-level direct
    // step timed separately. tree.eval() = upward_pass + downward_pass,
    // and downward_pass = downward_prologue + form_downward_expansions
    // + evaluate_direct_interactions. Splitting the direct step out lets
    // PbcAdaptiveTiming.direct_ms match the 3D EvalTiming.direct_ms
    // semantics (leaf-level near-field cost rather than just bookkeeping).
    tree.upward_pass();
    tree.downward_prologue();
    tree.form_downward_expansions();
    auto tCmli = std::chrono::steady_clock::now();
    tree.evaluate_direct_interactions();
    auto tC = std::chrono::steady_clock::now();

    // Self-correction on targets: for non-QP, pass the original real charges;
    // for QP, pass the (Re, Im) pair per target. Original targets = original
    // sources, so original charges have (q, 0) per density.
    if (qp) {
        std::vector<Real> trg_chg_qp((std::size_t)2 * n_src);
        for (int i = 0; i < n_src; i++) {
            trg_chg_qp[2 * i + 0] = charge[i];
            trg_chg_qp[2 * i + 1] = 0;
        }
        tree.apply_target_self_correction(trg_chg_qp.data());
    } else {
        tree.apply_target_self_correction(charge);
    }

    // Extract pot at original target positions (= original source positions).
    const int nd_out = tree.kernel_output_dim_trg;  // 1 non-QP, 2 QP
    for (int i = 0; i < n_src * nd_out; i++) pot[i] = 0;
    for (int i = 0; i < (int)tree.trg_perm.size(); i++) {
        int orig = tree.trg_perm[i];
        for (int k = 0; k < nd_out; k++)
            pot[orig * nd_out + k] = tree.pot_trg_sorted[i * nd_out + k];
    }

    // Add NUFFT far-field with ifself=0 (tree already handled self-correction)
    Real umat[DIM * DIM], qmat[DIM * DIM];
    lattice_setup<Real, DIM>(avec, umat, qmat);
    const Real beta = (Real)dmk::util::calc_bandlimiting(params);
    auto tD = std::chrono::steady_clock::now();
    far_field_nufft<Real, DIM>(umat, rc, (Real)params.eps, beta, n_src, r_src, charge, pot, 0, bloch);
    auto tE = std::chrono::steady_clock::now();

    double t_build  = std::chrono::duration<double, std::milli>(tB - tA).count();
    double t_mli    = std::chrono::duration<double, std::milli>(tCmli - tB).count();
    double t_direct = std::chrono::duration<double, std::milli>(tC - tCmli).count();
    double t_self   = std::chrono::duration<double, std::milli>(tD - tC).count();
    double t_nufft  = std::chrono::duration<double, std::milli>(tE - tD).count();
    if (timing) {
        timing->build_ms       = t_build;
        timing->eval_ms        = t_mli;
        timing->direct_ms      = t_direct;
        timing->self_desort_ms = t_self;
        timing->nufft_ms       = t_nufft;
    }
    if (dbg_prof) {
        logger->warn("  [prof] pbcdmk_adaptive: build={:.1f}  eval={:.1f}  direct={:.1f}  self+desort={:.1f}  nufft={:.1f} ms",
                     t_build, t_mli, t_direct, t_self, t_nufft);
    }

    logger->info("pbcdmk_adaptive finished");
}

// Explicit instantiations
template void far_field_nufft<float, 2>(const float *, float, float, float, int, const float *, const float *, float *,
                                        float, const float *);
template void far_field_nufft<float, 3>(const float *, float, float, float, int, const float *, const float *, float *,
                                        float, const float *);
template void far_field_nufft<double, 2>(const double *, double, double, double, int, const double *, const double *,
                                         double *, double, const double *);
template void far_field_nufft<double, 3>(const double *, double, double, double, int, const double *, const double *,
                                         double *, double, const double *);

template void pbcdmk<float, 2>(void *, const pdmk_params &, const float *, float, int, const float *, const float *,
                                const float *, const float *, float *, const float *);
template void pbcdmk<float, 3>(void *, const pdmk_params &, const float *, float, int, const float *, const float *,
                                const float *, const float *, float *, const float *);
template void pbcdmk<double, 2>(void *, const pdmk_params &, const double *, double, int, const double *,
                                 const double *, const double *, const double *, double *, const double *);
template void pbcdmk<double, 3>(void *, const pdmk_params &, const double *, double, int, const double *,
                                 const double *, const double *, const double *, double *, const double *);

template void pbcdmk_adaptive<float, 2>(const pdmk_params &, const float *, float, int, const float *, const float *,
                                        float *, const float *, PbcAdaptiveTiming *);
template void pbcdmk_adaptive<float, 3>(const pdmk_params &, const float *, float, int, const float *, const float *,
                                        float *, const float *, PbcAdaptiveTiming *);
template void pbcdmk_adaptive<double, 2>(const pdmk_params &, const double *, double, int, const double *,
                                          const double *, double *, const double *, PbcAdaptiveTiming *);
template void pbcdmk_adaptive<double, 3>(const pdmk_params &, const double *, double, int, const double *,
                                          const double *, double *, const double *, PbcAdaptiveTiming *);

} // namespace pbc
} // namespace dmk
