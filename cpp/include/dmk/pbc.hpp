#ifndef DMK_PBC_HPP
#define DMK_PBC_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <dmk.h>

namespace dmk {
namespace pbc {

/// QR factorization of lattice vectors to upper-triangular form.
/// Reorders columns by decreasing magnitude, then modified Gram-Schmidt.
/// avec: column-major dim x dim lattice vectors
/// umat: output upper-triangular matrix
/// qmat: output orthogonal rotation matrix (umat = qmat^T * avec_sorted)
template <typename Real, int DIM>
void lattice_setup(const Real *avec, Real *umat, Real *qmat) {
    std::array<Real, 3> anorm;
    std::array<int, 3> iord;
    for (int i = 0; i < DIM; i++) {
        Real s = 0;
        for (int k = 0; k < DIM; k++)
            s += avec[k + i * DIM] * avec[k + i * DIM];
        anorm[i] = std::sqrt(s);
        iord[i] = i;
    }

    for (int i = 0; i < DIM - 1; i++)
        for (int j = i + 1; j < DIM; j++)
            if (anorm[j] > anorm[i]) {
                std::swap(anorm[i], anorm[j]);
                std::swap(iord[i], iord[j]);
            }

    Real asort[9];
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
            asort[j + i * DIM] = avec[j + iord[i] * DIM];

    Real r11 = anorm[0];
    for (int i = 0; i < DIM; i++)
        qmat[i] = asort[i] / r11;

    Real r12 = 0;
    for (int i = 0; i < DIM; i++)
        r12 += qmat[i] * asort[i + DIM];
    for (int i = 0; i < DIM; i++)
        qmat[i + DIM] = asort[i + DIM] - r12 * qmat[i];
    Real r22 = 0;
    for (int i = 0; i < DIM; i++)
        r22 += qmat[i + DIM] * qmat[i + DIM];
    r22 = std::sqrt(r22);
    for (int i = 0; i < DIM; i++)
        qmat[i + DIM] /= r22;
    if (r22 < 0) {
        r22 = -r22;
        r12 = -r12;
        for (int i = 0; i < DIM; i++)
            qmat[i + DIM] = -qmat[i + DIM];
    }

    Real r13 = 0, r23 = 0, r33 = 0;
    if constexpr (DIM == 3) {
        for (int i = 0; i < 3; i++) {
            r13 += qmat[i] * asort[i + 2 * DIM];
            r23 += qmat[i + DIM] * asort[i + 2 * DIM];
        }
        for (int i = 0; i < 3; i++)
            qmat[i + 2 * DIM] = asort[i + 2 * DIM] - r13 * qmat[i] - r23 * qmat[i + DIM];
        for (int i = 0; i < 3; i++)
            r33 += qmat[i + 2 * DIM] * qmat[i + 2 * DIM];
        r33 = std::sqrt(r33);
        for (int i = 0; i < 3; i++)
            qmat[i + 2 * DIM] /= r33;
        if (r33 < 0) {
            r33 = -r33;
            r13 = -r13;
            r23 = -r23;
            for (int i = 0; i < 3; i++)
                qmat[i + 2 * DIM] = -qmat[i + 2 * DIM];
        }
    }

    for (int i = 0; i < DIM * DIM; i++)
        umat[i] = 0;
    umat[0] = r11;
    umat[DIM] = r12;
    umat[1 + DIM] = r22;
    if constexpr (DIM == 3) {
        umat[2 * DIM] = r13;
        umat[1 + 2 * DIM] = r23;
        umat[2 + 2 * DIM] = r33;
    }
}

/// Compute grid dimensions and origin for covering the unit cell.
template <typename Real, int DIM>
void compute_grid(const Real *umat, Real rc, int *mgrid, Real *xgrid0) {
    for (int i = DIM - 1; i >= 0; i--) {
        Real xmin = 0, xmax = umat[i + i * DIM];
        for (int j = i + 1; j < DIM; j++) {
            Real v = umat[i + j * DIM];
            xmin += std::min(Real(0), v);
            xmax += std::max(Real(0), v);
        }
        Real w = xmax - xmin;
        mgrid[i] = std::max(1, (int)std::ceil(w / rc));
        xgrid0[i] = xmin;
    }
}

/// Inverse of upper-triangular DIM x DIM matrix.
template <typename Real, int DIM>
void utri_inv(const Real *u, Real *v) {
    for (int i = 0; i < DIM * DIM; i++)
        v[i] = 0;
    v[0] = Real(1) / u[0];
    v[1 + DIM] = Real(1) / u[1 + DIM];
    v[DIM] = -u[DIM] / (u[0] * u[1 + DIM]);
    if constexpr (DIM == 3) {
        v[2 + 2 * DIM] = Real(1) / u[2 + 2 * DIM];
        v[1 + 2 * DIM] = -u[1 + 2 * DIM] / (u[1 + DIM] * u[2 + 2 * DIM]);
        v[2 * DIM] = (u[DIM] * u[1 + 2 * DIM] - u[2 * DIM] * u[1 + DIM])
                      / (u[0] * u[1 + DIM] * u[2 + 2 * DIM]);
    }
}

/// Count or fill image sources. If `bloch == nullptr`, fully periodic
/// (Bloch vector = 0) and output `chgimg` has `nd` reals per image, copied
/// from the source charge. If `bloch != nullptr`, quasi-periodic: each
/// image charge is multiplied by `e^{+i bloch · R_n}` where R_n is the
/// lattice shift. For QP the function assumes `nd == 1` (scalar Laplace)
/// and writes output `chgimg` with 2 reals per image: (Re, Im).
/// If srcimg/chgimg are nullptr, only counts. Returns number of images.
template <typename Real, int DIM>
int generate_images(const Real *umat, Real rc, const int *mgrid, const Real *xgrid0,
                    int ns, const Real *sources, int nd, const Real *charge,
                    Real *srcimg, Real *chgimg,
                    const Real *bloch = nullptr) {
    Real xextlo[3], xexthi[3], xlo[3], xhi[3];
    for (int i = 0; i < DIM; i++) {
        xextlo[i] = xgrid0[i] - rc;
        xexthi[i] = xgrid0[i] + (mgrid[i] + 1) * rc;
    }
    for (int i = 0; i < DIM; i++) {
        xlo[i] = xgrid0[i];
        Real w = umat[i + i * DIM];
        for (int j = i + 1; j < DIM; j++)
            w += std::abs(umat[i + j * DIM]);
        xhi[i] = xgrid0[i] + w;
    }

    const bool fill = (srcimg != nullptr);
    const bool qp = (bloch != nullptr);
    int nsimg = 0;

    if constexpr (DIM == 2) {
        int n2max = (int)((mgrid[1] + 1) * rc / umat[1 + DIM]) + 1;
        for (int n2 = -n2max; n2 <= n2max; n2++) {
            Real shift2 = umat[1 + DIM] * n2;
            if (xlo[1] + shift2 > xexthi[1] || xhi[1] + shift2 < xextlo[1])
                continue;
            Real off12 = umat[DIM] * n2;
            int n1max = (int)(((mgrid[0] + 1) * rc + std::abs(off12)) / umat[0]) + 1;
            for (int n1 = -n1max; n1 <= n1max; n1++) {
                if (n1 == 0 && n2 == 0)
                    continue;
                Real shift1 = umat[0] * n1 + off12;
                if (xlo[0] + shift1 > xexthi[0] || xhi[0] + shift1 < xextlo[0])
                    continue;
                Real cph = 1, sph = 0;
                if (qp) {
                    Real phase = bloch[0] * shift1 + bloch[1] * shift2;
                    cph = std::cos(phase);
                    sph = std::sin(phase);
                }
                for (int j = 0; j < ns; j++) {
                    Real x0 = sources[j * DIM] + shift1;
                    Real x1 = sources[j * DIM + 1] + shift2;
                    if (x0 < xextlo[0] || x0 > xexthi[0] || x1 < xextlo[1] || x1 > xexthi[1])
                        continue;
                    if (fill) {
                        srcimg[nsimg * DIM] = x0;
                        srcimg[nsimg * DIM + 1] = x1;
                        if (qp) {
                            chgimg[nsimg * 2 + 0] = charge[j] * cph;
                            chgimg[nsimg * 2 + 1] = charge[j] * sph;
                        } else {
                            for (int k = 0; k < nd; k++)
                                chgimg[nsimg * nd + k] = charge[j * nd + k];
                        }
                    }
                    nsimg++;
                }
            }
        }
    } else if constexpr (DIM == 3) {
        int n3max = (int)((mgrid[2] + 1) * rc / umat[2 + 2 * DIM]) + 1;
        for (int n3 = -n3max; n3 <= n3max; n3++) {
            Real shift3 = umat[2 + 2 * DIM] * n3;
            if (xlo[2] + shift3 > xexthi[2] || xhi[2] + shift3 < xextlo[2])
                continue;
            int n2max = (int)(((mgrid[1] + 1) * rc + std::abs(umat[1 + 2 * DIM] * n3)) / umat[1 + DIM]) + 1;
            for (int n2 = -n2max; n2 <= n2max; n2++) {
                Real shift2 = umat[1 + DIM] * n2 + umat[1 + 2 * DIM] * n3;
                if (xlo[1] + shift2 > xexthi[1] || xhi[1] + shift2 < xextlo[1])
                    continue;
                Real off12 = umat[DIM] * n2 + umat[2 * DIM] * n3;
                int n1max = (int)(((mgrid[0] + 1) * rc + std::abs(off12)) / umat[0]) + 1;
                for (int n1 = -n1max; n1 <= n1max; n1++) {
                    if (n1 == 0 && n2 == 0 && n3 == 0)
                        continue;
                    Real shift1 = umat[0] * n1 + off12;
                    if (xlo[0] + shift1 > xexthi[0] || xhi[0] + shift1 < xextlo[0])
                        continue;
                    Real cph = 1, sph = 0;
                    if (qp) {
                        Real phase = bloch[0] * shift1 + bloch[1] * shift2 + bloch[2] * shift3;
                        cph = std::cos(phase);
                        sph = std::sin(phase);
                    }
                    for (int j = 0; j < ns; j++) {
                        Real x0 = sources[j * DIM] + shift1;
                        Real x1 = sources[j * DIM + 1] + shift2;
                        Real x2 = sources[j * DIM + 2] + shift3;
                        if (x0 < xextlo[0] || x0 > xexthi[0] ||
                            x1 < xextlo[1] || x1 > xexthi[1] ||
                            x2 < xextlo[2] || x2 > xexthi[2])
                            continue;
                        if (fill) {
                            srcimg[nsimg * DIM] = x0;
                            srcimg[nsimg * DIM + 1] = x1;
                            srcimg[nsimg * DIM + 2] = x2;
                            if (qp) {
                                chgimg[nsimg * 2 + 0] = charge[j] * cph;
                                chgimg[nsimg * 2 + 1] = charge[j] * sph;
                            } else {
                                for (int k = 0; k < nd; k++)
                                    chgimg[nsimg * nd + k] = charge[j * nd + k];
                            }
                        }
                        nsimg++;
                    }
                }
            }
        }
    }
    return nsimg;
}

/// Set up the extended rectangular grid for PBC.
/// The inner grid has mgrid[i] boxes; the extended grid adds one layer
/// on each side, yielding (mgrid[i]+2) boxes per direction.
/// Returns total number of boxes in the extended grid.
/// centers: column-major (DIM x nboxes), box center positions.
/// colleagues: (ncoll_max x nboxes), colleague list per box (0 = no colleague).
///   ncoll_max = 3^DIM.
template <typename Real, int DIM>
int rect_grid_setup(const int *mgrid, Real rc, const Real *xgrid0,
                    Real *centers, int *colleagues) {
    int mle[3]; // extended grid dimensions
    Real xelo[3]; // extended grid origin
    for (int i = 0; i < DIM; i++) {
        mle[i] = mgrid[i] + 2;
        xelo[i] = xgrid0[i] - rc;
    }

    int nboxes = mle[0] * mle[1];
    if constexpr (DIM == 3)
        nboxes *= mle[2];

    constexpr int ncoll_max = DIM == 2 ? 9 : 27;
    const int ml1 = mle[0], ml2 = mle[1];

    // Compute centers and colleague lists
    if constexpr (DIM == 2) {
        int ibox = 0;
        for (int iy = 0; iy < ml2; iy++) {
            for (int ix = 0; ix < ml1; ix++) {
                centers[ibox * DIM + 0] = xelo[0] + (ix + Real(0.5)) * rc;
                centers[ibox * DIM + 1] = xelo[1] + (iy + Real(0.5)) * rc;

                int id = 0;
                for (int ky = -1; ky <= 1; ky++) {
                    int jy = iy + ky;
                    for (int kx = -1; kx <= 1; kx++) {
                        int jx = ix + kx;
                        if (jx < 0 || jx >= ml1 || jy < 0 || jy >= ml2)
                            colleagues[ibox * ncoll_max + id] = -1;
                        else
                            colleagues[ibox * ncoll_max + id] = jy * ml1 + jx;
                        id++;
                    }
                }
                ibox++;
            }
        }
    } else {
        const int ml3 = mle[2];
        const int ml12 = ml1 * ml2;
        int ibox = 0;
        for (int iz = 0; iz < ml3; iz++) {
            for (int iy = 0; iy < ml2; iy++) {
                for (int ix = 0; ix < ml1; ix++) {
                    centers[ibox * DIM + 0] = xelo[0] + (ix + Real(0.5)) * rc;
                    centers[ibox * DIM + 1] = xelo[1] + (iy + Real(0.5)) * rc;
                    centers[ibox * DIM + 2] = xelo[2] + (iz + Real(0.5)) * rc;

                    int id = 0;
                    for (int kz = -1; kz <= 1; kz++) {
                        int jz = iz + kz;
                        for (int ky = -1; ky <= 1; ky++) {
                            int jy = iy + ky;
                            for (int kx = -1; kx <= 1; kx++) {
                                int jx = ix + kx;
                                if (jx < 0 || jx >= ml1 || jy < 0 || jy >= ml2 ||
                                    jz < 0 || jz >= ml3)
                                    colleagues[ibox * ncoll_max + id] = -1;
                                else
                                    colleagues[ibox * ncoll_max + id] =
                                        jz * ml12 + jy * ml1 + jx;
                                id++;
                            }
                        }
                    }
                    ibox++;
                }
            }
        }
    }
    return nboxes;
}

/// Sort ns points into a rectangular grid.
/// Returns permutation isrc[k] = original index of k-th sorted point,
/// and isrcse[2*ibox], isrcse[2*ibox+1] = start/end range per box (inclusive).
template <typename Real, int DIM>
void bin_sort(const int *mgrid_ext, Real rc, const Real *xelo,
              int nboxes, int ns, const Real *sources,
              int *isrc, int *isrcse) {
    const int ml1 = mgrid_ext[0], ml2 = mgrid_ext[1];
    const int ml3 = (DIM == 3) ? mgrid_ext[2] : 1;
    const Real rcinv = Real(1) / rc;

    // Count particles per box
    std::vector<int> ncount(nboxes, 0);
    std::vector<int> iboxof(ns);

    for (int i = 0; i < ns; i++) {
        int ii[3];
        for (int k = 0; k < DIM; k++) {
            ii[k] = (int)((sources[i * DIM + k] - xelo[k]) * rcinv);
            if (ii[k] < 0) ii[k] = 0;
            if (ii[k] >= mgrid_ext[k]) ii[k] = mgrid_ext[k] - 1;
        }
        int ibox;
        if constexpr (DIM == 2)
            ibox = ii[1] * ml1 + ii[0];
        else
            ibox = ii[2] * ml1 * ml2 + ii[1] * ml1 + ii[0];
        iboxof[i] = ibox;
        ncount[ibox]++;
    }

    // Prefix sum → start/end per box
    std::vector<int> ip(nboxes + 1);
    ip[0] = 0;
    for (int ibox = 0; ibox < nboxes; ibox++) {
        ip[ibox + 1] = ip[ibox] + ncount[ibox];
        isrcse[2 * ibox] = ip[ibox];        // start (inclusive)
        isrcse[2 * ibox + 1] = ip[ibox + 1] - 1; // end (inclusive), -1 if empty
    }

    // Reset ip as running counters
    for (int ibox = 0; ibox < nboxes; ibox++)
        ip[ibox] = isrcse[2 * ibox];

    // Fill permutation
    for (int i = 0; i < ns; i++) {
        int ibox = iboxof[i];
        isrc[ip[ibox]] = i;
        ip[ibox]++;
    }
}

/// Build combined (original + image) source/charge arrays for PBCTree input.
/// Non-QP path (bloch == nullptr):
///   srcall has DIM × nsall reals, chgall has nd_in × nsall reals, image
///   charges are plain copies of the original charge.
/// QP path (bloch != nullptr):
///   chgall has 2·nd_in × nsall reals; each density becomes a (Re, Im)
///   pair per image, phased by e^{+i β·R_n}. Originals have (q, 0) per
///   density. Caller should pass nd_override = 2·nd_in to PBCTree.
/// Returns total nsall = n_src + n_img.
template <typename Real, int DIM>
int build_combined_sources(const Real *avec, Real rc,
                           int n_src, const Real *r_src,
                           int nd_in, const Real *charge,
                           std::vector<Real> &srcall, std::vector<Real> &chgall,
                           const Real *bloch = nullptr) {
    Real umat[DIM * DIM], qmat[DIM * DIM];
    lattice_setup<Real, DIM>(avec, umat, qmat);
    int mgrid[DIM];
    Real xgrid0[DIM];
    compute_grid<Real, DIM>(umat, rc, mgrid, xgrid0);

    const bool qp = (bloch != nullptr);
    const int nd_out = qp ? 2 * nd_in : nd_in;

    int nsimg = generate_images<Real, DIM>(umat, rc, mgrid, xgrid0,
                                            n_src, r_src, nd_in, charge,
                                            nullptr, nullptr, bloch);
    std::vector<Real> srcimg((std::size_t)DIM * std::max(nsimg, 1));
    std::vector<Real> chgimg((std::size_t)nd_out * std::max(nsimg, 1));
    if (nsimg > 0)
        generate_images<Real, DIM>(umat, rc, mgrid, xgrid0,
                                    n_src, r_src, nd_in, charge,
                                    srcimg.data(), chgimg.data(), bloch);

    const int nsall = n_src + nsimg;
    srcall.resize((std::size_t)DIM * nsall);
    chgall.resize((std::size_t)nd_out * nsall);
    std::memcpy(srcall.data(), r_src, (std::size_t)n_src * DIM * sizeof(Real));
    if (qp) {
        for (int i = 0; i < n_src; i++) {
            for (int k = 0; k < nd_in; k++) {
                chgall[(std::size_t)i * nd_out + 2 * k + 0] = charge[(std::size_t)i * nd_in + k];
                chgall[(std::size_t)i * nd_out + 2 * k + 1] = 0;
            }
        }
    } else {
        std::memcpy(chgall.data(), charge, (std::size_t)n_src * nd_out * sizeof(Real));
    }
    if (nsimg > 0) {
        std::memcpy(srcall.data() + (std::size_t)n_src * DIM, srcimg.data(),
                    (std::size_t)nsimg * DIM * sizeof(Real));
        std::memcpy(chgall.data() + (std::size_t)n_src * nd_out, chgimg.data(),
                    (std::size_t)nsimg * nd_out * sizeof(Real));
    }
    return nsall;
}

/// NUFFT-based periodic far-field evaluation.
/// Adds periodized windowed kernel contribution to pot (at source positions).
/// If bloch == nullptr: real charge[ns] in, real pot[ns] incremented.
/// If bloch != nullptr: real charge[ns] in (phased internally by e^{-iβ·x}),
///   shifted k-grid 2π L^{-T} m + β, complex pot[2*ns] incremented as
///   (Re, Im) interleaved per source.
/// Declared here, defined in pbc.cpp.
template <typename Real, int DIM>
void far_field_nufft(const Real *umat, Real rc, Real eps, Real beta,
                     int ns, const Real *sources, const Real *charge,
                     Real *pot, Real self_correction_value,
                     const Real *bloch = nullptr);

/// Recommended cube side rc for single-level pbcdmk.
///
/// Derived from balancing per-point costs of the direct residual
/// evaluation (c_d, ns/pair) and the NUFFT far-field (c_f, ns/point):
///
///     K_opt = c_f / (3^d · c_d)
///     rc    = (K_opt · V / N)^(1/d)
///
/// where V = |det avec| is the cell volume and K is the average number
/// of particles per cube. The K_opt table below is empirical (Laplace
/// 3D, double, clang++ 18, Intel Core Ultra 7 255H P-core @ 5.1 GHz)
/// and should be re-measured per host. See memory note
/// project_single_level_rc.md for the full derivation and sweep.
template <typename Real, int DIM>
inline Real single_level_optimal_rc(const pdmk_params &params,
                                    const Real *avec, int n_src) {
    // Cell volume from column-major lattice matrix.
    Real V;
    if constexpr (DIM == 3) {
        V = std::abs(
            avec[0] * (avec[4] * avec[8] - avec[5] * avec[7])
          - avec[3] * (avec[1] * avec[8] - avec[2] * avec[7])
          + avec[6] * (avec[1] * avec[5] - avec[2] * avec[4]));
    } else {
        V = std::abs(avec[0] * avec[3] - avec[1] * avec[2]);
    }

    // Empirical K_opt bracket table (Laplace 3D, double, this host).
    double K_opt;
    if      (params.eps >= 1e-3) K_opt = 15.0;
    else if (params.eps >= 1e-6) K_opt = 20.0;
    else if (params.eps >= 1e-9) K_opt = 40.0;
    else                         K_opt = 60.0;

    const double inv_d = 1.0 / static_cast<double>(DIM);
    const int N = std::max(n_src, 1);
    return static_cast<Real>(std::pow(K_opt * static_cast<double>(V) / N, inv_d));
}

/// Single-level PBC-DMK entry point.
/// If bloch == nullptr: fully periodic, pot[n_src] real output.
/// If bloch != nullptr: quasi-periodic (Laplace only for now, scalar
///   real charge input), pot[2*n_src] complex output (Re, Im) per source.
/// If rc <= 0 is passed, the tuned default from single_level_optimal_rc()
/// is used (recommended for typical uniform distributions).
template <typename Real, int DIM>
void pbcdmk(void *comm, const pdmk_params &params, const Real *avec, Real rc,
            int n_src, const Real *r_src, const Real *charge,
            const Real *normal, const Real *dipole_str, Real *pot,
            const Real *bloch = nullptr);

/// Per-stage timing breakdown surfaced by pbcdmk_adaptive when the caller
/// passes a non-null pointer. Wall-clock milliseconds, steady_clock.
/// build_ms + eval_ms + direct_ms + self_desort_ms + nufft_ms
///   ≈ total wall time.
struct PbcAdaptiveTiming {
    double build_ms       = 0.0;  ///< build_combined_sources + PBCTree ctor
    double eval_ms        = 0.0;  ///< proxy work only: upward_pass + downward_prologue
                                  ///<   + form_downward_expansions (MLI)
    double direct_ms      = 0.0;  ///< evaluate_direct_interactions (leaf-level near-field)
    double self_desort_ms = 0.0;  ///< apply_target_self_correction + desort to original order
    double nufft_ms       = 0.0;  ///< far_field_nufft
};

/// Adaptive (multilevel) PBC-DMK entry point. Uses PBCTree for near-field.
/// If bloch == nullptr: fully periodic, pot[n_src] real output.
/// If bloch != nullptr: quasi-periodic (Laplace only), pot[2*n_src] output
///   as (Re, Im) per source.
/// If timing != nullptr, the four-stage breakdown is recorded in *timing.
/// Declared here, defined in pbc.cpp.
template <typename Real, int DIM>
void pbcdmk_adaptive(const pdmk_params &params, const Real *avec, Real rc,
                     int n_src, const Real *r_src, const Real *charge, Real *pot,
                     const Real *bloch = nullptr,
                     PbcAdaptiveTiming *timing = nullptr);

} // namespace pbc
} // namespace dmk

#endif
