#include <pdmk/ewald_nufft.hpp>
#include <pdmk/cell_list.hpp>

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

#include <ducc0/nufft/nufft.h>
#include <ducc0/infra/mav.h>

#include "pdmk/nufft_backend.hpp"
#if PDMK_HAVE_FINUFFT
#  include <finufft.h>
#endif

namespace pdmk {

// ============================================================================
// Helpers
// ============================================================================

/// Choose alpha for the NUFFT Ewald.
///
/// For NUFFT-accelerated Ewald, reciprocal-space is O(M log M + N log(1/ε))
/// while real-space is O(N × n_neighbors). We choose alpha to limit the
/// real-space neighbor count per particle:
///
///   n_neighbors = ρ (4π/3) r_cut³   where ρ = N/V, r_cut = s/α
///
/// Setting n_neighbors = target_nbr and solving for α:
///   α = s × [4π N / (3 V × target_nbr)]^{1/3}
///
/// This ensures O(N × target_nbr) = O(N) real-space cost. The reciprocal
/// grid grows as (2sα)³ ∝ (N/target_nbr) but NUFFT handles this efficiently.
///
/// We also enforce α ≥ balanced_alpha to avoid needlessly large cutoffs
/// at low density.
static double choose_alpha(int N, double V,
                           const std::array<double,3> &/*heights*/, double s) {
    double alpha_balanced = std::pow(N * M_PI * M_PI * M_PI / (V * V), 1.0 / 6.0);

    // Target neighbors per particle in real-space.
    // Higher → smaller alpha → smaller k_max → smaller reciprocal grid.
    // Default 5000; override via env DMK_EWALD_TARGET_NBR for cells that
    // trip DUCC's per-axis limit (e.g. triclinic e1 at N=1e7).
    double target_nbr = 5000.0;
    if (const char* e = std::getenv("DMK_EWALD_TARGET_NBR")) {
        target_nbr = std::atof(e);
    }
    double rho = N / V;
    double alpha_density = s * std::cbrt(4.0 * M_PI * rho / (3.0 * target_nbr));

    return std::max(alpha_balanced, alpha_density);
}

/// Compute reciprocal grid dimensions so that all modes with |k| ≤ k_max
/// are included. Returns odd grid sizes (2*mmax+1).
struct RecipGrid {
    int ms, mt, mu;       // grid sizes (odd)
    int mmax[3];          // half-widths
};

static RecipGrid recip_grid_size(const Lattice &lat, double k_max) {
    Mat3 G = lat.reciprocal_matrix();
    Vec3 b[3];
    for (int i = 0; i < 3; ++i) b[i] = {G[i][0], G[i][1], G[i][2]};

    // Reciprocal-space "volume" and heights
    double Vr = std::abs(dot(b[0], cross(b[1], b[2])));
    double rh[3] = {
        Vr / norm(cross(b[1], b[2])),
        Vr / norm(cross(b[2], b[0])),
        Vr / norm(cross(b[0], b[1]))
    };

    RecipGrid rg;
    for (int d = 0; d < 3; ++d) {
        rg.mmax[d] = static_cast<int>(std::ceil(k_max / rh[d]));
    }
    rg.ms = 2 * rg.mmax[0] + 1;
    rg.mt = 2 * rg.mmax[1] + 1;
    rg.mu = 2 * rg.mmax[2] + 1;
    return rg;
}

// ============================================================================
// 3D type-1/type-2 NUFFT — DUCC or FINUFFT (selected at runtime)
// ============================================================================

namespace {

// DUCC type-1: points → uniform grid. Coords in [0, 2π); grid shape
// {mu, mt, ms} (z slowest). isign: -1 = exp(-ikx), +1 = exp(+ikx).
void nufft_type1_3d_ducc(
    const std::vector<double> &xj,
    const std::vector<double> &yj,
    const std::vector<double> &zj,
    const std::vector<std::complex<double>> &cj,
    int ms, int mt, int mu,
    double tol, int isign,
    std::vector<std::complex<double>> &grid)
{
    using namespace ducc0;
    size_t N = xj.size();
    std::vector<double> coord_flat(N * 3);
    for (size_t i = 0; i < N; ++i) {
        coord_flat[3*i + 0] = zj[i];
        coord_flat[3*i + 1] = yj[i];
        coord_flat[3*i + 2] = xj[i];
    }
    cmav<double, 2> coord(coord_flat.data(), std::array<size_t,2>{N, 3u});
    cmav<std::complex<double>, 1> points(cj.data(), std::array<size_t,1>{N});
    std::vector<size_t> shape = {(size_t)mu, (size_t)mt, (size_t)ms};
    grid.resize((size_t)ms * mt * mu);
    vfmav<std::complex<double>> uniform(grid.data(), shape);
    bool forward = (isign < 0);
    std::vector<double> periodicity = {2.0*M_PI, 2.0*M_PI, 2.0*M_PI};
    nu2u<double,double,double,double,double>(
        coord, points, forward, tol, /*nthreads*/0, uniform,
        /*verbosity*/0, /*sigma_min*/1.1, /*sigma_max*/2.6,
        periodicity, /*fft_order*/false);
}

void nufft_type2_3d_ducc(
    const std::vector<double> &xj,
    const std::vector<double> &yj,
    const std::vector<double> &zj,
    std::vector<std::complex<double>> &out,
    int ms, int mt, int mu,
    double tol, int isign,
    const std::vector<std::complex<double>> &grid)
{
    using namespace ducc0;
    size_t N = xj.size();
    std::vector<double> coord_flat(N * 3);
    for (size_t i = 0; i < N; ++i) {
        coord_flat[3*i + 0] = zj[i];
        coord_flat[3*i + 1] = yj[i];
        coord_flat[3*i + 2] = xj[i];
    }
    cmav<double, 2> coord(coord_flat.data(), std::array<size_t,2>{N, 3u});
    out.resize(N);
    vmav<std::complex<double>, 1> points(out.data(), std::array<size_t,1>{N});
    std::vector<size_t> shape = {(size_t)mu, (size_t)mt, (size_t)ms};
    cfmav<std::complex<double>> uniform(grid.data(), shape);
    bool forward = (isign < 0);
    std::vector<double> periodicity = {2.0*M_PI, 2.0*M_PI, 2.0*M_PI};
    u2nu<double,double,double,double,double>(
        coord, uniform, forward, tol, /*nthreads*/0, points,
        /*verbosity*/0, /*sigma_min*/1.1, /*sigma_max*/2.6,
        periodicity, /*fft_order*/false);
}

#if PDMK_HAVE_FINUFFT
// FINUFFT 3D type-1: points → uniform grid. Coords in [0, 2π) match
// FINUFFT's accepted range; modeord=0 (CMCL) matches DUCC fft_order=false.
void nufft_type1_3d_finufft(
    const std::vector<double> &xj,
    const std::vector<double> &yj,
    const std::vector<double> &zj,
    const std::vector<std::complex<double>> &cj,
    int ms, int mt, int mu,
    double tol, int isign,
    std::vector<std::complex<double>> &grid)
{
    int64_t N = (int64_t)xj.size();
    int64_t n_modes[3] = {ms, mt, mu};
    grid.resize((size_t)ms * (size_t)mt * (size_t)mu);

    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.nthreads  = 0;
    opts.modeord   = 0;
    opts.upsampfac = 2.0;

    // FINUFFT typically achieves ~10x the requested tolerance.
    // Tighten by 20x to match DUCC's effective accuracy at the requested
    // tol (e.g., tol=1e-12 → DUCC ~3e-13, FINUFFT at tol/20 ~5e-14 → 1e-12 actual).
    const double finufft_tol = std::max(tol * 0.05, 1e-15);

    finufft_plan plan = nullptr;
    int ier = finufft_makeplan(/*type=*/1, /*dim=*/3, n_modes,
                               /*iflag=*/(isign < 0 ? -1 : +1),
                               /*ntrans=*/1, finufft_tol, &plan, &opts);
    if (ier) throw std::runtime_error("FINUFFT 3D type-1 makeplan failed: ier=" + std::to_string(ier));
    ier = finufft_setpts(plan, N,
                         const_cast<double*>(xj.data()),
                         const_cast<double*>(yj.data()),
                         const_cast<double*>(zj.data()),
                         0, nullptr, nullptr, nullptr);
    if (ier) { finufft_destroy(plan); throw std::runtime_error("FINUFFT 3D type-1 setpts failed: ier=" + std::to_string(ier)); }
    ier = finufft_execute(plan,
                          const_cast<std::complex<double>*>(cj.data()),
                          grid.data());
    if (ier) { finufft_destroy(plan); throw std::runtime_error("FINUFFT 3D type-1 execute failed: ier=" + std::to_string(ier)); }
    finufft_destroy(plan);
}

void nufft_type2_3d_finufft(
    const std::vector<double> &xj,
    const std::vector<double> &yj,
    const std::vector<double> &zj,
    std::vector<std::complex<double>> &out,
    int ms, int mt, int mu,
    double tol, int isign,
    const std::vector<std::complex<double>> &grid)
{
    int64_t N = (int64_t)xj.size();
    int64_t n_modes[3] = {ms, mt, mu};
    out.resize((size_t)N);

    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.nthreads  = 0;
    opts.modeord   = 0;
    opts.upsampfac = 2.0;

    const double finufft_tol = std::max(tol * 0.05, 1e-15);

    finufft_plan plan = nullptr;
    int ier = finufft_makeplan(/*type=*/2, /*dim=*/3, n_modes,
                               /*iflag=*/(isign < 0 ? -1 : +1),
                               /*ntrans=*/1, finufft_tol, &plan, &opts);
    if (ier) throw std::runtime_error("FINUFFT 3D type-2 makeplan failed: ier=" + std::to_string(ier));
    ier = finufft_setpts(plan, N,
                         const_cast<double*>(xj.data()),
                         const_cast<double*>(yj.data()),
                         const_cast<double*>(zj.data()),
                         0, nullptr, nullptr, nullptr);
    if (ier) { finufft_destroy(plan); throw std::runtime_error("FINUFFT 3D type-2 setpts failed: ier=" + std::to_string(ier)); }
    ier = finufft_execute(plan,
                          out.data(),
                          const_cast<std::complex<double>*>(grid.data()));
    if (ier) { finufft_destroy(plan); throw std::runtime_error("FINUFFT 3D type-2 execute failed: ier=" + std::to_string(ier)); }
    finufft_destroy(plan);
}
#endif // PDMK_HAVE_FINUFFT

// ============================================================================
// 2D helpers — coords packed as {y, x} per point of length 2*N, matching
// grid shape {mt, ms} (DUCC fft_order=false, FINUFFT modeord=0).
// ============================================================================

void nufft_type1_2d_ducc(
    const std::vector<double> &coords_yx,
    const std::vector<std::complex<double>> &cj,
    int ms, int mt,
    double tol, int isign,
    std::vector<std::complex<double>> &grid)
{
    using namespace ducc0;
    size_t N = coords_yx.size() / 2;
    cmav<double, 2> coord(coords_yx.data(), std::array<size_t,2>{N, 2u});
    cmav<std::complex<double>, 1> pts(cj.data(), std::array<size_t,1>{N});
    grid.resize((size_t)ms * (size_t)mt);
    std::vector<size_t> shape = {(size_t)mt, (size_t)ms};
    vfmav<std::complex<double>> uniform(grid.data(), shape);
    std::vector<double> periodicity = {2.0*M_PI, 2.0*M_PI};
    bool forward = (isign < 0);
    nu2u<double,double,double,double,double>(
        coord, pts, forward, tol, /*nthreads*/0, uniform,
        /*verbosity*/0, /*sigma_min*/1.1, /*sigma_max*/2.6,
        periodicity, /*fft_order*/false);
}

void nufft_type2_2d_ducc(
    const std::vector<double> &coords_yx,
    std::vector<std::complex<double>> &out,
    int ms, int mt,
    double tol, int isign,
    const std::vector<std::complex<double>> &grid)
{
    using namespace ducc0;
    size_t N = coords_yx.size() / 2;
    cmav<double, 2> coord(coords_yx.data(), std::array<size_t,2>{N, 2u});
    out.resize(N);
    vmav<std::complex<double>, 1> pts_out(out.data(), std::array<size_t,1>{N});
    std::vector<size_t> shape = {(size_t)mt, (size_t)ms};
    cfmav<std::complex<double>> uniform(grid.data(), shape);
    std::vector<double> periodicity = {2.0*M_PI, 2.0*M_PI};
    bool forward = (isign < 0);
    u2nu<double,double,double,double,double>(
        coord, uniform, forward, tol, /*nthreads*/0, pts_out,
        /*verbosity*/0, /*sigma_min*/1.1, /*sigma_max*/2.6,
        periodicity, /*fft_order*/false);
}

#if PDMK_HAVE_FINUFFT
void nufft_type1_2d_finufft(
    const std::vector<double> &coords_yx,
    const std::vector<std::complex<double>> &cj,
    int ms, int mt,
    double tol, int isign,
    std::vector<std::complex<double>> &grid)
{
    int64_t N = (int64_t)(coords_yx.size() / 2);
    std::vector<double> xs((size_t)N), ys((size_t)N);
    for (int64_t i = 0; i < N; ++i) {
        ys[(size_t)i] = coords_yx[2*(size_t)i + 0];
        xs[(size_t)i] = coords_yx[2*(size_t)i + 1];
    }
    int64_t n_modes[2] = {ms, mt};
    grid.resize((size_t)ms * (size_t)mt);

    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.nthreads  = 0;
    opts.modeord   = 0;
    opts.upsampfac = 2.0;
    const double finufft_tol = std::max(tol * 0.05, 1e-15);

    finufft_plan plan = nullptr;
    int ier = finufft_makeplan(1, 2, n_modes, (isign<0?-1:+1),
                               /*ntrans=*/1, finufft_tol, &plan, &opts);
    if (ier) throw std::runtime_error("FINUFFT 2D type-1 makeplan failed: ier=" + std::to_string(ier));
    ier = finufft_setpts(plan, N, xs.data(), ys.data(), nullptr,
                         0, nullptr, nullptr, nullptr);
    if (ier) { finufft_destroy(plan); throw std::runtime_error("FINUFFT 2D type-1 setpts failed: ier=" + std::to_string(ier)); }
    ier = finufft_execute(plan,
                          const_cast<std::complex<double>*>(cj.data()),
                          grid.data());
    if (ier) { finufft_destroy(plan); throw std::runtime_error("FINUFFT 2D type-1 execute failed: ier=" + std::to_string(ier)); }
    finufft_destroy(plan);
}

void nufft_type2_2d_finufft(
    const std::vector<double> &coords_yx,
    std::vector<std::complex<double>> &out,
    int ms, int mt,
    double tol, int isign,
    const std::vector<std::complex<double>> &grid)
{
    int64_t N = (int64_t)(coords_yx.size() / 2);
    std::vector<double> xs((size_t)N), ys((size_t)N);
    for (int64_t i = 0; i < N; ++i) {
        ys[(size_t)i] = coords_yx[2*(size_t)i + 0];
        xs[(size_t)i] = coords_yx[2*(size_t)i + 1];
    }
    int64_t n_modes[2] = {ms, mt};
    out.resize((size_t)N);

    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.nthreads  = 0;
    opts.modeord   = 0;
    opts.upsampfac = 2.0;
    const double finufft_tol = std::max(tol * 0.05, 1e-15);

    finufft_plan plan = nullptr;
    int ier = finufft_makeplan(2, 2, n_modes, (isign<0?-1:+1),
                               /*ntrans=*/1, finufft_tol, &plan, &opts);
    if (ier) throw std::runtime_error("FINUFFT 2D type-2 makeplan failed: ier=" + std::to_string(ier));
    ier = finufft_setpts(plan, N, xs.data(), ys.data(), nullptr,
                         0, nullptr, nullptr, nullptr);
    if (ier) { finufft_destroy(plan); throw std::runtime_error("FINUFFT 2D type-2 setpts failed: ier=" + std::to_string(ier)); }
    ier = finufft_execute(plan, out.data(),
                          const_cast<std::complex<double>*>(grid.data()));
    if (ier) { finufft_destroy(plan); throw std::runtime_error("FINUFFT 2D type-2 execute failed: ier=" + std::to_string(ier)); }
    finufft_destroy(plan);
}
#endif // PDMK_HAVE_FINUFFT

} // anonymous namespace

// 2D dispatching public-static helpers.
static void nufft_type1_2d(
    const std::vector<double> &coords_yx,
    const std::vector<std::complex<double>> &cj,
    int ms, int mt,
    double tol, int isign,
    std::vector<std::complex<double>> &grid)
{
#if PDMK_HAVE_FINUFFT
    if (pdmk::pick_nufft_backend() == pdmk::NufftBackend::Finufft)
        nufft_type1_2d_finufft(coords_yx, cj, ms, mt, tol, isign, grid);
    else
        nufft_type1_2d_ducc(coords_yx, cj, ms, mt, tol, isign, grid);
#else
    nufft_type1_2d_ducc(coords_yx, cj, ms, mt, tol, isign, grid);
#endif
}

static void nufft_type2_2d(
    const std::vector<double> &coords_yx,
    std::vector<std::complex<double>> &out,
    int ms, int mt,
    double tol, int isign,
    const std::vector<std::complex<double>> &grid)
{
#if PDMK_HAVE_FINUFFT
    if (pdmk::pick_nufft_backend() == pdmk::NufftBackend::Finufft)
        nufft_type2_2d_finufft(coords_yx, out, ms, mt, tol, isign, grid);
    else
        nufft_type2_2d_ducc(coords_yx, out, ms, mt, tol, isign, grid);
#else
    nufft_type2_2d_ducc(coords_yx, out, ms, mt, tol, isign, grid);
#endif
}

// Dispatching public-static helpers used by ewald_nufft_targets.
static void nufft_type1(
    const std::vector<double> &xj,
    const std::vector<double> &yj,
    const std::vector<double> &zj,
    const std::vector<std::complex<double>> &cj,
    int ms, int mt, int mu,
    double tol, int isign,
    std::vector<std::complex<double>> &grid)
{
#if PDMK_HAVE_FINUFFT
    if (pdmk::pick_nufft_backend() == pdmk::NufftBackend::Finufft)
        nufft_type1_3d_finufft(xj, yj, zj, cj, ms, mt, mu, tol, isign, grid);
    else
        nufft_type1_3d_ducc(xj, yj, zj, cj, ms, mt, mu, tol, isign, grid);
#else
    nufft_type1_3d_ducc(xj, yj, zj, cj, ms, mt, mu, tol, isign, grid);
#endif
}

static void nufft_type2(
    const std::vector<double> &xj,
    const std::vector<double> &yj,
    const std::vector<double> &zj,
    std::vector<std::complex<double>> &out,
    int ms, int mt, int mu,
    double tol, int isign,
    const std::vector<std::complex<double>> &grid)
{
#if PDMK_HAVE_FINUFFT
    if (pdmk::pick_nufft_backend() == pdmk::NufftBackend::Finufft)
        nufft_type2_3d_finufft(xj, yj, zj, out, ms, mt, mu, tol, isign, grid);
    else
        nufft_type2_3d_ducc(xj, yj, zj, out, ms, mt, mu, tol, isign, grid);
#else
    nufft_type2_3d_ducc(xj, yj, zj, out, ms, mt, mu, tol, isign, grid);
#endif
}

// ============================================================================
// ewald_nufft
// ============================================================================

EwaldResult ewald_nufft(
    const Lattice &lat,
    const std::vector<Vec3> &positions,
    const std::vector<double> &charges,
    double s,
    double nufft_tol)
{
    int N = static_cast<int>(positions.size());
    double V = lat.volume();
    auto heights = lat.cell_heights();
    double alpha = choose_alpha(N, V, heights, s);
    double r_cut = s / alpha;
    double k_max = 2.0 * s * alpha;

    // Optional profiling
    const bool profile = (std::getenv("PDMK_PROFILE_EWALD") != nullptr);
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_phase = t_start;
    auto ms_since = [&]() {
        auto now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - t_phase).count();
        t_phase = now;
        return ms;
    };

    if (profile) {
        auto heights = lat.cell_heights();
        std::fprintf(stderr, "[ewald_nufft] N=%d V=%.1f alpha=%.4f r_cut=%.2f k_max=%.2f\n",
            N, V, alpha, r_cut, k_max);
        std::fprintf(stderr, "  cell_heights=(%.2f,%.2f,%.2f) nbins=(%d,%d,%d)\n",
            heights[0], heights[1], heights[2],
            std::max(1, (int)(heights[0]/r_cut)),
            std::max(1, (int)(heights[1]/r_cut)),
            std::max(1, (int)(heights[2]/r_cut)));
    }

    EwaldResult result;
    result.potentials.assign(N, 0.0);
    result.forces.resize(N, {0.0, 0.0, 0.0});
    result.energy = 0.0;

    // ==================================================================
    // 1. Real-space sum (cell-list accelerated, OpenMP parallel)
    // ==================================================================
    double two_alpha_over_sqrtpi = 2.0 * alpha / std::sqrt(M_PI);

    CellList cl(lat.cell_matrix(), r_cut, positions);

    // Each particle accumulates its own potential and force from all neighbors.
    // This does 2× the work of a half-pair approach but is trivially parallel.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        cl.for_each_neighbor(i, [&](int j, const Vec3 &dr, double r2) {
            double r = std::sqrt(r2);
            double ar = alpha * r;
            double erfc_ar = std::erfc(ar);

            result.potentials[i] += charges[j] * erfc_ar / r;

            // dr = rj - ri; force on i is in direction -dr
            double fmag = charges[i] * charges[j] *
                (erfc_ar / r2 + two_alpha_over_sqrtpi * std::exp(-ar * ar) / r) / r;
            result.forces[i][0] -= fmag * dr[0];
            result.forces[i][1] -= fmag * dr[1];
            result.forces[i][2] -= fmag * dr[2];
        });
    }

    if (profile) std::fprintf(stderr, "  real-space:  %.1f ms\n", ms_since());

    // ==================================================================
    // 2. Reciprocal-space sum (NUFFT accelerated)
    // ==================================================================
    RecipGrid rg = recip_grid_size(lat, k_max);
    int ms = rg.ms, mt = rg.mt, mu = rg.mu;
    size_t M = (size_t)ms * mt * mu;
    if (profile) std::fprintf(stderr, "  recip grid: %dx%dx%d = %zu modes\n", ms, mt, mu, M);

    // Fractional coords scaled to [0, 2π)
    Mat3 h_inv_mat = lat.inverse_matrix();
    std::vector<double> xj(N), yj(N), zj(N);
    for (int i = 0; i < N; ++i) {
        Vec3 s_frac = lat.to_fractional(positions[i]);
        xj[i] = 2.0 * M_PI * s_frac[0];
        yj[i] = 2.0 * M_PI * s_frac[1];
        zj[i] = 2.0 * M_PI * s_frac[2];
    }

    // Type-1 NUFFT: structure factor S(m) = Σ_j q_j exp(-i m·x_j)
    std::vector<std::complex<double>> cj(N);
    for (int i = 0; i < N; ++i) cj[i] = charges[i];

    std::vector<std::complex<double>> S;
    nufft_type1(xj, yj, zj, cj, ms, mt, mu, nufft_tol, -1, S);

    if (profile) std::fprintf(stderr, "  nufft type1: %.1f ms\n", ms_since());

    // Build weight array and k-vectors
    Mat3 G = lat.reciprocal_matrix();
    Vec3 b[3];
    for (int d = 0; d < 3; ++d) b[d] = {G[d][0], G[d][1], G[d][2]};

    double inv4a2 = 1.0 / (4.0 * alpha * alpha);
    double prefac = 4.0 * M_PI / V;
    double k_max2 = k_max * k_max;

    std::vector<double> w_grid(M, 0.0);
    std::vector<double> kx_grid(M, 0.0), ky_grid(M, 0.0), kz_grid(M, 0.0);

    for (int i3 = 0; i3 < mu; ++i3) {
        int m3 = i3 - rg.mmax[2];
        for (int i2 = 0; i2 < mt; ++i2) {
            int m2 = i2 - rg.mmax[1];
            for (int i1 = 0; i1 < ms; ++i1) {
                int m1 = i1 - rg.mmax[0];
                if (m1 == 0 && m2 == 0 && m3 == 0) continue;

                Vec3 kv = {
                    m1 * b[0][0] + m2 * b[1][0] + m3 * b[2][0],
                    m1 * b[0][1] + m2 * b[1][1] + m3 * b[2][1],
                    m1 * b[0][2] + m2 * b[1][2] + m3 * b[2][2]
                };
                double k2 = dot(kv, kv);
                if (k2 > k_max2) continue;

                size_t idx = (size_t)i3 * mt * ms + (size_t)i2 * ms + i1;
                w_grid[idx]  = prefac * std::exp(-k2 * inv4a2) / k2;
                kx_grid[idx] = kv[0];
                ky_grid[idx] = kv[1];
                kz_grid[idx] = kv[2];
            }
        }
    }

    // Energy: E_recip = 0.5 * Σ w(k) |S(k)|²
    double E_recip = 0.0;
    for (size_t i = 0; i < M; ++i) {
        E_recip += w_grid[i] * std::norm(S[i]);
    }
    E_recip *= 0.5;

    // Potential: type-2 NUFFT of w(m)*S(m) evaluated at particle positions
    // u_recip[j] = Re{ Σ_m w(m) S(m) exp(+i m·x_j) }
    // Note: S was computed with isign=-1, i.e. S(m) = Σ q_j exp(-i m·x_j).
    // The potential needs Σ w(m) * [Σ_k q_k exp(-i m·r_k)] * exp(+i m·r_j)
    // = Σ_k q_k Σ_m w(m) exp(i m·(r_j - r_k))
    // This is the type-2 NUFFT with isign=+1 of the grid w*S.
    std::vector<std::complex<double>> wS(M);
    for (size_t i = 0; i < M; ++i) wS[i] = w_grid[i] * S[i];

    std::vector<std::complex<double>> pot_recip;
    nufft_type2(xj, yj, zj, pot_recip, ms, mt, mu, nufft_tol, +1, wS);

    if (profile) std::fprintf(stderr, "  nufft type2 (pot): %.1f ms\n", ms_since());

    for (int i = 0; i < N; ++i) {
        result.potentials[i] += pot_recip[i].real();
    }

    // Forces: 3× type-2 NUFFT
    // From E = 0.5 Σ_k w|S|², ∂E/∂r_{j,d} = q_j Σ_k w k_d Im{S* exp(-ikr_j)}
    //   = -q_j Σ_k w k_d Im{S exp(+ikr_j)}
    // So f_{j,d} = -∂E/∂r_{j,d} = q_j Im{ Σ_k k_d w(k) S(k) exp(+ik·r_j) }
    // Use G_d(m) = k_d * w(m) * S(m) (= k_d * wS[m]) with isign=+1.
    double *kd_grids[3] = {kx_grid.data(), ky_grid.data(), kz_grid.data()};
    for (int d = 0; d < 3; ++d) {
        std::vector<std::complex<double>> Gd(M);
        for (size_t i = 0; i < M; ++i) {
            Gd[i] = kd_grids[d][i] * wS[i];
        }
        std::vector<std::complex<double>> force_d;
        nufft_type2(xj, yj, zj, force_d, ms, mt, mu, nufft_tol, +1, Gd);
        for (int j = 0; j < N; ++j) {
            result.forces[j][d] += charges[j] * force_d[j].imag();
        }
    }

    if (profile) std::fprintf(stderr, "  nufft type2 (3×force): %.1f ms\n", ms_since());

    // ==================================================================
    // 3. Self-energy correction
    // ==================================================================
    double E_self = -(alpha / std::sqrt(M_PI));
    double sum_q2 = 0.0;
    for (int i = 0; i < N; ++i) {
        result.potentials[i] -= 2.0 * alpha / std::sqrt(M_PI) * charges[i];
        sum_q2 += charges[i] * charges[i];
    }
    E_self *= sum_q2;

    // ==================================================================
    // 4. Charged-system correction
    // ==================================================================
    double Q_tot = 0.0;
    for (int i = 0; i < N; ++i) Q_tot += charges[i];
    double E_charged = -M_PI * Q_tot * Q_tot / (2.0 * V * alpha * alpha);

    // ==================================================================
    // 5. Total energy
    // ==================================================================
    double E_real = 0.0;
    for (int i = 0; i < N; ++i) {
        E_real += charges[i] * result.potentials[i];
    }
    // Note: E_real here includes real+recip+self potential contributions
    // summed as Σ q_i u_i. The actual energy is 0.5 * Σ q_i u_i + E_charged.
    result.energy = 0.5 * E_real + E_charged;

    return result;
}

// ============================================================================
// ewald_nufft_targets — subset-target variant for accuracy references
// at large N. Real-space loop is over M targets only (via CellList);
// reciprocal type-2 NUFFT runs at M target positions (not N sources).
// Forces and energy are not computed.
// ============================================================================

std::vector<double> ewald_nufft_targets(
    const Lattice &lat,
    const std::vector<Vec3> &positions,
    const std::vector<double> &charges,
    const std::vector<int> &target_indices,
    double s,
    double nufft_tol)
{
    int N = static_cast<int>(positions.size());
    int Mt = static_cast<int>(target_indices.size());
    double V = lat.volume();
    auto heights = lat.cell_heights();
    double alpha = choose_alpha(N, V, heights, s);
    double r_cut = s / alpha;
    double k_max = 2.0 * s * alpha;

    std::vector<double> out(Mt, 0.0);

    // 1. Real-space sum at target indices only (cell-list accelerated,
    //    OpenMP parallel over targets).
    CellList cl(lat.cell_matrix(), r_cut, positions);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < Mt; ++m) {
        int i = target_indices[m];
        double acc = 0.0;
        cl.for_each_neighbor(i, [&](int j, const Vec3 &dr, double r2) {
            (void)dr;
            double r = std::sqrt(r2);
            double ar = alpha * r;
            double erfc_ar = std::erfc(ar);
            acc += charges[j] * erfc_ar / r;
        });
        out[m] = acc;
    }

    // 2. Reciprocal-space sum via NUFFT.
    //    Type-1 runs over all N sources; type-2 only at M targets.
    RecipGrid rg = recip_grid_size(lat, k_max);
    int ms = rg.ms, mt = rg.mt, mu = rg.mu;
    size_t Mk = (size_t)ms * mt * mu;

    Mat3 h_inv_mat = lat.inverse_matrix();
    std::vector<double> xj(N), yj(N), zj(N);
    for (int i = 0; i < N; ++i) {
        Vec3 s_frac = lat.to_fractional(positions[i]);
        xj[i] = 2.0 * M_PI * s_frac[0];
        yj[i] = 2.0 * M_PI * s_frac[1];
        zj[i] = 2.0 * M_PI * s_frac[2];
    }

    std::vector<std::complex<double>> cj(N);
    for (int i = 0; i < N; ++i) cj[i] = charges[i];

    std::vector<std::complex<double>> S;
    nufft_type1(xj, yj, zj, cj, ms, mt, mu, nufft_tol, -1, S);

    Mat3 G = lat.reciprocal_matrix();
    Vec3 b[3];
    for (int d = 0; d < 3; ++d) b[d] = {G[d][0], G[d][1], G[d][2]};

    double inv4a2 = 1.0 / (4.0 * alpha * alpha);
    double prefac = 4.0 * M_PI / V;
    double k_max2 = k_max * k_max;

    std::vector<std::complex<double>> wS(Mk);
    for (int i3 = 0; i3 < mu; ++i3) {
        int m3 = i3 - rg.mmax[2];
        for (int i2 = 0; i2 < mt; ++i2) {
            int m2 = i2 - rg.mmax[1];
            for (int i1 = 0; i1 < ms; ++i1) {
                int m1 = i1 - rg.mmax[0];
                size_t idx = (size_t)i3 * mt * ms + (size_t)i2 * ms + i1;
                if (m1 == 0 && m2 == 0 && m3 == 0) { wS[idx] = 0.0; continue; }
                Vec3 kv = {
                    m1 * b[0][0] + m2 * b[1][0] + m3 * b[2][0],
                    m1 * b[0][1] + m2 * b[1][1] + m3 * b[2][1],
                    m1 * b[0][2] + m2 * b[1][2] + m3 * b[2][2]
                };
                double k2 = dot(kv, kv);
                if (k2 > k_max2) { wS[idx] = 0.0; continue; }
                double w = prefac * std::exp(-k2 * inv4a2) / k2;
                wS[idx] = w * S[idx];
            }
        }
    }

    // Target fractional coords scaled to [0, 2π)
    std::vector<double> xt(Mt), yt(Mt), zt(Mt);
    for (int m = 0; m < Mt; ++m) {
        int i = target_indices[m];
        Vec3 s_frac = lat.to_fractional(positions[i]);
        xt[m] = 2.0 * M_PI * s_frac[0];
        yt[m] = 2.0 * M_PI * s_frac[1];
        zt[m] = 2.0 * M_PI * s_frac[2];
    }

    std::vector<std::complex<double>> pot_recip;
    nufft_type2(xt, yt, zt, pot_recip, ms, mt, mu, nufft_tol, +1, wS);
    for (int m = 0; m < Mt; ++m) out[m] += pot_recip[m].real();

    // 3. Self-correction (only at targets).
    double self_const = -2.0 * alpha / std::sqrt(M_PI);
    for (int m = 0; m < Mt; ++m) {
        int i = target_indices[m];
        out[m] += self_const * charges[i];
    }

    return out;
}

// ============================================================================
// ewald_direct (brute-force O(N²) reference)
// ============================================================================

EwaldResult ewald_direct(
    const Lattice &lat,
    const std::vector<Vec3> &positions,
    const std::vector<double> &charges,
    double s)
{
    int N = static_cast<int>(positions.size());
    double V = lat.volume();
    // ewald_direct uses balanced alpha (O(N²) anyway)
    double alpha = std::pow(N * M_PI * M_PI * M_PI / (V * V), 1.0 / 6.0);
    double r_cut = s / alpha;
    double k_max = 2.0 * s * alpha;

    EwaldResult result;
    result.potentials.assign(N, 0.0);
    result.forces.resize(N, {0.0, 0.0, 0.0});
    result.energy = 0.0;

    Vec3 a[3];
    for (int d = 0; d < 3; ++d) a[d] = lat.lattice_vector(d);

    Mat3 G = lat.reciprocal_matrix();
    Vec3 b[3];
    for (int d = 0; d < 3; ++d) b[d] = {G[d][0], G[d][1], G[d][2]};

    // Real-space cutoff shells
    auto heights = lat.cell_heights();
    int nmax[3];
    for (int d = 0; d < 3; ++d)
        nmax[d] = static_cast<int>(std::ceil(r_cut / heights[d]));

    double r_cut2 = r_cut * r_cut;
    double two_alpha_over_sqrtpi = 2.0 * alpha / std::sqrt(M_PI);

    // Real-space sum
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int n0 = -nmax[0]; n0 <= nmax[0]; ++n0)
            for (int n1 = -nmax[1]; n1 <= nmax[1]; ++n1)
            for (int n2 = -nmax[2]; n2 <= nmax[2]; ++n2) {
                if (i == j && n0 == 0 && n1 == 0 && n2 == 0) continue;
                Vec3 dr = positions[i] - positions[j];
                dr = dr + static_cast<double>(n0) * a[0]
                        + static_cast<double>(n1) * a[1]
                        + static_cast<double>(n2) * a[2];
                double r2 = dot(dr, dr);
                if (r2 > r_cut2) continue;
                double r = std::sqrt(r2);
                double ar = alpha * r;
                double erfc_ar = std::erfc(ar);
                result.potentials[i] += charges[j] * erfc_ar / r;

                // dr = r_i - r_j + R_n; force on i is f_i = +fmag * dr
                // (repulsion points from j to i, same direction as dr)
                double fmag = charges[i] * charges[j] *
                    (erfc_ar / r2 + two_alpha_over_sqrtpi * std::exp(-ar * ar) / r) / r;
                result.forces[i][0] += fmag * dr[0];
                result.forces[i][1] += fmag * dr[1];
                result.forces[i][2] += fmag * dr[2];
            }
        }
    }

    // Reciprocal-space sum
    double Vr = std::abs(dot(b[0], cross(b[1], b[2])));
    double rh[3] = {
        Vr / norm(cross(b[1], b[2])),
        Vr / norm(cross(b[2], b[0])),
        Vr / norm(cross(b[0], b[1]))
    };
    int mmax[3];
    for (int d = 0; d < 3; ++d)
        mmax[d] = static_cast<int>(std::ceil(k_max / rh[d]));

    double km2 = k_max * k_max;
    double inv4a2 = 1.0 / (4.0 * alpha * alpha);
    double prefac = 4.0 * M_PI / V;

    for (int m3 = -mmax[2]; m3 <= mmax[2]; ++m3)
    for (int m2 = -mmax[1]; m2 <= mmax[1]; ++m2)
    for (int m1 = -mmax[0]; m1 <= mmax[0]; ++m1) {
        if (m1 == 0 && m2 == 0 && m3 == 0) continue;
        Vec3 kv = {
            m1*b[0][0] + m2*b[1][0] + m3*b[2][0],
            m1*b[0][1] + m2*b[1][1] + m3*b[2][1],
            m1*b[0][2] + m2*b[1][2] + m3*b[2][2]
        };
        double k2 = dot(kv, kv);
        if (k2 > km2) continue;
        double w = prefac * std::exp(-k2 * inv4a2) / k2;

        // Structure factor
        std::complex<double> Sk(0.0, 0.0);
        for (int j = 0; j < N; ++j) {
            double phase = dot(kv, positions[j]);
            Sk += charges[j] * std::complex<double>(std::cos(phase), -std::sin(phase));
        }

        // Potential
        for (int i = 0; i < N; ++i) {
            double phase = dot(kv, positions[i]);
            std::complex<double> eikx(std::cos(phase), std::sin(phase));
            result.potentials[i] += (w * Sk * eikx).real();
        }

        // Forces: f_{j,d} = q_j Im{ Σ_k k_d w(k) S(k) exp(+ik·r_j) }
        for (int j = 0; j < N; ++j) {
            double phase = dot(kv, positions[j]);
            std::complex<double> val = w * Sk * std::complex<double>(std::cos(phase), std::sin(phase));
            for (int d = 0; d < 3; ++d) {
                result.forces[j][d] += charges[j] * kv[d] * val.imag();
            }
        }
    }

    // Self-energy correction
    double sum_q2 = 0.0;
    for (int i = 0; i < N; ++i) {
        result.potentials[i] -= 2.0 * alpha / std::sqrt(M_PI) * charges[i];
        sum_q2 += charges[i] * charges[i];
    }

    // Charged-system correction
    double Q_tot = 0.0;
    for (int i = 0; i < N; ++i) Q_tot += charges[i];
    double E_charged = -M_PI * Q_tot * Q_tot / (2.0 * V * alpha * alpha);

    // Total energy
    double E_sum = 0.0;
    for (int i = 0; i < N; ++i) E_sum += charges[i] * result.potentials[i];
    result.energy = 0.5 * E_sum + E_charged;

    return result;
}

// ============================================================================
// ewald_direct_2d — 2D log-kernel Ewald (brute-force reference)
// ============================================================================
//
// Splits G(r) = -log(r)/(2π)  =  G_short(r) + G_long(r) with
//   G_short(r) = E1(α² r²) / (4π)          (real space, decays like e^{-α²r²})
//   G_long(r)  = (1/V) Σ_{k≠0} exp(-k²/(4α²))/k² exp(+i k·r)
//
// The k=0 Fourier mode is explicitly excluded (gauge). For charge-neutral
// systems the output has zero mean. For non-neutral systems the output is
// still well-defined up to a constant determined by the gauge.
//
// The self-image (i==j, n==0) is skipped because G_short has a log singularity
// at r=0. The caller receives phi_i = the external-field potential at r_i,
// summed over all other sources and their periodic images.

namespace {

// Exponential integral E1(x) = ∫_x^∞ e^{-t}/t dt for x > 0.
// Series for small x, continued fraction (Lentz) for large x.
inline double exp_int_E1(double x) {
    if (x <= 0.0) return std::numeric_limits<double>::infinity();
    if (x < 1.0) {
        const double gamma_em = 0.5772156649015329;
        double s = -gamma_em - std::log(x);
        double term = 1.0;
        for (int n = 1; n < 60; ++n) {
            term *= -x / (double)n;
            double add = -term / (double)n;
            s += add;
            if (std::abs(add) < 1e-18 * std::abs(s)) break;
        }
        return s;
    } else {
        double b = x + 1.0;
        double c = 1e300;
        double d = 1.0 / b;
        double h = d;
        for (int i = 1; i <= 200; ++i) {
            double a = -(double)i * (double)i;
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            double delta = c * d;
            h *= delta;
            if (std::abs(delta - 1.0) < 1e-16) break;
        }
        return h * std::exp(-x);
    }
}

}  // namespace

EwaldResult2D ewald_direct_2d(
    const Mat2 &cell,
    const std::vector<Vec2> &positions,
    const std::vector<double> &charges,
    double s,
    double alpha_override)
{
    int N = (int)positions.size();

    // Area of 2D cell (signed)
    double V = std::abs(cell[0][0] * cell[1][1] - cell[0][1] * cell[1][0]);

    // Balanced alpha for log-kernel 2D Ewald (O(N²) in this direct impl).
    // A correct Ewald sum is alpha-independent; alpha only controls the
    // real/reciprocal balance. Caller can override (0.0 = default).
    double alpha = alpha_override > 0.0 ? alpha_override : std::sqrt(M_PI / V);
    double r_cut = s / alpha;
    double k_max = 2.0 * s * alpha;

    EwaldResult2D result;
    result.potentials.assign(N, 0.0);
    result.forces.assign(N, Vec2{0.0, 0.0});
    result.energy = 0.0;

    // Reciprocal-lattice basis (columns of (cell^T)^{-1})
    // cell[j] is lattice vector a_j. In 2D: a_0=(a0x,a0y), a_1=(a1x,a1y).
    // Reciprocal b_k satisfies a_j · b_k = δ_jk (no 2π factor here; we
    // apply 2π at the mode index).
    double a0x = cell[0][0], a0y = cell[0][1];
    double a1x = cell[1][0], a1y = cell[1][1];
    double det = a0x * a1y - a0y * a1x;
    Vec2 b0{ a1y / det, -a1x / det};
    Vec2 b1{-a0y / det,  a0x / det};

    // Heights for cutoff shell counts
    auto norm2 = [](Vec2 v) { return std::sqrt(v[0]*v[0] + v[1]*v[1]); };
    // Height along a_0 direction = |det| / |a_1|, etc.
    double h[2] = {std::abs(det) / norm2({a1x, a1y}),
                   std::abs(det) / norm2({a0x, a0y})};
    int nmax[2];
    for (int d = 0; d < 2; ++d) nmax[d] = (int)std::ceil(r_cut / h[d]) + 1;

    auto norm_b = [&](int d) -> double {
        if (d == 0) return norm2(b0);
        return norm2(b1);
    };
    int mmax[2];
    for (int d = 0; d < 2; ++d) mmax[d] = (int)std::ceil(k_max / (2.0 * M_PI * norm_b(d))) + 1;

    double r_cut2 = r_cut * r_cut;
    double k_max2 = k_max * k_max;
    double inv4a2 = 1.0 / (4.0 * alpha * alpha);
    double inv4pi = 1.0 / (4.0 * M_PI);

    // Real-space sum: phi_i += q_j * E1(α² r²) / (4π)
    for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
        for (int n0 = -nmax[0]; n0 <= nmax[0]; ++n0)
        for (int n1 = -nmax[1]; n1 <= nmax[1]; ++n1) {
            if (i == j && n0 == 0 && n1 == 0) continue;
            double dx = positions[i][0] - positions[j][0] - n0 * a0x - n1 * a1x;
            double dy = positions[i][1] - positions[j][1] - n0 * a0y - n1 * a1y;
            double r2 = dx*dx + dy*dy;
            if (r2 > r_cut2) continue;
            double G_s = exp_int_E1(alpha * alpha * r2) * inv4pi;
            result.potentials[i] += charges[j] * G_s;
            // Force: -∇ G_short(r) = (1/(2π)) exp(-α² r²) * r̂ / r
            //      = (1/(2π)) exp(-α² r²) / r² * (dx, dy)
            double fmag = charges[i] * charges[j]
                * std::exp(-alpha * alpha * r2) / (2.0 * M_PI * r2);
            result.forces[i][0] += fmag * dx;
            result.forces[i][1] += fmag * dy;
        }

    // Reciprocal-space sum: phi_i += Re{ (1/V) Σ_{k≠0} w(k) S(k) e^{+ik·r_i} }
    // where w(k) = exp(-k²/(4α²)) / k²
    for (int m0 = -mmax[0]; m0 <= mmax[0]; ++m0)
    for (int m1 = -mmax[1]; m1 <= mmax[1]; ++m1) {
        if (m0 == 0 && m1 == 0) continue;
        double kx = 2.0 * M_PI * (m0 * b0[0] + m1 * b1[0]);
        double ky = 2.0 * M_PI * (m0 * b0[1] + m1 * b1[1]);
        double k2 = kx*kx + ky*ky;
        if (k2 > k_max2) continue;
        double w = std::exp(-k2 * inv4a2) / (k2 * V);

        std::complex<double> Sk(0.0, 0.0);
        for (int j = 0; j < N; ++j) {
            double phase = kx * positions[j][0] + ky * positions[j][1];
            Sk += charges[j] * std::complex<double>(std::cos(phase), -std::sin(phase));
        }
        for (int i = 0; i < N; ++i) {
            double phase = kx * positions[i][0] + ky * positions[i][1];
            std::complex<double> eikx(std::cos(phase), std::sin(phase));
            result.potentials[i] += (w * Sk * eikx).real();
        }
        // Forces: f_j = q_j Im{ Σ_k k w(k) S(k) e^{+ik·r_j} }
        for (int j = 0; j < N; ++j) {
            double phase = kx * positions[j][0] + ky * positions[j][1];
            std::complex<double> v = w * Sk
                * std::complex<double>(std::cos(phase), std::sin(phase));
            result.forces[j][0] += charges[j] * kx * v.imag();
            result.forces[j][1] += charges[j] * ky * v.imag();
        }
    }

    // Self-correction: the real-space sum skips (j=i, n=0) because
    // E1 diverges at r=0, but the regularized limit of G_s(r) +
    // log(r)/(2π) at r→0 is -(γ + 2 log ξ)/(4π). Add this back per
    // source so the full periodic Green's function evaluated at a
    // source point is self-consistent. Without this term the result
    // drifts with α.
    const double gamma_em = 0.5772156649015329;
    const double self_const = -(gamma_em + 2.0 * std::log(alpha)) / (4.0 * M_PI);
    for (int i = 0; i < N; ++i)
        result.potentials[i] += charges[i] * self_const;

    // Energy: 0.5 Σ q_i phi_i. (No separate charged-system correction
    // here because the k=0 mode is pinned to zero; for non-neutral sets
    // the returned energy is gauge-dependent.)
    double E = 0.0;
    for (int i = 0; i < N; ++i) E += charges[i] * result.potentials[i];
    result.energy = 0.5 * E;

    return result;
}

// ============================================================================
// ewald_nufft_2d — NUFFT-accelerated reciprocal sum for 2D log kernel
// ============================================================================
//
// Same splitting as ewald_direct_2d:
//   G_short(r) = E1(α² r²) / (4π)
//   G_long(r)  = (1/V) Σ_{k≠0} w(k) e^{+i k·r},  w(k) = exp(-k²/(4α²))/k²
//   φ_self[i] += q_i · (-(γ + 2 log α) / (4π))
//
// The reciprocal sum is computed as:
//   S(k) = Σ_j q_j e^{-i k·r_j}            [type-1 NUFFT, N points → grid]
//   φ_j  = Re{ (1/V) Σ_k w(k) S(k) e^{+i k·r_j} }   [type-2 NUFFT]
// Forces (per-component d):
//   f_{j,d} = q_j · Im{ Σ_k k_d · w(k) · S(k) · e^{+i k·r_j} / V }
//
// Real-space sum is currently O(N²) with image-shell truncation (same as
// ewald_direct_2d). Can be replaced with a 2D cell list if needed.

EwaldResult2D ewald_nufft_2d(
    const Mat2 &cell,
    const std::vector<Vec2> &positions,
    const std::vector<double> &charges,
    double s,
    double nufft_tol)
{
    using std::size_t;
    int N = (int)positions.size();
    double V = std::abs(cell[0][0] * cell[1][1] - cell[0][1] * cell[1][0]);

    // Balanced default: same choice as ewald_direct_2d.
    double alpha_balanced = std::sqrt(M_PI / V);

    // Density-based cap: keep real-space neighbor count ≈ target_nbr so that
    // the cell list stays effective for large cells.  In 2D, the expected
    // number of neighbors within r_cut is π·r_cut²·ρ (2D number density).
    // Solving π·(s/α)²·(N/V) = target_nbr  →  α_density = s·√(π·N/(V·target_nbr)).
    double target_nbr = 500.0;
    double alpha_density = s * std::sqrt(M_PI * N / (V * target_nbr));

    double alpha = std::max(alpha_balanced, alpha_density);
    double r_cut = s / alpha;
    double k_max = 2.0 * s * alpha;

    EwaldResult2D result;
    result.potentials.assign(N, 0.0);
    result.forces.assign(N, Vec2{0.0, 0.0});
    result.energy = 0.0;

    // Reciprocal basis (same convention as ewald_direct_2d: no 2π factor).
    double a0x = cell[0][0], a0y = cell[0][1];
    double a1x = cell[1][0], a1y = cell[1][1];
    double det = a0x * a1y - a0y * a1x;
    Vec2 b0{ a1y / det, -a1x / det};
    Vec2 b1{-a0y / det,  a0x / det};

    auto norm2v = [](Vec2 v) { return std::sqrt(v[0]*v[0] + v[1]*v[1]); };
    double cell_height[2] = {
        std::abs(det) / norm2v({a1x, a1y}),
        std::abs(det) / norm2v({a0x, a0y})
    };
    int nmax[2];
    for (int d = 0; d < 2; ++d)
        nmax[d] = (int)std::ceil(r_cut / cell_height[d]) + 1;

    double r_cut2 = r_cut * r_cut;
    double inv4pi = 1.0 / (4.0 * M_PI);

    // ==================================================================
    // 1. Real-space sum (cell-list accelerated).
    // ==================================================================
    CellList2D cl(cell, r_cut, positions);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        cl.for_each_neighbor(i, [&](int j, const Vec2 &dr, double r2) {
            // dr = pos[j] + image - pos[i]; old loop used -dr.
            double G_s = exp_int_E1(alpha * alpha * r2) * inv4pi;
            result.potentials[i] += charges[j] * G_s;
            double fmag = charges[i] * charges[j]
                * std::exp(-alpha * alpha * r2) / (2.0 * M_PI * r2);
            result.forces[i][0] -= fmag * dr[0];
            result.forces[i][1] -= fmag * dr[1];
        });
    }

    // ==================================================================
    // 2. Reciprocal-space sum via 2D NUFFT.
    // ==================================================================
    // Fractional coords scaled to [0, 2π). For cell A = [a0 a1] (columns),
    // A^{-T} is the reciprocal matrix (rows = b0, b1). For a point x,
    // k·x with k = 2π (m0 b0 + m1 b1) gives 2π(m · A^{-1} x), so the
    // NUFFT coord is 2π · (A^{-1} x) per component.
    std::vector<double> xj_nu(N), yj_nu(N);
    for (int i = 0; i < N; ++i) {
        // Solve A · s = x  for s (fractional coord). A is 2×2; invert inline.
        double x = positions[i][0], y = positions[i][1];
        double s0 = ( a1y * x - a1x * y) / det;
        double s1 = (-a0y * x + a0x * y) / det;
        xj_nu[i] = 2.0 * M_PI * s0;
        yj_nu[i] = 2.0 * M_PI * s1;
    }

    // Grid size: choose (ms, mt) so that the k-grid covers the k_max disk.
    // Grid index m ∈ [-ms/2, ms/2) × [-mt/2, mt/2) maps to
    //   k = 2π · (m0 b0 + m1 b1),
    // so we need |m_d| ≤ k_max / (2π |b_d|).
    int mmax[2];
    mmax[0] = (int)std::ceil(k_max / (2.0 * M_PI * norm2v(b0))) + 1;
    mmax[1] = (int)std::ceil(k_max / (2.0 * M_PI * norm2v(b1))) + 1;
    // DUCC likes even grid sizes.
    auto round_up_even = [](int x) { return (x + 1) / 2 * 2; };
    int ms = round_up_even(std::max(2, 2 * mmax[0]));
    int mt = round_up_even(std::max(2, 2 * mmax[1]));
    size_t M = (size_t)ms * mt;

    // Build coord array {t, s} per point (slowest first).
    std::vector<double> coords((size_t)N * 2);
    for (int j = 0; j < N; ++j) {
        coords[2*(size_t)j + 0] = yj_nu[j];  // slow axis
        coords[2*(size_t)j + 1] = xj_nu[j];  // fast axis
    }

    // Type-1 NUFFT: S(m) = Σ_j q_j exp(-i m · x_j)
    std::vector<std::complex<double>> cj(N);
    for (int i = 0; i < N; ++i) cj[i] = charges[i];

    std::vector<std::complex<double>> Sgrid;
    nufft_type1_2d(coords, cj, ms, mt, nufft_tol, /*isign=*/-1, Sgrid);

    // Build w(k) grid and k-vectors.
    double inv4a2 = 1.0 / (4.0 * alpha * alpha);
    double k_max2 = k_max * k_max;

    std::vector<double> wgrid(M, 0.0), kxgrid(M, 0.0), kygrid(M, 0.0);
    for (int j = 0; j < mt; ++j) {
        int m1 = j - mt/2;
        for (int i = 0; i < ms; ++i) {
            int m0 = i - ms/2;
            size_t idx = (size_t)j * ms + i;
            if (m0 == 0 && m1 == 0) continue;     // gauge: k=0 → 0
            double kx = 2.0 * M_PI * (m0 * b0[0] + m1 * b1[0]);
            double ky = 2.0 * M_PI * (m0 * b0[1] + m1 * b1[1]);
            double k2 = kx*kx + ky*ky;
            if (k2 > k_max2) continue;
            wgrid[idx]  = std::exp(-k2 * inv4a2) / (k2 * V);
            kxgrid[idx] = kx;
            kygrid[idx] = ky;
        }
    }

    // Energy from reciprocal: 0.5 Σ w(k) |S(k)|²
    double E_recip = 0.0;
    for (size_t k = 0; k < M; ++k) E_recip += wgrid[k] * std::norm(Sgrid[k]);
    E_recip *= 0.5;

    // Type-2 NUFFT of w(m)·S(m) for potential (isign = +1)
    std::vector<std::complex<double>> wS(M);
    for (size_t k = 0; k < M; ++k) wS[k] = wgrid[k] * Sgrid[k];
    {
        std::vector<std::complex<double>> pot_recip;
        nufft_type2_2d(coords, pot_recip, ms, mt, nufft_tol, /*isign=*/+1, wS);
        for (int i = 0; i < N; ++i)
            result.potentials[i] += pot_recip[i].real();
    }

    // Forces: 2× type-2 NUFFT of k_d · w(m) · S(m), keep imag part.
    for (int d = 0; d < 2; ++d) {
        const double *kd = (d == 0) ? kxgrid.data() : kygrid.data();
        std::vector<std::complex<double>> Gd(M);
        for (size_t k = 0; k < M; ++k) Gd[k] = kd[k] * wS[k];
        std::vector<std::complex<double>> force_d;
        nufft_type2_2d(coords, force_d, ms, mt, nufft_tol, /*isign=*/+1, Gd);
        for (int j = 0; j < N; ++j)
            result.forces[j][d] += charges[j] * force_d[j].imag();
    }

    // ==================================================================
    // 3. Self-correction and energy.
    // ==================================================================
    const double gamma_em = 0.5772156649015329;
    const double self_const = -(gamma_em + 2.0 * std::log(alpha)) / (4.0 * M_PI);
    for (int i = 0; i < N; ++i)
        result.potentials[i] += charges[i] * self_const;

    double E = 0.0;
    for (int i = 0; i < N; ++i) E += charges[i] * result.potentials[i];
    result.energy = 0.5 * E;

    return result;
}

// ─────────────────────────────────────────────────────────────────────────
// ewald_nufft_2d_targets — subset-target variant for accuracy references
// at large N. Real-space loop is over M targets only (via CellList2D);
// reciprocal type-2 NUFFT runs at M target positions (not N sources).
// Forces and energy are not computed.
// ─────────────────────────────────────────────────────────────────────────

std::vector<double> ewald_nufft_2d_targets(
    const Mat2 &cell,
    const std::vector<Vec2> &positions,
    const std::vector<double> &charges,
    const std::vector<int> &target_indices,
    double s,
    double nufft_tol)
{
    using std::size_t;
    int N = (int)positions.size();
    int Mt = (int)target_indices.size();
    double V = std::abs(cell[0][0] * cell[1][1] - cell[0][1] * cell[1][0]);

    double alpha_balanced = std::sqrt(M_PI / V);
    double target_nbr = 500.0;
    double alpha_density = s * std::sqrt(M_PI * N / (V * target_nbr));
    double alpha = std::max(alpha_balanced, alpha_density);
    double r_cut = s / alpha;
    double k_max = 2.0 * s * alpha;

    std::vector<double> out(Mt, 0.0);

    double a0x = cell[0][0], a0y = cell[0][1];
    double a1x = cell[1][0], a1y = cell[1][1];
    double det = a0x * a1y - a0y * a1x;
    Vec2 b0{ a1y / det, -a1x / det};
    Vec2 b1{-a0y / det,  a0x / det};

    auto norm2v = [](Vec2 v) { return std::sqrt(v[0]*v[0] + v[1]*v[1]); };
    double inv4pi = 1.0 / (4.0 * M_PI);

    // ==================================================================
    // 1. Real-space sum at target indices only (cell-list accelerated).
    // ==================================================================
    CellList2D cl(cell, r_cut, positions);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < Mt; ++m) {
        int i = target_indices[m];
        double acc = 0.0;
        cl.for_each_neighbor(i, [&](int j, const Vec2 &dr, double r2) {
            (void)dr;
            double G_s = exp_int_E1(alpha * alpha * r2) * inv4pi;
            acc += charges[j] * G_s;
        });
        out[m] = acc;
    }

    // ==================================================================
    // 2. Reciprocal-space sum via 2D NUFFT.
    //    Type-1 still runs over all N sources (to build S(k));
    //    Type-2 runs at M target positions only.
    // ==================================================================
    std::vector<double> coords_src((size_t)N * 2);
    for (int j = 0; j < N; ++j) {
        double x = positions[j][0], y = positions[j][1];
        double s0 = ( a1y * x - a1x * y) / det;
        double s1 = (-a0y * x + a0x * y) / det;
        coords_src[2*(size_t)j + 0] = 2.0 * M_PI * s1;  // slow axis
        coords_src[2*(size_t)j + 1] = 2.0 * M_PI * s0;  // fast axis
    }

    int mmax0 = (int)std::ceil(k_max / (2.0 * M_PI * norm2v(b0))) + 1;
    int mmax1 = (int)std::ceil(k_max / (2.0 * M_PI * norm2v(b1))) + 1;
    auto round_up_even = [](int x) { return (x + 1) / 2 * 2; };
    int ms = round_up_even(std::max(2, 2 * mmax0));
    int mt = round_up_even(std::max(2, 2 * mmax1));
    size_t Mk = (size_t)ms * mt;

    std::vector<std::complex<double>> cj(N);
    for (int i = 0; i < N; ++i) cj[i] = charges[i];

    std::vector<std::complex<double>> Sgrid;
    nufft_type1_2d(coords_src, cj, ms, mt, nufft_tol, /*isign=*/-1, Sgrid);

    double inv4a2 = 1.0 / (4.0 * alpha * alpha);
    double k_max2 = k_max * k_max;

    std::vector<std::complex<double>> wS(Mk);
    for (int j = 0; j < mt; ++j) {
        int m1 = j - mt/2;
        for (int i = 0; i < ms; ++i) {
            int m0 = i - ms/2;
            size_t idx = (size_t)j * ms + i;
            if (m0 == 0 && m1 == 0) { wS[idx] = 0.0; continue; }
            double kx = 2.0 * M_PI * (m0 * b0[0] + m1 * b1[0]);
            double ky = 2.0 * M_PI * (m0 * b0[1] + m1 * b1[1]);
            double k2 = kx*kx + ky*ky;
            if (k2 > k_max2) { wS[idx] = 0.0; continue; }
            double w = std::exp(-k2 * inv4a2) / (k2 * V);
            wS[idx] = w * Sgrid[idx];
        }
    }

    // Target positions in NUFFT frac coords.
    std::vector<double> coords_trg((size_t)Mt * 2);
    for (int m = 0; m < Mt; ++m) {
        int i = target_indices[m];
        double x = positions[i][0], y = positions[i][1];
        double s0 = ( a1y * x - a1x * y) / det;
        double s1 = (-a0y * x + a0x * y) / det;
        coords_trg[2*(size_t)m + 0] = 2.0 * M_PI * s1;  // slow axis
        coords_trg[2*(size_t)m + 1] = 2.0 * M_PI * s0;  // fast axis
    }

    std::vector<std::complex<double>> pot_recip;
    nufft_type2_2d(coords_trg, pot_recip, ms, mt, nufft_tol, /*isign=*/+1, wS);
    for (int m = 0; m < Mt; ++m)
        out[m] += pot_recip[m].real();

    // ==================================================================
    // 3. Self-correction (only at targets).
    // ==================================================================
    const double gamma_em = 0.5772156649015329;
    const double self_const = -(gamma_em + 2.0 * std::log(alpha)) / (4.0 * M_PI);
    for (int m = 0; m < Mt; ++m) {
        int i = target_indices[m];
        out[m] += charges[i] * self_const;
    }

    return out;
}

} // namespace pdmk
