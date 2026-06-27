#pragma once

#include <pdmk/types.hpp>
#include <pdmk/lattice.hpp>

#include <vector>

namespace pdmk {

/// Result of an Ewald summation: per-particle potentials, forces, and total energy.
struct EwaldResult {
    std::vector<double> potentials;  ///< u[i] = Σ'_{j,n} q_j / |x_i - x_j - R_n|
    std::vector<Vec3>   forces;      ///< f[i] = -∇_i E (Cartesian force on particle i)
    double energy;                   ///< 0.5 * Σ_i q_i * u[i]
};

/// O(N log N) Ewald summation using DUCC NUFFT + cell-list neighbor search.
///
/// Computes the periodic Coulomb potential, forces, and energy for a set of
/// point charges in an arbitrary (triclinic) periodic cell.
///
/// @param lat       Lattice geometry (cell vectors).
/// @param positions Cartesian positions of N particles.
/// @param charges   Charges of N particles.
/// @param s         Accuracy parameter (error ~ exp(-s²)). Default 4.8 ≈ 10 digits.
/// @param nufft_tol NUFFT relative tolerance. Default 1e-12.
/// @return EwaldResult with potentials, forces, and energy.
EwaldResult ewald_nufft(
    const Lattice &lat,
    const std::vector<Vec3> &positions,
    const std::vector<double> &charges,
    double s = 4.8,
    double nufft_tol = 1e-12);

/// Same physics as `ewald_nufft` but returns potentials only at the
/// sources indexed by `target_indices` (subset of [0, N)). The real-space
/// sum runs over those M targets only (cell-list over N sources); the
/// reciprocal side still does one type-1 NUFFT over N sources, then a
/// type-2 NUFFT at M target positions only. Forces and energy are NOT
/// computed. Intended for accuracy references at very large N where
/// evaluating at all particles is infeasible.
std::vector<double> ewald_nufft_targets(
    const Lattice &lat,
    const std::vector<Vec3> &positions,
    const std::vector<double> &charges,
    const std::vector<int> &target_indices,
    double s = 4.8,
    double nufft_tol = 1e-12);

/// O(N²) brute-force Ewald reference. Useful for small-N cross-validation.
///
/// @param lat       Lattice geometry.
/// @param positions Cartesian positions of N particles.
/// @param charges   Charges of N particles.
/// @param s         Accuracy parameter. Default 5.0.
/// @return EwaldResult with potentials, forces, and energy.
EwaldResult ewald_direct(
    const Lattice &lat,
    const std::vector<Vec3> &positions,
    const std::vector<double> &charges,
    double s = 5.0);

/// Result of a 2D Ewald sum for the log Green's function.
struct EwaldResult2D {
    std::vector<double> potentials;  ///< u[i], gauge W(k=0)=0
    std::vector<Vec2>   forces;      ///< f[i] = -∇_i E
    double energy;
};

/// O(N²) brute-force 2D Ewald for the **physical** Green's function
/// G(r) = -log(r)/(2π). Output is mean-zero for neutral charge sets
/// (k=0 reciprocal mode pinned to zero).
///
/// Split via Frullani identity:
///   G_short(r) = E1(α² r²) / (4π)
///   G_long(r)  = (1/V) Σ_{k≠0} exp(-k²/(4α²)) / k² · e^{+i k·r}
///   φ_self    += q_i · (γ + 2 log α) / (-4π)
/// The self-correction closes the α-invariance (verified to 6e-15 across
/// α ∈ [0.5, 5]). Supports general 2D triclinic cells.
///
/// @param alpha_override 0 → default sqrt(π/V); > 0 → use this α.
EwaldResult2D ewald_direct_2d(
    const Mat2 &cell,
    const std::vector<Vec2> &positions,
    const std::vector<double> &charges,
    double s = 5.0,
    double alpha_override = 0.0);

/// O(N log N) NUFFT-accelerated 2D Ewald for G(r) = -log(r)/(2π).
/// Same splitting and self-correction as ewald_direct_2d, but the
/// reciprocal sum Σ_{k≠0} w(k) S(k) e^{+ik·r_i} is evaluated via a pair
/// of 2D DUCC NUFFTs (type-1 for S(k), type-2 for pot). Real-space sum
/// is currently O(N²) with image-shell truncation at r_cut = s/α (this
/// dominates only at very large N or small α; can be replaced with a
/// cell list later).
///
/// @param s         controls real/reciprocal cutoffs (error ~ exp(-s²)).
/// @param nufft_tol relative NUFFT tolerance.
EwaldResult2D ewald_nufft_2d(
    const Mat2 &cell,
    const std::vector<Vec2> &positions,
    const std::vector<double> &charges,
    double s = 4.8,
    double nufft_tol = 1e-12);

/// Same physics as `ewald_nufft_2d` but returns potentials only at the
/// sources indexed by `target_indices` (subset of [0, N)). Skips both
/// the O(N) real-space contribution to non-targets and the N-sized
/// type-2 NUFFT — only 200-ish target positions are fed to u2nu.
/// Forces and energy are NOT computed.
///
/// Intended for accuracy references at very large N where evaluating
/// at all particles is infeasible.
std::vector<double> ewald_nufft_2d_targets(
    const Mat2 &cell,
    const std::vector<Vec2> &positions,
    const std::vector<double> &charges,
    const std::vector<int> &target_indices,
    double s = 5.5,
    double nufft_tol = 1e-12);

} // namespace pdmk
