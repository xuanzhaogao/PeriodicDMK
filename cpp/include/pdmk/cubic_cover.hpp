#pragma once

#include <pdmk/lattice.hpp>
#include <pdmk/types.hpp>
#include <tuple>

namespace pdmk {

/// CubicCover: geometry of how a parallelepiped cell is covered by a grid of
/// equal-sized cubes aligned to an orthonormal frame (e1, e2, e3).
///
/// The inner grid has dimensions (m, n, p).  The padded grid adds one extra
/// cube on each side in every direction, yielding (m+2) x (n+2) x (p+2).
///
/// Cover parameters (frame, cube size, inner dimensions) are computed
/// externally by divide.jl and passed in at construction time.
class CubicCover {
  public:
    /// Construct from cover parameters.
    /// @param lat   Lattice describing the parallelepiped cell.
    /// @param m,n,p Inner grid dimensions along e1, e2, e3.
    /// @param e1,e2,e3 Orthonormal frame (cube axes).
    /// @param s     Cube side length.
    CubicCover(const Lattice &lat,
               int m, int n, int p,
               const Vec3 &e1, const Vec3 &e2, const Vec3 &e3,
               Real s);

    // ── Grid dimensions ────────────────────────────────────────────────────

    int grid_m() const;      ///< Inner dimension along e1.
    int grid_n() const;      ///< Inner dimension along e2.
    int grid_p() const;      ///< Inner dimension along e3.
    int padded_m() const;    ///< m + 2.
    int padded_n() const;    ///< n + 2.
    int padded_p() const;    ///< p + 2.
    int num_cubes() const;   ///< (m+2)*(n+2)*(p+2).
    Real cube_size() const;  ///< s.

    // ── Frame ──────────────────────────────────────────────────────────────

    const Vec3 &frame_e1() const;
    const Vec3 &frame_e2() const;
    const Vec3 &frame_e3() const;

    // ── Cube indexing ──────────────────────────────────────────────────────

    /// Row-major linear index from (i, j, k).
    /// i in [0, m+1], j in [0, n+1], k in [0, p+1].
    int cube_index(int i, int j, int k) const;

    /// Inverse of cube_index: linear index -> (i, j, k).
    std::tuple<int, int, int> cube_ijk(int idx) const;

    // ── Cube geometry ──────────────────────────────────────────────────────

    /// World-space corner of cube (i,j,k) with smallest frame coordinates.
    Vec3 cube_origin(int i, int j, int k) const;

    /// World-space center of cube (i,j,k).
    Vec3 cube_center(int i, int j, int k) const;

    // ── Inner / padding classification ─────────────────────────────────────

    /// True if (i,j,k) is an inner cube: i in [1,m], j in [1,n], k in [1,p].
    bool is_inner(int i, int j, int k) const;

    /// True if (i,j,k) is a padding cube (complement of is_inner).
    bool is_padding(int i, int j, int k) const;

    // ── Origin ─────────────────────────────────────────────────────────────

    /// The origin of the padded box — corner of cube (0,0,0).
    const Vec3 &padded_origin() const;

    /// The underlying lattice.
    const Lattice &lattice() const;

  private:
    const Lattice *lat_;   ///< Non-owning pointer to lattice.
    int m_, n_, p_;        ///< Inner grid dimensions.
    Vec3 e1_, e2_, e3_;    ///< Orthonormal frame.
    Real s_;               ///< Cube side length.
    Vec3 padded_origin_;   ///< Corner of cube (0,0,0).
};

} // namespace pdmk
