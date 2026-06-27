#pragma once

#include <pdmk/types.hpp>
#include <pdmk/lattice.hpp>

namespace pdmk {

struct CoverResult {
    int m, n, p;          ///< Inner grid dimensions
    Vec3 e1, e2, e3;      ///< Orthonormal frame (right-handed)
    double s;             ///< Cube side length
    double cost;          ///< Overhead ratio: ((m+2)(n+2)(p+2) s^3) / V
};

/// Compute the cubic cover for a parallelepiped lattice by lattice reduction.
///
/// The algorithm sorts the lattice vectors by norm (descending), aligns the
/// longest with x̂ and the second-longest into the xOy plane, then sets
/// s = Wz / nz, m = ceil(Wx / s), n = ceil(Wy / s), p = nz,
/// where Wx, Wy, Wz are the bounding-box widths in the rotated frame.
///
/// @param lattice  The cell
/// @param nz       Number of cube layers along the shortest (z) axis; nz >= 1
/// @return         Cover parameters
/// @throws std::invalid_argument if nz <= 0
CoverResult optimize_cubic_cover(const Lattice& lattice, int nz);

} // namespace pdmk
