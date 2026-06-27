#pragma once

#include <pdmk/types.hpp>

namespace pdmk {

/// Lattice geometry: stores the cell matrix h = [a1 | a2 | a3] and its inverse.
/// h[col][row]: column j is lattice vector a_j.
class Lattice {
  public:
    /// Construct from cell matrix h = [a1 | a2 | a3].
    explicit Lattice(const Mat3 &h);

    /// Cell volume |det(h)|.
    Real volume() const;

    /// Cell matrix h.
    const Mat3 &cell_matrix() const;

    /// Inverse cell matrix h^{-1}.
    const Mat3 &inverse_matrix() const;

    /// Reciprocal lattice matrix G = 2*pi * h^{-T}.
    Mat3 reciprocal_matrix() const;

    /// Convert Cartesian coordinates to fractional: s = h^{-1} * r.
    Vec3 to_fractional(const Vec3 &r) const;

    /// Convert fractional coordinates to Cartesian: r = h * s.
    Vec3 to_cartesian(const Vec3 &s) const;

    /// Wrap a Cartesian point into the cell: fractional coords mapped to [0,1),
    /// then converted back to Cartesian.
    Vec3 wrap_into_cell(const Vec3 &r) const;

    /// Perpendicular face-to-face distances of the cell.
    std::array<Real, 3> cell_heights() const;

    /// Column j of h (lattice vector a_j), j in {0,1,2}.
    Vec3 lattice_vector(int j) const;

  private:
    Mat3 h_;      ///< Cell matrix.
    Mat3 h_inv_;  ///< Inverse cell matrix.
    Real vol_;    ///< Cell volume |det(h)|.
};

} // namespace pdmk
