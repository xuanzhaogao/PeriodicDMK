#include <pdmk/lattice.hpp>
#include <cmath>
#include <stdexcept>

namespace pdmk {

namespace {

/// Compute determinant of a 3x3 column-major matrix.
Real det3(const Mat3 &h) {
    return h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
         - h[1][0] * (h[0][1] * h[2][2] - h[0][2] * h[2][1])
         + h[2][0] * (h[0][1] * h[1][2] - h[0][2] * h[1][1]);
}

/// Compute inverse of a 3x3 column-major matrix.
/// Returns h_inv such that h_inv[col][row] stores the inverse in the same convention.
Mat3 inv3(const Mat3 &h, Real d) {
    Real inv_d = 1.0 / d;
    Mat3 hi;

    // Cofactor matrix C[i][j] = cofactor of h[row=i][col=j] (classical).
    // For column-major storage h[col][row], the inverse is:
    //   h_inv[col][row] = cofactor(row, col) / det
    // where cofactor(i,j) = (-1)^{i+j} * minor(i,j).

    // h_inv[0][0] = (h[1][1]*h[2][2] - h[1][2]*h[2][1]) / det
    hi[0][0] = (h[1][1] * h[2][2] - h[1][2] * h[2][1]) * inv_d;
    // h_inv[0][1] = -(h[0][1]*h[2][2] - h[0][2]*h[2][1]) / det
    hi[0][1] = -(h[0][1] * h[2][2] - h[0][2] * h[2][1]) * inv_d;
    // h_inv[0][2] = (h[0][1]*h[1][2] - h[0][2]*h[1][1]) / det
    hi[0][2] = (h[0][1] * h[1][2] - h[0][2] * h[1][1]) * inv_d;

    // h_inv[1][0] = -(h[1][0]*h[2][2] - h[1][2]*h[2][0]) / det
    hi[1][0] = -(h[1][0] * h[2][2] - h[1][2] * h[2][0]) * inv_d;
    // h_inv[1][1] = (h[0][0]*h[2][2] - h[0][2]*h[2][0]) / det
    hi[1][1] = (h[0][0] * h[2][2] - h[0][2] * h[2][0]) * inv_d;
    // h_inv[1][2] = -(h[0][0]*h[1][2] - h[0][2]*h[1][0]) / det
    hi[1][2] = -(h[0][0] * h[1][2] - h[0][2] * h[1][0]) * inv_d;

    // h_inv[2][0] = (h[1][0]*h[2][1] - h[1][1]*h[2][0]) / det
    hi[2][0] = (h[1][0] * h[2][1] - h[1][1] * h[2][0]) * inv_d;
    // h_inv[2][1] = -(h[0][0]*h[2][1] - h[0][1]*h[2][0]) / det
    hi[2][1] = -(h[0][0] * h[2][1] - h[0][1] * h[2][0]) * inv_d;
    // h_inv[2][2] = (h[0][0]*h[1][1] - h[0][1]*h[1][0]) / det
    hi[2][2] = (h[0][0] * h[1][1] - h[0][1] * h[1][0]) * inv_d;

    return hi;
}

/// Matrix-vector product for column-major Mat3: result[i] = sum_j m[j][i]*v[j].
Vec3 matvec(const Mat3 &m, const Vec3 &v) {
    return {m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2]};
}

} // anonymous namespace

Lattice::Lattice(const Mat3 &h) : h_(h) {
    Real d = det3(h_);
    if (std::abs(d) < 1e-30) {
        throw std::invalid_argument("Lattice: cell matrix is singular (zero determinant)");
    }
    vol_ = std::abs(d);
    h_inv_ = inv3(h_, d);
}

Real Lattice::volume() const { return vol_; }

const Mat3 &Lattice::cell_matrix() const { return h_; }

const Mat3 &Lattice::inverse_matrix() const { return h_inv_; }

Mat3 Lattice::reciprocal_matrix() const {
    // G = 2*pi * h^{-T}
    // h_inv_ is stored column-major: h_inv_[col][row].
    // Transpose means: G[col][row] = 2*pi * h_inv_[row][col].
    constexpr Real two_pi = 2.0 * M_PI;
    Mat3 G;
    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 3; ++row) {
            G[col][row] = two_pi * h_inv_[row][col];
        }
    }
    return G;
}

Vec3 Lattice::to_fractional(const Vec3 &r) const {
    return matvec(h_inv_, r);
}

Vec3 Lattice::to_cartesian(const Vec3 &s) const {
    return matvec(h_, s);
}

Vec3 Lattice::wrap_into_cell(const Vec3 &r) const {
    Vec3 s = to_fractional(r);
    for (int i = 0; i < 3; ++i) {
        s[i] -= std::floor(s[i]);
    }
    return to_cartesian(s);
}

std::array<Real, 3> Lattice::cell_heights() const {
    Vec3 a0 = lattice_vector(0);
    Vec3 a1 = lattice_vector(1);
    Vec3 a2 = lattice_vector(2);
    return {vol_ / norm(cross(a1, a2)),
            vol_ / norm(cross(a2, a0)),
            vol_ / norm(cross(a0, a1))};
}

Vec3 Lattice::lattice_vector(int j) const {
    return h_[j];
}

} // namespace pdmk
