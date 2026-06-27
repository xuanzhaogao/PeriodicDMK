#include <pdmk/cubic_cover_optimizer.hpp>

#include <array>
#include <cmath>
#include <stdexcept>

namespace pdmk {

namespace {

inline double norm(const Vec3 &v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

inline Vec3 scale(const Vec3 &v, double s) {
    return {v[0] * s, v[1] * s, v[2] * s};
}

inline Vec3 sub(const Vec3 &a, const Vec3 &b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

} // anonymous namespace

CoverResult optimize_cubic_cover(const Lattice &lattice, int nz) {
    if (nz <= 0) {
        throw std::invalid_argument("cubic cover nz must be positive");
    }

    // Gather lattice vectors.
    const Vec3 a[3] = {
        lattice.lattice_vector(0),
        lattice.lattice_vector(1),
        lattice.lattice_vector(2),
    };

    // Sort indices by descending norm (stable: earlier index wins on ties).
    std::array<int, 3> idx = {0, 1, 2};
    double norms[3] = {norm(a[0]), norm(a[1]), norm(a[2])};
    // Simple 3-element insertion sort (stable).
    if (norms[idx[0]] < norms[idx[1]]) std::swap(idx[0], idx[1]);
    if (norms[idx[1]] < norms[idx[2]]) std::swap(idx[1], idx[2]);
    if (norms[idx[0]] < norms[idx[1]]) std::swap(idx[0], idx[1]);

    const Vec3 &b1 = a[idx[0]];
    const Vec3 &b2 = a[idx[1]];

    // Gram-Schmidt orthonormal frame.
    double nb1 = norm(b1);
    if (nb1 <= 0.0) {
        throw std::invalid_argument("cubic cover: degenerate lattice (zero-norm vector)");
    }
    Vec3 e1 = scale(b1, 1.0 / nb1);

    double b2_e1 = dot(b2, e1);
    Vec3 v2 = sub(b2, scale(e1, b2_e1));
    double nv2 = norm(v2);
    if (nv2 <= 0.0) {
        throw std::invalid_argument("cubic cover: degenerate lattice (parallel vectors)");
    }
    Vec3 e2 = scale(v2, 1.0 / nv2);
    Vec3 e3 = cross(e1, e2);

    // Bounding-box widths in the rotated frame.
    double Wx = std::abs(dot(a[0], e1)) + std::abs(dot(a[1], e1)) + std::abs(dot(a[2], e1));
    double Wy = std::abs(dot(a[0], e2)) + std::abs(dot(a[1], e2)) + std::abs(dot(a[2], e2));
    double Wz = std::abs(dot(a[0], e3)) + std::abs(dot(a[1], e3)) + std::abs(dot(a[2], e3));

    // Cube side from the shortest (z) axis; derive m, n to keep cubes uniform.
    double s = Wz / static_cast<double>(nz);
    int m = static_cast<int>(std::ceil(Wx / s));
    int n = static_cast<int>(std::ceil(Wy / s));
    int p = nz;
    if (m < 1) m = 1;
    if (n < 1) n = 1;

    double V = lattice.volume();
    double cost = static_cast<double>((m + 2) * (n + 2) * (p + 2)) * (s * s * s) / V;

    return CoverResult{m, n, p, e1, e2, e3, s, cost};
}

} // namespace pdmk
