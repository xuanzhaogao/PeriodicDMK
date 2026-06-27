#pragma once

#include <array>
#include <cmath>
#include <complex>

namespace pdmk {

template <typename T> using Vec3T = std::array<T, 3>;
template <typename T> using Mat3T = std::array<std::array<T, 3>, 3>;
template <typename T> using ComplexT = std::complex<T>;

// Backward-compatible aliases. Existing call sites unchanged.
using Real = double;
using Complex = ComplexT<double>;
using Mat3 = Mat3T<Real>;
using Vec3 = Vec3T<Real>;

// 2D geometry types.
using Vec2 = std::array<double, 2>;
using Mat2 = std::array<std::array<double, 2>, 2>;

// DIM-generic type traits.
template <int DIM> struct dim_types;
template <> struct dim_types<2> { using Vec = Vec2; using Mat = Mat2; };
template <> struct dim_types<3> { using Vec = Vec3; using Mat = Mat3; };

template <int DIM> using VecND = typename dim_types<DIM>::Vec;
template <int DIM> using MatND = typename dim_types<DIM>::Mat;

// ── Vec3 arithmetic (templated; deducible from argument) ─────────────────────

template <typename T>
inline T dot(const Vec3T<T> &a, const Vec3T<T> &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
inline Vec3T<T> cross(const Vec3T<T> &a, const Vec3T<T> &b) {
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

template <typename T>
inline T norm(const Vec3T<T> &a) {
    return std::sqrt(dot(a, a));
}

template <typename T>
inline Vec3T<T> operator+(const Vec3T<T> &a, const Vec3T<T> &b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

template <typename T>
inline Vec3T<T> operator-(const Vec3T<T> &a, const Vec3T<T> &b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// Scalar-vector multiplication. The scalar type is deduced independently so
// `int * Vec3T<double>` keeps working via implicit conversion of the scalar
// to the vector's element type.
template <typename S, typename T>
inline Vec3T<T> operator*(S s, const Vec3T<T> &v) {
    const T st = static_cast<T>(s);
    return {st * v[0], st * v[1], st * v[2]};
}

template <typename S, typename T>
inline Vec3T<T> operator*(const Vec3T<T> &v, S s) {
    const T st = static_cast<T>(s);
    return {v[0] * st, v[1] * st, v[2] * st};
}

// Global identifier for a box across all cubes.
struct GlobalBoxId {
    int cube_idx;
    int box_idx;

    bool operator==(const GlobalBoxId &o) const {
        return cube_idx == o.cube_idx && box_idx == o.box_idx;
    }
    bool operator<(const GlobalBoxId &o) const {
        if (cube_idx != o.cube_idx) return cube_idx < o.cube_idx;
        return box_idx < o.box_idx;
    }
};

} // namespace pdmk
