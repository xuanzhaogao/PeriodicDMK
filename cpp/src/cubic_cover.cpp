#include <pdmk/cubic_cover.hpp>

#include <algorithm>
#include <limits>

namespace pdmk {

CubicCover::CubicCover(const Lattice &lat,
                       int m, int n, int p,
                       const Vec3 &e1, const Vec3 &e2, const Vec3 &e3,
                       Real s)
    : lat_(&lat), m_(m), n_(n), p_(p),
      e1_(e1), e2_(e2), e3_(e3), s_(s) {

    // Compute padded_origin following divide.jl reconstruct_layout:
    // 1. Get the 8 vertices of the parallelepiped {0, u, v, w, u+v, u+w, v+w, u+v+w}.
    Vec3 u = lat.lattice_vector(0);
    Vec3 v = lat.lattice_vector(1);
    Vec3 w = lat.lattice_vector(2);

    Vec3 zero = {0.0, 0.0, 0.0};
    Vec3 vertices[8] = {
        zero,
        u,
        v,
        w,
        u + v,
        u + w,
        v + w,
        u + v + w
    };

    // 2. Project each vertex into the frame; find minimum in each direction.
    Real min_x = std::numeric_limits<Real>::max();
    Real min_y = std::numeric_limits<Real>::max();
    Real min_z = std::numeric_limits<Real>::max();

    for (const auto &vert : vertices) {
        Real fx = dot(vert, e1_);
        Real fy = dot(vert, e2_);
        Real fz = dot(vert, e3_);
        min_x = std::min(min_x, fx);
        min_y = std::min(min_y, fy);
        min_z = std::min(min_z, fz);
    }

    // 3. Inner origin in world coords.
    Vec3 inner_origin = min_x * e1_ + min_y * e2_ + min_z * e3_;

    // 4. Padded origin is one cube before in each direction.
    padded_origin_ = inner_origin - s_ * e1_ - s_ * e2_ - s_ * e3_;
}

// ── Grid dimensions ────────────────────────────────────────────────────────

int CubicCover::grid_m() const { return m_; }
int CubicCover::grid_n() const { return n_; }
int CubicCover::grid_p() const { return p_; }
int CubicCover::padded_m() const { return m_ + 2; }
int CubicCover::padded_n() const { return n_ + 2; }
int CubicCover::padded_p() const { return p_ + 2; }
int CubicCover::num_cubes() const { return (m_ + 2) * (n_ + 2) * (p_ + 2); }
Real CubicCover::cube_size() const { return s_; }

// ── Frame ──────────────────────────────────────────────────────────────────

const Vec3 &CubicCover::frame_e1() const { return e1_; }
const Vec3 &CubicCover::frame_e2() const { return e2_; }
const Vec3 &CubicCover::frame_e3() const { return e3_; }

// ── Cube indexing ──────────────────────────────────────────────────────────

int CubicCover::cube_index(int i, int j, int k) const {
    // Row-major: i varies slowest, k varies fastest.
    return i * (n_ + 2) * (p_ + 2) + j * (p_ + 2) + k;
}

std::tuple<int, int, int> CubicCover::cube_ijk(int idx) const {
    int np = (n_ + 2) * (p_ + 2);
    int i = idx / np;
    int rem = idx % np;
    int j = rem / (p_ + 2);
    int k = rem % (p_ + 2);
    return {i, j, k};
}

// ── Cube geometry ──────────────────────────────────────────────────────────

Vec3 CubicCover::cube_origin(int i, int j, int k) const {
    return padded_origin_ + (i * s_) * e1_ + (j * s_) * e2_ + (k * s_) * e3_;
}

Vec3 CubicCover::cube_center(int i, int j, int k) const {
    return cube_origin(i, j, k) + (0.5 * s_) * (e1_ + e2_ + e3_);
}

// ── Inner / padding ────────────────────────────────────────────────────────

bool CubicCover::is_inner(int i, int j, int k) const {
    return i >= 1 && i <= m_ && j >= 1 && j <= n_ && k >= 1 && k <= p_;
}

bool CubicCover::is_padding(int i, int j, int k) const {
    return !is_inner(i, j, k);
}

// ── Origin and lattice ─────────────────────────────────────────────────────

const Vec3 &CubicCover::padded_origin() const { return padded_origin_; }

const Lattice &CubicCover::lattice() const { return *lat_; }

} // namespace pdmk
