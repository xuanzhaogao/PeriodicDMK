#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <pdmk/cubic_cover.hpp>
#include <cmath>

using namespace pdmk;

TEST_CASE("cubic_cover: cubic cell identity frame") {
    // h = 2*I, frame = identity, m=n=p=1, s=2.0
    Mat3 h = {{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}}};
    Lattice lat(h);

    Vec3 e1 = {1, 0, 0};
    Vec3 e2 = {0, 1, 0};
    Vec3 e3 = {0, 0, 1};

    CubicCover cover(lat, 1, 1, 1, e1, e2, e3, 2.0);

    CHECK(cover.grid_m() == 1);
    CHECK(cover.grid_n() == 1);
    CHECK(cover.grid_p() == 1);
    CHECK(cover.padded_m() == 3);
    CHECK(cover.padded_n() == 3);
    CHECK(cover.padded_p() == 3);
    CHECK(cover.num_cubes() == 27);
    CHECK(cover.cube_size() == doctest::Approx(2.0));

    // Inner cube (1,1,1) center should be near (1,1,1).
    // Vertices of h=2I are {0..2}^3.  Minimum projections are all 0.
    // inner_origin = (0,0,0), padded_origin = (-2,-2,-2).
    // cube_origin(1,1,1) = (-2,-2,-2) + 2*(1,0,0) + 2*(0,1,0) + 2*(0,0,1) = (0,0,0).
    // cube_center(1,1,1) = (0,0,0) + 0.5*2*(1,1,1) = (1,1,1).
    Vec3 c = cover.cube_center(1, 1, 1);
    CHECK(c[0] == doctest::Approx(1.0));
    CHECK(c[1] == doctest::Approx(1.0));
    CHECK(c[2] == doctest::Approx(1.0));
}

TEST_CASE("cubic_cover: cube indexing round-trip") {
    Mat3 h = {{{3, 0, 0}, {0.5, 2.8, 0}, {0.3, 0.4, 2.5}}};
    Lattice lat(h);

    Vec3 e1 = {1, 0, 0};
    Vec3 e2 = {0, 1, 0};
    Vec3 e3 = {0, 0, 1};

    int m = 2, n = 3, p = 4;
    CubicCover cover(lat, m, n, p, e1, e2, e3, 1.5);

    int total = cover.num_cubes();
    CHECK(total == (m + 2) * (n + 2) * (p + 2));

    for (int i = 0; i <= m + 1; ++i) {
        for (int j = 0; j <= n + 1; ++j) {
            for (int k = 0; k <= p + 1; ++k) {
                int idx = cover.cube_index(i, j, k);
                CHECK(idx >= 0);
                CHECK(idx < total);

                auto [ri, rj, rk] = cover.cube_ijk(idx);
                CHECK(ri == i);
                CHECK(rj == j);
                CHECK(rk == k);
            }
        }
    }
}

TEST_CASE("cubic_cover: is_inner / is_padding") {
    Mat3 h = {{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}}};
    Lattice lat(h);

    Vec3 e1 = {1, 0, 0};
    Vec3 e2 = {0, 1, 0};
    Vec3 e3 = {0, 0, 1};

    int m = 2, n = 3, p = 2;
    CubicCover cover(lat, m, n, p, e1, e2, e3, 1.0);

    int inner_count = 0;
    int padding_count = 0;

    for (int i = 0; i <= m + 1; ++i) {
        for (int j = 0; j <= n + 1; ++j) {
            for (int k = 0; k <= p + 1; ++k) {
                bool inner = cover.is_inner(i, j, k);
                bool padding = cover.is_padding(i, j, k);
                // Mutually exclusive and exhaustive.
                CHECK(inner != padding);

                if (inner) {
                    CHECK(i >= 1);
                    CHECK(i <= m);
                    CHECK(j >= 1);
                    CHECK(j <= n);
                    CHECK(k >= 1);
                    CHECK(k <= p);
                    ++inner_count;
                } else {
                    ++padding_count;
                }
            }
        }
    }

    CHECK(inner_count == m * n * p);
    CHECK(inner_count + padding_count == cover.num_cubes());
}

TEST_CASE("cubic_cover: cube centers are s apart") {
    Mat3 h = {{{3, 0, 0}, {0.5, 2.8, 0}, {0.3, 0.4, 2.5}}};
    Lattice lat(h);

    Vec3 e1 = {1, 0, 0};
    Vec3 e2 = {0, 1, 0};
    Vec3 e3 = {0, 0, 1};

    int m = 2, n = 2, p = 2;
    Real s = 1.5;
    CubicCover cover(lat, m, n, p, e1, e2, e3, s);

    // Adjacent along e1 direction.
    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= n + 1; ++j) {
            for (int k = 0; k <= p + 1; ++k) {
                Vec3 c1 = cover.cube_center(i, j, k);
                Vec3 c2 = cover.cube_center(i + 1, j, k);
                CHECK(norm(c2 - c1) == doctest::Approx(s).epsilon(1e-12));
            }
        }
    }

    // Adjacent along e2 direction.
    for (int i = 0; i <= m + 1; ++i) {
        for (int j = 0; j <= n; ++j) {
            for (int k = 0; k <= p + 1; ++k) {
                Vec3 c1 = cover.cube_center(i, j, k);
                Vec3 c2 = cover.cube_center(i, j + 1, k);
                CHECK(norm(c2 - c1) == doctest::Approx(s).epsilon(1e-12));
            }
        }
    }

    // Adjacent along e3 direction.
    for (int i = 0; i <= m + 1; ++i) {
        for (int j = 0; j <= n + 1; ++j) {
            for (int k = 0; k <= p; ++k) {
                Vec3 c1 = cover.cube_center(i, j, k);
                Vec3 c2 = cover.cube_center(i, j, k + 1);
                CHECK(norm(c2 - c1) == doctest::Approx(s).epsilon(1e-12));
            }
        }
    }
}

TEST_CASE("cubic_cover: parallelepiped vertices inside inner cubes") {
    // h = 2*I, identity frame, m=n=p=1, s=2.
    Mat3 h = {{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}}};
    Lattice lat(h);

    Vec3 e1 = {1, 0, 0};
    Vec3 e2 = {0, 1, 0};
    Vec3 e3 = {0, 0, 1};

    CubicCover cover(lat, 1, 1, 1, e1, e2, e3, 2.0);

    // The inner box spans from cube_origin(1,1,1) to cube_origin(1,1,1) + s*(e1+e2+e3).
    // i.e. from (0,0,0) to (2,2,2).
    // The 8 parallelepiped vertices are the corners of [0,2]^3,
    // so they should all lie within (or on the boundary of) the inner box.

    Vec3 u = lat.lattice_vector(0);
    Vec3 v = lat.lattice_vector(1);
    Vec3 w = lat.lattice_vector(2);

    Vec3 zero = {0.0, 0.0, 0.0};
    Vec3 vertices[8] = {zero, u, v, w, u + v, u + w, v + w, u + v + w};

    // Inner box origin in frame coords: (0,0,0). Inner box end: (m*s, n*s, p*s) = (2,2,2).
    // A vertex is in the inner box if its frame projection lies in [0, m*s] x [0, n*s] x [0, p*s].
    // We need to compute: frame_coord = (dot(vert, e1), dot(vert, e2), dot(vert, e3))
    // and then shift by -inner_origin_frame = -(min_x, min_y, min_z).
    // For identity frame and h=2I, min_x = min_y = min_z = 0.

    Real inner_Lx = cover.grid_m() * cover.cube_size();
    Real inner_Ly = cover.grid_n() * cover.cube_size();
    Real inner_Lz = cover.grid_p() * cover.cube_size();

    // Padded origin in frame coords = inner_origin - s
    // So inner_origin = padded_origin + s*(e1+e2+e3)
    Vec3 inner_origin = cover.padded_origin() + cover.cube_size() * (e1 + e2 + e3);

    for (const auto &vert : vertices) {
        // Project vertex relative to inner_origin into the frame.
        Vec3 rel = vert - inner_origin;
        Real fx = dot(rel, e1);
        Real fy = dot(rel, e2);
        Real fz = dot(rel, e3);

        // Should be in [0, inner_L] (within tolerance).
        CHECK(fx >= -1e-12);
        CHECK(fx <= inner_Lx + 1e-12);
        CHECK(fy >= -1e-12);
        CHECK(fy <= inner_Ly + 1e-12);
        CHECK(fz >= -1e-12);
        CHECK(fz <= inner_Lz + 1e-12);
    }
}

TEST_CASE("cubic_cover: triclinic cell with rotated frame") {
    // Triclinic lattice with a non-identity (but still axis-aligned) test.
    Mat3 h = {{{3.0, 0.0, 0.0}, {0.5, 2.8, 0.0}, {0.3, 0.4, 2.5}}};
    Lattice lat(h);

    Vec3 e1 = {1, 0, 0};
    Vec3 e2 = {0, 1, 0};
    Vec3 e3 = {0, 0, 1};

    int m = 3, n = 3, p = 3;
    Real s = 1.5;
    CubicCover cover(lat, m, n, p, e1, e2, e3, s);

    // All 8 parallelepiped vertices should lie inside the inner box.
    Vec3 u = lat.lattice_vector(0);
    Vec3 v = lat.lattice_vector(1);
    Vec3 w = lat.lattice_vector(2);

    Vec3 zero = {0.0, 0.0, 0.0};
    Vec3 vertices[8] = {zero, u, v, w, u + v, u + w, v + w, u + v + w};

    Real inner_Lx = cover.grid_m() * cover.cube_size();
    Real inner_Ly = cover.grid_n() * cover.cube_size();
    Real inner_Lz = cover.grid_p() * cover.cube_size();

    Vec3 inner_origin = cover.padded_origin() + cover.cube_size() * (e1 + e2 + e3);

    for (const auto &vert : vertices) {
        Vec3 rel = vert - inner_origin;
        Real fx = dot(rel, e1);
        Real fy = dot(rel, e2);
        Real fz = dot(rel, e3);

        CHECK(fx >= -1e-12);
        CHECK(fx <= inner_Lx + 1e-12);
        CHECK(fy >= -1e-12);
        CHECK(fy <= inner_Ly + 1e-12);
        CHECK(fz >= -1e-12);
        CHECK(fz <= inner_Lz + 1e-12);
    }
}
