#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <pdmk/lattice.hpp>
#include <cmath>

using namespace pdmk;

TEST_CASE("lattice: cubic cell") {
    // Identity matrix -> cubic unit cell
    Mat3 h = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    Lattice lat(h);

    CHECK(lat.volume() == doctest::Approx(1.0));

    // Fractional of (0.5, 0.5, 0.5) should be (0.5, 0.5, 0.5)
    Vec3 r = {0.5, 0.5, 0.5};
    Vec3 s = lat.to_fractional(r);
    CHECK(s[0] == doctest::Approx(0.5));
    CHECK(s[1] == doctest::Approx(0.5));
    CHECK(s[2] == doctest::Approx(0.5));

    // Reciprocal matrix diagonal should be 2*pi
    Mat3 G = lat.reciprocal_matrix();
    constexpr double two_pi = 2.0 * M_PI;
    CHECK(G[0][0] == doctest::Approx(two_pi));
    CHECK(G[1][1] == doctest::Approx(two_pi));
    CHECK(G[2][2] == doctest::Approx(two_pi));
    // Off-diagonals should be zero
    CHECK(G[1][0] == doctest::Approx(0.0).epsilon(1e-14));
    CHECK(G[0][1] == doctest::Approx(0.0).epsilon(1e-14));

    // Cell heights all = 1
    auto heights = lat.cell_heights();
    CHECK(heights[0] == doctest::Approx(1.0));
    CHECK(heights[1] == doctest::Approx(1.0));
    CHECK(heights[2] == doctest::Approx(1.0));
}

TEST_CASE("lattice: triclinic cell") {
    // h = [[3,0,0],[0.5,2.8,0],[0.3,0.4,2.5]]  (columns)
    // Column-major: h[col][row]
    Mat3 h = {{{3.0, 0.0, 0.0}, {0.5, 2.8, 0.0}, {0.3, 0.4, 2.5}}};
    Lattice lat(h);

    // volume = 3.0 * 2.8 * 2.5 = 21.0
    // (upper triangular -> det = product of diagonal)
    CHECK(lat.volume() == doctest::Approx(21.0));

    // Round-trip: Cartesian -> fractional -> Cartesian
    Vec3 r = {1.7, 0.9, 1.3};
    Vec3 s = lat.to_fractional(r);
    Vec3 r2 = lat.to_cartesian(s);
    CHECK(r2[0] == doctest::Approx(r[0]).epsilon(1e-12));
    CHECK(r2[1] == doctest::Approx(r[1]).epsilon(1e-12));
    CHECK(r2[2] == doctest::Approx(r[2]).epsilon(1e-12));

    // Cell heights > 0
    auto heights = lat.cell_heights();
    CHECK(heights[0] > 0.0);
    CHECK(heights[1] > 0.0);
    CHECK(heights[2] > 0.0);
}

TEST_CASE("lattice: wrap_into_cell") {
    Mat3 h = {{{3.0, 0.0, 0.0}, {0.5, 2.8, 0.0}, {0.3, 0.4, 2.5}}};
    Lattice lat(h);

    // A point far outside the cell
    Vec3 r = {10.0, -5.0, 8.0};
    Vec3 wrapped = lat.wrap_into_cell(r);

    // Wrapped fractional coords should be in [0, 1)
    Vec3 s = lat.to_fractional(wrapped);
    CHECK(s[0] >= 0.0);
    CHECK(s[0] < 1.0);
    CHECK(s[1] >= 0.0);
    CHECK(s[1] < 1.0);
    CHECK(s[2] >= 0.0);
    CHECK(s[2] < 1.0);

    // The difference between original and wrapped should be an integer
    // linear combination of lattice vectors, i.e., fractional difference is integer.
    Vec3 s_orig = lat.to_fractional(r);
    for (int i = 0; i < 3; ++i) {
        double diff = s_orig[i] - s[i];
        CHECK(std::abs(diff - std::round(diff)) < 1e-12);
    }
}

TEST_CASE("lattice: reciprocal lattice orthogonality") {
    // Triclinic cell
    Mat3 h = {{{3.0, 0.0, 0.0}, {0.5, 2.8, 0.0}, {0.3, 0.4, 2.5}}};
    Lattice lat(h);

    Mat3 G = lat.reciprocal_matrix();
    constexpr double two_pi = 2.0 * M_PI;

    // b_i . a_j = 2*pi * delta_{ij}
    // b_i = column i of G, a_j = column j of h
    for (int i = 0; i < 3; ++i) {
        Vec3 bi = {G[i][0], G[i][1], G[i][2]};
        for (int j = 0; j < 3; ++j) {
            Vec3 aj = lat.lattice_vector(j);
            double expected = (i == j) ? two_pi : 0.0;
            CHECK(dot(bi, aj) == doctest::Approx(expected).epsilon(1e-12));
        }
    }
}
