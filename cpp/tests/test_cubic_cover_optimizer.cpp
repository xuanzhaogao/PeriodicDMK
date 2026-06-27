#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <pdmk/cubic_cover_optimizer.hpp>
#include <cmath>
#include <stdexcept>

TEST_CASE("optimize_cubic_cover: cubic cell") {
    pdmk::Mat3 h;
    h[0] = {1.0, 0.0, 0.0};
    h[1] = {0.0, 1.0, 0.0};
    h[2] = {0.0, 0.0, 1.0};
    pdmk::Lattice lat(h);
    auto result = pdmk::optimize_cubic_cover(lat, 4);

    CHECK(result.m == 4);
    CHECK(result.n == 4);
    CHECK(result.p == 4);
    CHECK(result.s == doctest::Approx(0.25).epsilon(1e-12));
    CHECK(result.cost > 0.0);
    CHECK(std::isfinite(result.cost));

    // Frame should be orthonormal and right-handed.
    CHECK(pdmk::norm(result.e1) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(pdmk::norm(result.e2) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(pdmk::norm(result.e3) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(std::abs(pdmk::dot(result.e1, result.e2)) < 1e-12);
    CHECK(std::abs(pdmk::dot(result.e1, result.e3)) < 1e-12);
    CHECK(std::abs(pdmk::dot(result.e2, result.e3)) < 1e-12);
}

TEST_CASE("optimize_cubic_cover: rectangular cell has axis-aligned counts") {
    pdmk::Mat3 h;
    h[0] = {4.0, 0.0, 0.0};
    h[1] = {0.0, 2.0, 0.0};
    h[2] = {0.0, 0.0, 1.0};
    pdmk::Lattice lat(h);
    auto result = pdmk::optimize_cubic_cover(lat, 4);

    // Longest axis → m, second → n, shortest → p = nz.
    CHECK(result.p == 4);
    CHECK(result.s == doctest::Approx(0.25).epsilon(1e-12));
    CHECK(result.m == 16);
    CHECK(result.n == 8);
}

TEST_CASE("optimize_cubic_cover: slender 16x2x2 is densely covered along x") {
    pdmk::Mat3 h;
    h[0] = {16.0, 0.0, 0.0};
    h[1] = {0.0,  2.0, 0.0};
    h[2] = {0.0,  0.0, 2.0};
    pdmk::Lattice lat(h);
    auto result = pdmk::optimize_cubic_cover(lat, 4);

    CHECK(result.p == 4);
    CHECK(result.s == doctest::Approx(0.5).epsilon(1e-12));
    CHECK(result.m == 32);
    CHECK(result.n == 4);
}

TEST_CASE("optimize_cubic_cover: nz=1 is allowed") {
    pdmk::Mat3 h;
    h[0] = {2.0, 0.0, 0.0};
    h[1] = {0.0, 2.0, 0.0};
    h[2] = {0.0, 0.0, 2.0};
    pdmk::Lattice lat(h);
    auto result = pdmk::optimize_cubic_cover(lat, 1);
    CHECK(result.p == 1);
    CHECK(result.s == doctest::Approx(2.0).epsilon(1e-12));
    CHECK(result.m == 1);
    CHECK(result.n == 1);
}

TEST_CASE("optimize_cubic_cover: rejects non-positive nz") {
    pdmk::Mat3 h;
    h[0] = {1.0, 0.0, 0.0};
    h[1] = {0.3, 1.0, 0.0};
    h[2] = {0.1, 0.2, 0.8};
    pdmk::Lattice lat(h);
    CHECK_THROWS_AS(pdmk::optimize_cubic_cover(lat, 0),  std::invalid_argument);
    CHECK_THROWS_AS(pdmk::optimize_cubic_cover(lat, -1), std::invalid_argument);
}

TEST_CASE("optimize_cubic_cover: triclinic frame is orthonormal") {
    pdmk::Mat3 h;
    h[0] = {5.0, 0.0, 0.0};
    h[1] = {1.0, 4.5, 0.0};
    h[2] = {0.5, 0.8, 4.0};
    pdmk::Lattice lat(h);
    auto result = pdmk::optimize_cubic_cover(lat, 4);

    CHECK(pdmk::norm(result.e1) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(pdmk::norm(result.e2) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(pdmk::norm(result.e3) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(std::abs(pdmk::dot(result.e1, result.e2)) < 1e-12);
    CHECK(std::abs(pdmk::dot(result.e1, result.e3)) < 1e-12);
    CHECK(std::abs(pdmk::dot(result.e2, result.e3)) < 1e-12);
    CHECK(result.p == 4);
    CHECK(result.m >= 1);
    CHECK(result.n >= 1);
    CHECK(result.s > 0.0);
    CHECK(std::isfinite(result.cost));
}
