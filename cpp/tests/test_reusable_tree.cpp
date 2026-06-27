#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vector>
#include <random>
#include <cmath>
#include <pdmk/periodic_dmk.hpp>

TEST_CASE("PeriodicDMKTree reuses tree across charge sets") {
    using namespace pdmk;
    const double L = 3.0;
    Mat3 cell{{ {L,0,0}, {0,L,0}, {0,0,L} }};
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> U(0.05 * L, 0.95 * L);
    std::normal_distribution<double> G(0, 1);
    const int N = 64;
    std::vector<Vec3> positions(N);
    for (auto& p : positions) p = {U(rng), U(rng), U(rng)};

    std::vector<double> q1(N), q2(N);
    for (int i = 0; i < N; ++i) { q1[i] = G(rng); q2[i] = G(rng); }

    PeriodicDMKTreeT<double> tree(cell, positions, 6, 16, 4);
    auto p1 = tree.evaluate(q1);
    auto p2 = tree.evaluate(q2);
    auto p1b = tree.evaluate(q1);
    // Repeated evaluate(q1) should agree to many digits, but not necessarily
    // bit-exactly: FINUFFT's threaded spread/interp can reorder reductions
    // across calls, producing last-bit differences.
    for (int i = 0; i < N; ++i) {
        CHECK(p1[i] == doctest::Approx(p1b[i]).epsilon(1e-12));
    }
    // q1 != q2 should give different potentials (not trivially zero).
    double diff = 0;
    for (int i = 0; i < N; ++i) diff += std::abs(p1[i] - p2[i]);
    CHECK(diff > 1e-10);
}
