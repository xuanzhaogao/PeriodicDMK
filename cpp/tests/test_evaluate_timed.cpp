#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <pdmk/periodic_dmk.hpp>
#include <pdmk/eval_timing.hpp>
#include <pdmk/types.hpp>

#include <chrono>
#include <cmath>
#include <random>
#include <vector>

using namespace pdmk;

TEST_CASE("evaluate_timed matches evaluate and partitions wall time") {
    const int N = 500;
    const double L = 3.0;
    Mat3 cell = {{{L, 0, 0}, {0, L, 0}, {0, 0, L}}};

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> U(0.01, 0.99);
    std::normal_distribution<double> Z(0.0, 1.0);

    std::vector<Vec3> positions(N);
    std::vector<double> charges(N);
    double q_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        positions[i] = {L*U(rng), L*U(rng), L*U(rng)};
        charges[i] = Z(rng);
        q_sum += charges[i];
    }
    charges[0] -= q_sum;

    PeriodicDMKTreeT<double> tree(cell, positions,
                                   /*n_digits=*/6,
                                   /*n_per_leaf=*/280,
                                   /*nz=*/7);

    auto pot_plain = tree.evaluate(charges);

    EvalTiming t{};
    auto t_outer0 = std::chrono::steady_clock::now();
    auto pot_timed = tree.evaluate_timed(charges, t);
    auto t_outer1 = std::chrono::steady_clock::now();
    double outer_ms = std::chrono::duration<double, std::milli>(t_outer1 - t_outer0).count();
    double sum_ms = t.dmk_ms + t.direct_ms + t.nufft_ms;
    CHECK(sum_ms <= outer_ms);
    CHECK(sum_ms >= 0.95 * outer_ms);

    REQUIRE(pot_plain.size() == (size_t)N);
    REQUIRE(pot_timed.size() == (size_t)N);

    double diff2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        double d = pot_plain[i] - pot_timed[i];
        diff2 += d*d;
        ref2  += pot_plain[i]*pot_plain[i];
    }
    double rel = std::sqrt(diff2 / ref2);
    CHECK(rel < 1e-10);

    CHECK(t.dmk_ms    > 0.0);
    CHECK(t.direct_ms > 0.0);
    CHECK(t.nufft_ms  > 0.0);
    CHECK(t.dmk_ms    < 1e6);
    CHECK(t.direct_ms < 1e6);
    CHECK(t.nufft_ms  < 1e6);
}
