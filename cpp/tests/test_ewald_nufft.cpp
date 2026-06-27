#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <pdmk/ewald_nufft.hpp>
#include <pdmk/lattice.hpp>
#include <pdmk/types.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace pdmk;

// ============================================================================
// Helpers
// ============================================================================

static double l2_rel(const std::vector<double> &a, const std::vector<double> &b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return std::sqrt(num / den);
}

static void generate_random_system(
    const Lattice &lat, int N, unsigned seed,
    std::vector<Vec3> &pos, std::vector<double> &charges)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    pos.resize(N);
    charges.resize(N);
    double total_q = 0;
    for (int i = 0; i < N; ++i) {
        Vec3 s = {u01(rng), u01(rng), u01(rng)};
        pos[i] = lat.to_cartesian(s);
        charges[i] = (i % 2 == 0) ? 1.0 : -1.0;
        charges[i] *= (1.0 + 0.1 * u01(rng));
        total_q += charges[i];
    }
    // Enforce charge neutrality
    charges[N - 1] -= total_q;
}

// ============================================================================
// Tests: cross-validate NUFFT vs brute-force
// ============================================================================

TEST_CASE("NaCl Madelung constant — NUFFT vs direct") {
    double a = 1.0;
    Mat3 h = {{{a, 0, 0}, {0, a, 0}, {0, 0, a}}};
    Lattice lat(h);

    std::vector<Vec3> pos = {
        {0.0, 0.0, 0.0}, {0.5*a, 0.5*a, 0.0},
        {0.5*a, 0.0, 0.5*a}, {0.0, 0.5*a, 0.5*a},
        {0.5*a, 0.0, 0.0}, {0.0, 0.5*a, 0.0},
        {0.0, 0.0, 0.5*a}, {0.5*a, 0.5*a, 0.5*a},
    };
    std::vector<double> q = {1, 1, 1, 1, -1, -1, -1, -1};

    auto res_nufft  = ewald_nufft(lat, pos, q, 5.5);
    auto res_direct = ewald_direct(lat, pos, q, 5.5);

    double M_ref = 1.7475645946331822;
    double M_nufft  = -res_nufft.energy * a / 8.0;
    double M_direct = -res_direct.energy * a / 8.0;

    printf("  Madelung: ref=%.12f nufft=%.12f direct=%.12f\n", M_ref, M_nufft, M_direct);
    printf("  |nufft-ref|=%.2e  |direct-ref|=%.2e  |nufft-direct|=%.2e\n",
        std::abs(M_nufft - M_ref), std::abs(M_direct - M_ref),
        std::abs(M_nufft - M_direct));

    CHECK(std::abs(M_nufft - M_ref) < 1e-8);
    CHECK(std::abs(M_nufft - M_direct) < 1e-10);

    // Potential agreement
    double l2 = l2_rel(res_nufft.potentials, res_direct.potentials);
    printf("  Potential L2 rel error: %.2e\n", l2);
    CHECK(l2 < 1e-10);
}

TEST_CASE("Triclinic cell — NUFFT vs direct") {
    Mat3 h = {{{3.0, 0.0, 0.0}, {0.5, 2.8, 0.0}, {0.3, 0.4, 2.5}}};
    Lattice lat(h);

    std::vector<Vec3> pos;
    std::vector<double> q;
    generate_random_system(lat, 20, 42, pos, q);

    auto res_nufft  = ewald_nufft(lat, pos, q, 5.0);
    auto res_direct = ewald_direct(lat, pos, q, 5.0);

    double rel_E = std::abs(res_nufft.energy - res_direct.energy) /
                   std::abs(res_direct.energy);
    printf("  Triclinic energy: nufft=%.12f direct=%.12f rel_err=%.2e\n",
        res_nufft.energy, res_direct.energy, rel_E);
    CHECK(rel_E < 1e-10);

    double l2 = l2_rel(res_nufft.potentials, res_direct.potentials);
    printf("  Potential L2 rel error: %.2e\n", l2);
    CHECK(l2 < 1e-10);
}

TEST_CASE("Force — NUFFT vs direct") {
    Mat3 h = {{{3.0, 0.0, 0.0}, {0.5, 2.8, 0.0}, {0.3, 0.4, 2.5}}};
    Lattice lat(h);

    std::vector<Vec3> pos;
    std::vector<double> q;
    generate_random_system(lat, 8, 123, pos, q);

    auto res_nufft  = ewald_nufft(lat, pos, q, 5.0);
    auto res_direct = ewald_direct(lat, pos, q, 5.0);

    double max_ferr = 0.0;
    for (int i = 0; i < 8; ++i) {
        for (int d = 0; d < 3; ++d) {
            double err = std::abs(res_nufft.forces[i][d] - res_direct.forces[i][d]);
            max_ferr = std::max(max_ferr, err);
        }
    }
    printf("  Force max |nufft - direct|: %.2e\n", max_ferr);
    CHECK(max_ferr < 1e-8);
}

TEST_CASE("Force — finite difference check") {
    Mat3 h = {{{3.0, 0.0, 0.0}, {0.5, 2.8, 0.0}, {0.3, 0.4, 2.5}}};
    Lattice lat(h);

    std::vector<Vec3> pos;
    std::vector<double> q;
    generate_random_system(lat, 8, 456, pos, q);

    auto res = ewald_nufft(lat, pos, q, 5.0);

    double delta = 1e-5;
    double max_fd_err = 0.0;
    for (int i = 0; i < 4; ++i) {  // spot-check first 4 particles
        for (int d = 0; d < 3; ++d) {
            auto pos_p = pos;
            auto pos_m = pos;
            pos_p[i][d] += delta;
            pos_m[i][d] -= delta;
            auto Ep = ewald_nufft(lat, pos_p, q, 5.0);
            auto Em = ewald_nufft(lat, pos_m, q, 5.0);
            double fd = -(Ep.energy - Em.energy) / (2.0 * delta);
            double err = std::abs(fd - res.forces[i][d]);
            max_fd_err = std::max(max_fd_err, err);
        }
    }
    printf("  Force FD max error: %.2e\n", max_fd_err);
    CHECK(max_fd_err < 1e-5);
}

TEST_CASE("Cell-list fallback — tiny cell") {
    // Very small cell where cell heights < 2*r_cut
    double a = 1.0;
    Mat3 h = {{{a, 0, 0}, {0, a, 0}, {0, 0, a}}};
    Lattice lat(h);

    std::vector<Vec3> pos = {{0.1, 0.2, 0.3}, {0.6, 0.7, 0.8}};
    std::vector<double> q = {1.0, -1.0};

    auto res_nufft  = ewald_nufft(lat, pos, q, 4.0);
    auto res_direct = ewald_direct(lat, pos, q, 4.0);

    double rel_E = std::abs(res_nufft.energy - res_direct.energy) /
                   std::abs(res_direct.energy);
    printf("  Tiny cell energy rel error: %.2e\n", rel_E);
    CHECK(rel_E < 1e-10);
}

TEST_CASE("Large N — scaling smoke test") {
    double L = 10.0;
    Mat3 h = {{{L, 0, 0}, {0, L, 0}, {0, 0, L}}};
    Lattice lat(h);

    std::vector<Vec3> pos;
    std::vector<double> q;
    generate_random_system(lat, 1000, 789, pos, q);

    auto res = ewald_nufft(lat, pos, q, 4.0);
    // Just check it doesn't crash and energy is finite
    CHECK(std::isfinite(res.energy));
    printf("  N=1000 energy: %.8f\n", res.energy);
}
