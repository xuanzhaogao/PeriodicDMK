#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <pdmk/ewald_nufft.hpp>
#include <pdmk/types.hpp>

#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace pdmk;

static std::vector<double> mean_sub(std::vector<double> v) {
    double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    for (auto &x : v) x -= m;
    return v;
}

static double l2_rel(const std::vector<double> &a, const std::vector<double> &b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return std::sqrt(num / den);
}

static double vec_rel(const std::vector<Vec2> &a, const std::vector<Vec2> &b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        double dx = a[i][0] - b[i][0], dy = a[i][1] - b[i][1];
        num += dx*dx + dy*dy;
        den += b[i][0]*b[i][0] + b[i][1]*b[i][1];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

TEST_CASE("ewald_nufft_2d matches ewald_direct_2d (square cell, neutral)") {
    Mat2 cell{{{1.0, 0.0}, {0.0, 1.0}}};
    int N = 80;
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> U(0.02, 0.98);
    std::vector<Vec2> pos(N);
    std::vector<double> q(N);
    for (int i = 0; i < N; ++i) {
        pos[i] = {U(rng), U(rng)};
        q[i] = (i < N/2 ? 1.0 : -1.0);
    }

    auto d = ewald_direct_2d(cell, pos, q, /*s=*/5.0);
    auto n = ewald_nufft_2d (cell, pos, q, /*s=*/5.0, /*tol=*/1e-12);

    double pot_err = l2_rel(mean_sub(n.potentials), mean_sub(d.potentials));
    double f_err   = vec_rel(n.forces, d.forces);
    double e_err   = std::abs(n.energy - d.energy) /
                     std::max(std::abs(d.energy), 1e-30);

    INFO("pot rel L2 = " << pot_err
         << ", force rel L2 = " << f_err
         << ", energy rel = " << e_err);
    CHECK(pot_err < 1e-10);
    CHECK(f_err   < 1e-10);
    CHECK(e_err   < 1e-10);
}

TEST_CASE("ewald_nufft_2d matches ewald_direct_2d (skewed cell)") {
    Mat2 cell{{{2.0, 0.0}, {0.4, 1.0}}};
    int N = 60;
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> V(0.0, 1.0);
    std::vector<Vec2> pos(N);
    std::vector<double> q(N);
    for (int i = 0; i < N; ++i) {
        double u = V(rng), v = V(rng);
        pos[i] = {cell[0][0]*u + cell[1][0]*v,
                  cell[0][1]*u + cell[1][1]*v};
        q[i] = (i < N/2 ? 1.0 : -1.0);
    }

    auto d = ewald_direct_2d(cell, pos, q, /*s=*/5.0);
    auto n = ewald_nufft_2d (cell, pos, q, /*s=*/5.0, /*tol=*/1e-12);

    double pot_err = l2_rel(mean_sub(n.potentials), mean_sub(d.potentials));
    double f_err   = vec_rel(n.forces, d.forces);

    INFO("pot rel L2 = " << pot_err << ", force rel L2 = " << f_err);
    CHECK(pot_err < 1e-10);
    CHECK(f_err   < 1e-10);
}

TEST_CASE("ewald_nufft_2d zero charge gives zero output") {
    Mat2 cell{{{1.0, 0.0}, {0.0, 1.0}}};
    std::vector<Vec2> pos = {{0.1, 0.2}, {0.7, 0.3}, {0.4, 0.8}};
    std::vector<double> q(pos.size(), 0.0);
    auto r = ewald_nufft_2d(cell, pos, q, 5.0, 1e-12);
    for (double v : r.potentials) CHECK(std::abs(v) < 1e-14);
    for (auto  f : r.forces)      { CHECK(std::abs(f[0]) < 1e-14); CHECK(std::abs(f[1]) < 1e-14); }
}
