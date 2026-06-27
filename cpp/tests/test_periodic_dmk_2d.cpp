#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <pdmk/periodic_dmk.hpp>
#include <pdmk/types.hpp>
#include <pdmk/ewald_nufft.hpp>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>

namespace {

// DMK 2D Laplace uses the unphysical convention G(r) = +log(r) (see
// LaplacePolyEvaluator2D in vector_kernels.hpp). To compare against the
// physical Ewald reference (which uses G_phys(r) = -log(r)/(2π)), we
// rescale the DMK output by the convention factor `-2π`.
constexpr double DMK_2D_TO_PHYSICAL = -1.0 / (2.0 * M_PI);

std::vector<double> mean_sub(std::vector<double> v) {
    double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    for (double& x : v) x -= m;
    return v;
}

double rel_l2(const std::vector<double>& test, const std::vector<double>& ref) {
    double e2 = 0, r2 = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = test[i] - ref[i];
        e2 += d*d;
        r2 += ref[i]*ref[i];
    }
    return std::sqrt(e2 / r2);
}

// Wraps a DMK 2D evaluate() output into physical convention (G=-log(r)/(2π))
// and mean-subtracts.
std::vector<double> dmk_to_physical_mean_sub(std::vector<double> dmk_out) {
    for (double& v : dmk_out) v *= DMK_2D_TO_PHYSICAL;
    return mean_sub(std::move(dmk_out));
}

} // namespace

TEST_CASE("PeriodicDMKTreeT<double,2>: square cell matches Ewald") {
    pdmk::Mat2 cell{{{1.0, 0.0}, {0.0, 1.0}}};
    int n = 30;
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> U(0.02, 0.98);
    std::vector<pdmk::Vec2> pos(n);
    std::vector<double> q(n);
    for (int i = 0; i < n; ++i) {
        pos[i] = {U(rng), U(rng)};
        q[i] = (i < n/2 ? 1.0 : -1.0);  // neutral
    }

    // Reference: 2D Ewald for the physical G(r) = -log(r)/(2π) kernel,
    // k=0 mode pinned (zero mean on neutral sets).
    auto phi_ref_full = pdmk::ewald_direct_2d(cell, pos, q, /*s=*/5.0).potentials;
    auto phi_ref = mean_sub(phi_ref_full);

    pdmk::PeriodicDMKTreeT<double, 2> tree(cell, pos, /*n_digits=*/6);
    auto phi_test = dmk_to_physical_mean_sub(tree.evaluate(q));

    INFO("rel L2 = " << rel_l2(phi_test, phi_ref));
    CHECK(rel_l2(phi_test, phi_ref) < 1e-4);
}

TEST_CASE("PeriodicDMKTreeT<double,2>: zero charges -> zero output") {
    pdmk::Mat2 cell{{{1.0, 0.0}, {0.0, 1.0}}};
    std::vector<pdmk::Vec2> pos = {{0.1, 0.2}, {0.7, 0.3}, {0.4, 0.8}};
    std::vector<double> q(pos.size(), 0.0);
    pdmk::PeriodicDMKTreeT<double, 2> tree(cell, pos, 6);
    auto phi = tree.evaluate(q);
    for (double v : phi) CHECK(std::abs(v) < 1e-12);
}

TEST_CASE("PeriodicDMKTreeT<double,2>: skewed 2D cell matches Ewald") {
    // Non-orthogonal: aspect 2:1 with skew
    pdmk::Mat2 cell{{{2.0, 0.0}, {0.4, 1.0}}};
    int n = 24;
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<pdmk::Vec2> pos(n);
    std::vector<double> q(n);
    for (int i = 0; i < n; ++i) {
        double u = U(rng), v = U(rng);
        pos[i] = {cell[0][0]*u + cell[1][0]*v,
                  cell[0][1]*u + cell[1][1]*v};
        q[i] = (i < n/2 ? 1.0 : -1.0);
    }

    auto phi_ref_full = pdmk::ewald_direct_2d(cell, pos, q, 5.0).potentials;
    auto phi_ref = mean_sub(phi_ref_full);

    pdmk::PeriodicDMKTreeT<double, 2> tree(cell, pos, 6);
    auto phi_test = dmk_to_physical_mean_sub(tree.evaluate(q));

    INFO("rel L2 = " << rel_l2(phi_test, phi_ref));
    CHECK(rel_l2(phi_test, phi_ref) < 1e-4);
}
