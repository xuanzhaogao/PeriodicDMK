#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <dmk/pbc.hpp>
#include <dmk/util.hpp>
#include <dmk.h>
#include <vector>
#include <random>
#include <cmath>

TEST_CASE("far_field_nufft<double,2> smoke: finite output on square cell") {
    constexpr int DIM = 2;
    double umat[4] = {1.0, 0.0,
                      0.0, 1.0};
    int n = 16;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> U(0.05, 0.95);
    std::vector<double> x(DIM*n), q(n);
    for (int i = 0; i < n; ++i) {
        x[2*i+0] = U(rng); x[2*i+1] = U(rng);
        q[i] = (i < n/2 ? 1.0 : -1.0);  // neutral
    }
    pdmk_params p{};
    p.n_dim = DIM; p.eps = 1e-6; p.kernel = DMK_LAPLACE;
    double beta = (double)dmk::util::calc_bandlimiting(p);
    double rc = 0.25;

    std::vector<double> pot(n, 0.0);
    dmk::pbc::far_field_nufft<double, DIM>(
        umat, rc, p.eps, beta, n, x.data(), q.data(),
        pot.data(), 0.0, nullptr);

    double nrm2 = 0.0;
    for (double v : pot) {
        CHECK(std::isfinite(v));
        nrm2 += v * v;
    }
    CHECK(nrm2 > 0.0);
}

TEST_CASE("far_field_nufft<double,2> rejects QP (bloch != nullptr)") {
    constexpr int DIM = 2;
    double umat[4] = {1.0, 0.0, 0.0, 1.0};
    double bloch[2] = {0.1, 0.1};
    std::vector<double> x = {0.5, 0.5};
    std::vector<double> q = {1.0};
    std::vector<double> pot(1, 0.0);
    CHECK_THROWS(dmk::pbc::far_field_nufft<double, DIM>(
        umat, 0.25, 1e-6, 1.0, 1, x.data(), q.data(),
        pot.data(), 0.0, bloch));
}
