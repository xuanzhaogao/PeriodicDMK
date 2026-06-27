#include <cmath>
#include <vector>
#include <dmk.h>
#include <dmk/pbc.hpp>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("dmk::pbc::pbcdmk_adaptive single-charge smoke") {
    const int N = 1;
    std::vector<double> avec = {2.0, 0.0, 0.0,
                                 0.0, 2.0, 0.0,
                                 0.0, 0.0, 2.0};
    std::vector<double> r_src = {0.5, 0.5, 0.5};
    std::vector<double> charge = {1.0};
    std::vector<double> pot(N, 0.0);

    pdmk_params p;
    p.n_dim = 3;
    p.eps = 1e-3;
    p.kernel = DMK_LAPLACE;
    p.pgh_src = DMK_POTENTIAL;
    p.pgh_trg = DMK_POTENTIAL;
    p.use_periodic = 1;
    p.n_per_leaf = 200;
    p.log_level = 6;

    dmk::pbc::pbcdmk_adaptive<double, 3>(p, avec.data(), -1.0,
                                           N, r_src.data(), charge.data(),
                                           pot.data());

    CHECK(std::isfinite(pot[0]));
}
