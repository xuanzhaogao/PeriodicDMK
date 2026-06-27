#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <pdmk/periodic_dmk.hpp>
#include <pdmk/types.hpp>
#include <vector>
#include <complex>
#include <random>
#include <cmath>

namespace {

std::vector<pdmk::Vec3> random_positions(int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.02, 0.98);
    std::vector<pdmk::Vec3> p(n);
    for (int i = 0; i < n; ++i) p[i] = {U(rng), U(rng), U(rng)};
    return p;
}

}  // namespace

TEST_CASE("QP bloch=0 matches fully periodic Laplace 3D") {
    pdmk::Mat3 cell{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
    int n = 50;
    auto pos = random_positions(n, 3);
    std::mt19937_64 rng(4);
    std::uniform_real_distribution<double> Uq(-1.0, 1.0);
    std::vector<double> q(n);
    for (auto& v : q) v = Uq(rng);

    pdmk::PeriodicDMKTreeT<double, 3> tree_plain(cell, pos, 6);
    auto phi_real = tree_plain.evaluate(q);

    std::array<double, 3> bloch{0.0, 0.0, 0.0};
    pdmk::PeriodicDMKTreeT<double, 3> tree_qp(cell, pos, 6, 300, 4, bloch);
    auto phi_c = tree_qp.evaluate_complex(q);

    for (int i = 0; i < n; ++i) {
        CHECK(std::abs(phi_c[i].imag()) < 1e-5);
        CHECK(std::abs(phi_c[i].real() - phi_real[i]) < 1e-5);
    }
}

TEST_CASE("QP self-consistency across nz (rc)") {
    pdmk::Mat3 cell{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
    int n = 40;
    auto pos = random_positions(n, 5);
    std::mt19937_64 rng(6);
    std::uniform_real_distribution<double> Uq(-1.0, 1.0);
    std::vector<double> q(n);
    for (auto& v : q) v = Uq(rng);

    std::array<double, 3> bloch{2.0, -1.0, 0.5};
    pdmk::PeriodicDMKTreeT<double, 3> t1(cell, pos, 6, 300, /*nz=*/4, bloch);
    pdmk::PeriodicDMKTreeT<double, 3> t2(cell, pos, 6, 300, /*nz=*/8, bloch);
    auto a = t1.evaluate_complex(q);
    auto b = t2.evaluate_complex(q);
    for (int i = 0; i < n; ++i) {
        CHECK(std::abs(a[i] - b[i]) < 1e-4);
    }
}

// A quasi-periodicity test (u(x + a_k) = e^{i β·a_k} u(x)) would require
// constructing the tree with sources shifted outside the fundamental cell,
// which PeriodicDMKTreeT does not support. The bloch=0 reduction and the
// nz-independence check above exercise the QP plumbing end-to-end.
