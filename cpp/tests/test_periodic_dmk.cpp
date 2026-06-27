#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vector>
#include <random>
#include <cmath>
#include <pdmk/periodic_dmk.hpp>
#include <pdmk/ewald_nufft.hpp>

using namespace pdmk;

static double rel_l2(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i]; num += d * d; den += b[i] * b[i];
    }
    return std::sqrt(num / den);
}

TEST_CASE("PeriodicDMKTree cubic-cell random vs Ewald") {
    const double L = 3.0;
    // Mat3 uses column-major convention: cell[j] is lattice vector j.
    Mat3 cell{{ {L,0,0}, {0,L,0}, {0,0,L} }};
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> U(0.03 * L, 0.97 * L);
    std::normal_distribution<double> G(0, 1);
    const int N = 200;
    std::vector<Vec3> positions(N);
    std::vector<double> charges(N);
    double cs = 0;
    for (int i = 0; i < N; ++i) {
        positions[i] = {U(rng), U(rng), U(rng)};
        charges[i] = G(rng); cs += charges[i];
    }
    charges[0] -= cs;

    PeriodicDMKTreeT<double> tree(cell, positions, 6, 20, 4);
    auto pot = tree.evaluate(charges);

    Lattice lat(cell);
    EwaldResult ref = ewald_nufft(lat, positions, charges, 5.0);
    CHECK(rel_l2(pot, ref.potentials) < 1e-4);
}

TEST_CASE("PeriodicDMKTree triclinic random vs Ewald") {
    // Column-major: cell[j][i] = component i of lattice vector j.
    // a1 = (5.0, 0.0, 0.0); a2 = (1.0, 4.5, 0.0); a3 = (0.5, 0.8, 4.0).
    Mat3 cell{{ {5.0, 0.0, 0.0}, {1.0, 4.5, 0.0}, {0.5, 0.8, 4.0} }};
    Lattice lat(cell);

    std::mt19937 rng(77);
    std::uniform_real_distribution<double> U(0.01, 0.99);
    std::normal_distribution<double> G(0, 1);
    const int N = 100;
    std::vector<Vec3> positions(N);
    std::vector<double> charges(N);
    double cs = 0;
    for (int i = 0; i < N; ++i) {
        positions[i] = lat.to_cartesian({U(rng), U(rng), U(rng)});
        charges[i] = G(rng); cs += charges[i];
    }
    charges[0] -= cs;

    PeriodicDMKTreeT<double> tree(cell, positions, 6, 20, 4);
    auto pot = tree.evaluate(charges);
    EwaldResult ref = ewald_nufft(lat, positions, charges, 5.0);
    CHECK(rel_l2(pot, ref.potentials) < 1e-4);
}
