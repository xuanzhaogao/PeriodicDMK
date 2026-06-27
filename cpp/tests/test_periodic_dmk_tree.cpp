#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vector>
#include <cmath>
#include <random>
#include <pdmk/periodic_dmk.hpp>
#include <pdmk/ewald_nufft.hpp>

TEST_CASE("PeriodicDMKTreeT<double> matches ewald_nufft") {
    using namespace pdmk;
    const double L = 3.0;
    Mat3 cell{{ {L,0,0}, {0,L,0}, {0,0,L} }};

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> U(0.03 * L, 0.97 * L);
    std::normal_distribution<double> G(0, 1);

    const int N = 100;
    std::vector<Vec3> positions(N);
    std::vector<double> charges(N);
    double csum = 0;
    for (int i = 0; i < N; ++i) {
        positions[i] = {U(rng), U(rng), U(rng)};
        charges[i] = G(rng);
        csum += charges[i];
    }
    charges[0] -= csum;

    PeriodicDMKTreeT<double> tree(cell, positions, 6, 20, 4);
    auto pot1 = tree.evaluate(charges);
    // Second eval to exercise update_charges path
    auto pot2 = tree.evaluate(charges);

    REQUIRE(pot1.size() == (size_t)N);
    // Repeated evaluate() agrees to many digits but not bit-exactly under
    // FINUFFT (threaded spread/interp can reorder reductions across calls).
    for (int i = 0; i < N; ++i)
        CHECK(pot1[i] == doctest::Approx(pot2[i]).epsilon(1e-12));

    // Reference: O(N log N) Ewald via NUFFT.
    Lattice lat(cell);
    EwaldResult ref = ewald_nufft(lat, positions, charges, /*s=*/5.0);
    const auto& potref = ref.potentials;

    double num = 0, den = 0;
    for (int i = 0; i < N; ++i) {
        double d = pot1[i] - potref[i];
        num += d * d;
        den += potref[i] * potref[i];
    }
    CHECK(std::sqrt(num / den) < 1e-4);
}
