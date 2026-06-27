/// Tests for non-uniform particle distributions and deep octrees.
/// These exercise tree paths that uniform distributions rarely reach:
///   - high max_level from dense clusters
///   - 2:1 balance across large level gaps
///   - direct interaction lists with very different leaf sizes
///   - interaction lists spanning many levels

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <pdmk/periodic_dmk.hpp>
#include <pdmk/ewald_nufft.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace pdmk;

// ============================================================================
// Helpers
// ============================================================================

/// L2 relative error between two potential vectors.
static double l2_rel(const std::vector<double> &a, const std::vector<double> &b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return std::sqrt(num / den);
}

/// Generate a charge-neutral system in fractional coords, converted to Cartesian.
static void uniform_system(
    const Lattice &lat, int N, unsigned seed,
    std::vector<Vec3> &pos, std::vector<double> &charges)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.01, 0.99);
    pos.resize(N);
    charges.resize(N);
    double Q = 0;
    for (int i = 0; i < N; ++i) {
        pos[i] = lat.to_cartesian({u01(rng), u01(rng), u01(rng)});
        charges[i] = (i % 2 == 0) ? 1.0 : -1.0;
        charges[i] *= (1.0 + 0.1 * u01(rng));
        Q += charges[i];
    }
    charges[0] -= Q;
}

/// Place N_cluster particles in a small ball of radius `r` centered at `center`,
/// plus N_bg uniform background particles. Returns charge-neutral system.
static void clustered_system(
    const Lattice &lat,
    const Vec3 &center, double r, int N_cluster,
    int N_bg, unsigned seed,
    std::vector<Vec3> &pos, std::vector<double> &charges)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.01, 0.99);
    std::normal_distribution<double> gauss(0.0, r / 3.0);

    pos.clear();
    charges.clear();

    // Cluster particles (Gaussian ball, clamped to cell)
    Mat3 h = lat.cell_matrix();
    for (int i = 0; i < N_cluster; ++i) {
        Vec3 p = {center[0] + gauss(rng),
                  center[1] + gauss(rng),
                  center[2] + gauss(rng)};
        // Wrap into cell via fractional coords
        Vec3 frac = lat.to_fractional(p);
        for (int d = 0; d < 3; ++d)
            frac[d] -= std::floor(frac[d]);
        pos.push_back(lat.to_cartesian(frac));
        charges.push_back((i % 2 == 0) ? 1.0 : -1.0);
    }

    // Background particles
    for (int i = 0; i < N_bg; ++i) {
        pos.push_back(lat.to_cartesian({u01(rng), u01(rng), u01(rng)}));
        charges.push_back((i % 2 == 0) ? 0.5 : -0.5);
    }

    // Enforce charge neutrality
    double Q = 0;
    for (auto q : charges) Q += q;
    charges[0] -= Q;
}

/// Multi-scale clustered system: several clusters of different sizes.
static void multiscale_system(
    const Lattice &lat, int N_per_cluster, unsigned seed,
    std::vector<Vec3> &pos, std::vector<double> &charges)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.01, 0.99);

    pos.clear();
    charges.clear();

    // Cluster centers in fractional coords, with varying radii
    struct ClusterDef { Vec3 frac_center; double radius; };
    std::vector<ClusterDef> clusters = {
        {{0.1, 0.1, 0.1}, 0.005},  // Very tight cluster → deep refinement
        {{0.5, 0.5, 0.5}, 0.05},   // Medium cluster
        {{0.9, 0.2, 0.8}, 0.15},   // Loose cluster
    };

    for (auto &cl : clusters) {
        Vec3 center = lat.to_cartesian(cl.frac_center);
        double L = std::cbrt(lat.volume());
        double r = cl.radius * L;
        std::normal_distribution<double> gauss(0.0, r / 3.0);

        for (int i = 0; i < N_per_cluster; ++i) {
            Vec3 p = {center[0] + gauss(rng),
                      center[1] + gauss(rng),
                      center[2] + gauss(rng)};
            Vec3 frac = lat.to_fractional(p);
            for (int d = 0; d < 3; ++d)
                frac[d] -= std::floor(frac[d]);
            pos.push_back(lat.to_cartesian(frac));
            charges.push_back((i % 2 == 0) ? 1.0 : -1.0);
        }
    }

    // Enforce charge neutrality
    double Q = 0;
    for (auto q : charges) Q += q;
    charges[0] -= Q;
}

// ============================================================================
// Tests: single tight cluster forces deep tree
// ============================================================================

TEST_CASE("nonuniform: single tight cluster — cubic cell") {
    double L = 5.0;
    Mat3 h = {{{L, 0, 0}, {0, L, 0}, {0, 0, L}}};
    Lattice lat(h);

    // 200 particles in a ball of radius 0.02*L at corner, plus 100 background
    std::vector<Vec3> pos;
    std::vector<double> charges;
    clustered_system(lat, {0.5, 0.5, 0.5}, 0.02 * L, 200, 100, 42, pos, charges);

    int N = static_cast<int>(pos.size());
    MESSAGE("N = ", N, " (200 cluster + 100 bg)");

    // Small n_per_leaf to force deeper tree
    PeriodicDMKTree tree(h, pos, 6, 20, 4);
    auto pot = tree.evaluate(charges);

    REQUIRE(pot.size() == static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        CHECK(std::isfinite(pot[i]));
    }

    // Compare against Ewald NUFFT reference
    auto ref = ewald_nufft(lat, pos, charges, 5.0);
    double err = l2_rel(pot, ref.potentials);
    MESSAGE("L2 relative error: ", err);
    CHECK(err < 1e-4);
}

TEST_CASE("nonuniform: single tight cluster — triclinic cell") {
    // Column-major: h[j] is lattice vector j.
    Mat3 h = {{{5.0, 0.0, 0.0}, {1.0, 4.5, 0.0}, {0.5, 0.8, 4.0}}};
    Lattice lat(h);

    std::vector<Vec3> pos;
    std::vector<double> charges;
    double L = std::cbrt(lat.volume());
    clustered_system(lat, lat.to_cartesian({0.3, 0.3, 0.3}), 0.02 * L,
                     200, 100, 77, pos, charges);

    int N = static_cast<int>(pos.size());

    PeriodicDMKTree tree(h, pos, 6, 20, 4);
    auto pot = tree.evaluate(charges);

    REQUIRE(pot.size() == static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        CHECK(std::isfinite(pot[i]));

    auto ref = ewald_nufft(lat, pos, charges, 5.0);
    double err = l2_rel(pot, ref.potentials);
    MESSAGE("Triclinic cluster L2 error: ", err);
    CHECK(err < 1e-4);
}

// ============================================================================
// Tests: multi-scale clusters
// ============================================================================

TEST_CASE("nonuniform: multi-scale clusters") {
    double L = 5.0;
    Mat3 h = {{{L, 0, 0}, {0, L, 0}, {0, 0, L}}};
    Lattice lat(h);

    std::vector<Vec3> pos;
    std::vector<double> charges;
    multiscale_system(lat, 100, 42, pos, charges);

    int N = static_cast<int>(pos.size());
    MESSAGE("N = ", N, " (3 clusters × 100)");

    PeriodicDMKTree tree(h, pos, 6, 15, 4);
    auto pot = tree.evaluate(charges);

    REQUIRE(pot.size() == static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        CHECK(std::isfinite(pot[i]));

    auto ref = ewald_nufft(lat, pos, charges, 5.0);
    double err = l2_rel(pot, ref.potentials);
    MESSAGE("Multi-scale L2 error: ", err);
    CHECK(err < 1e-4);
}

// ============================================================================
// Tests: very small n_per_leaf (deep tree) with moderate N
// ============================================================================

TEST_CASE("nonuniform: deep tree via small n_per_leaf") {
    double L = 3.0;
    Mat3 h = {{{L, 0, 0}, {0, L, 0}, {0, 0, L}}};
    Lattice lat(h);

    // 500 uniform particles but n_per_leaf=5 → many levels
    std::vector<Vec3> pos;
    std::vector<double> charges;
    uniform_system(lat, 500, 123, pos, charges);

    PeriodicDMKTree tree(h, pos, 6, 5, 4);
    auto pot = tree.evaluate(charges);

    REQUIRE(pot.size() == 500);
    for (int i = 0; i < 500; ++i)
        CHECK(std::isfinite(pot[i]));

    auto ref = ewald_nufft(lat, pos, charges, 5.0);
    double err = l2_rel(pot, ref.potentials);
    MESSAGE("Deep tree (n_per_leaf=5, N=500) L2 error: ", err);
    CHECK(err < 1e-4);
}

TEST_CASE("nonuniform: deep tree with clustered particles") {
    double L = 5.0;
    Mat3 h = {{{L, 0, 0}, {0, L, 0}, {0, 0, L}}};
    Lattice lat(h);

    // 300 in tight cluster + 200 bg, n_per_leaf=10 → deepest possible tree
    std::vector<Vec3> pos;
    std::vector<double> charges;
    clustered_system(lat, {2.5, 2.5, 2.5}, 0.01 * L, 300, 200, 99, pos, charges);

    int N = static_cast<int>(pos.size());

    PeriodicDMKTree tree(h, pos, 6, 10, 4);
    auto pot = tree.evaluate(charges);

    REQUIRE(pot.size() == static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        CHECK(std::isfinite(pot[i]));

    auto ref = ewald_nufft(lat, pos, charges, 5.0);
    double err = l2_rel(pot, ref.potentials);
    MESSAGE("Tight cluster + deep tree L2 error: ", err);
    CHECK(err < 1e-4);
}

// ============================================================================
// Tests: high accuracy (n_digits=9) with non-uniform
// ============================================================================

TEST_CASE("nonuniform: high accuracy (9 digits) with cluster") {
    double L = 4.0;
    Mat3 h = {{{L, 0, 0}, {0, L, 0}, {0, 0, L}}};
    Lattice lat(h);

    std::vector<Vec3> pos;
    std::vector<double> charges;
    clustered_system(lat, {2.0, 2.0, 2.0}, 0.05 * L, 100, 50, 55, pos, charges);

    int N = static_cast<int>(pos.size());

    PeriodicDMKTree tree(h, pos, 9, 20, 4);
    auto pot = tree.evaluate(charges);

    REQUIRE(pot.size() == static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        CHECK(std::isfinite(pot[i]));

    auto ref = ewald_nufft(lat, pos, charges, 5.5);
    double err = l2_rel(pot, ref.potentials);
    MESSAGE("9-digit cluster L2 error: ", err);
    CHECK(err < 1e-7);
}

// ============================================================================
// Tests: larger N non-uniform (1000+ particles)
// ============================================================================

TEST_CASE("nonuniform: 2000 particles with clusters — energy check") {
    double L = 8.0;
    Mat3 h = {{{L, 0, 0}, {0, L, 0}, {0, 0, L}}};
    Lattice lat(h);

    // 500 tight cluster + 500 medium cluster + 1000 bg
    std::vector<Vec3> pos;
    std::vector<double> charges;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> u01(0.01, 0.99);
    std::normal_distribution<double> tight(0.0, 0.05);
    std::normal_distribution<double> medium(0.0, 0.3);

    auto wrap = [&](Vec3 p) {
        Vec3 frac = lat.to_fractional(p);
        for (int d = 0; d < 3; ++d)
            frac[d] -= std::floor(frac[d]);
        return lat.to_cartesian(frac);
    };

    // Tight cluster at corner
    Vec3 c1 = lat.to_cartesian({0.05, 0.05, 0.05});
    for (int i = 0; i < 500; ++i) {
        pos.push_back(wrap({c1[0] + tight(rng), c1[1] + tight(rng), c1[2] + tight(rng)}));
        charges.push_back((i % 2 == 0) ? 1.0 : -1.0);
    }

    // Medium cluster at center
    Vec3 c2 = lat.to_cartesian({0.5, 0.5, 0.5});
    for (int i = 0; i < 500; ++i) {
        pos.push_back(wrap({c2[0] + medium(rng), c2[1] + medium(rng), c2[2] + medium(rng)}));
        charges.push_back((i % 2 == 0) ? 0.5 : -0.5);
    }

    // Background
    for (int i = 0; i < 1000; ++i) {
        pos.push_back(lat.to_cartesian({u01(rng), u01(rng), u01(rng)}));
        charges.push_back((i % 2 == 0) ? 0.3 : -0.3);
    }

    double Q = 0;
    for (auto q : charges) Q += q;
    charges[0] -= Q;

    int N = static_cast<int>(pos.size());
    MESSAGE("N = ", N);

    PeriodicDMKTree tree(h, pos, 6, 50, 4);
    auto pot = tree.evaluate(charges);

    REQUIRE(pot.size() == static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        CHECK(std::isfinite(pot[i]));

    auto ref = ewald_nufft(lat, pos, charges, 5.0);
    double err = l2_rel(pot, ref.potentials);
    MESSAGE("2000-particle clustered L2 error: ", err);
    CHECK(err < 1e-4);

    // Also check total energy
    double E_dmk = 0, E_ref = 0;
    for (int i = 0; i < N; ++i) {
        E_dmk += 0.5 * charges[i] * pot[i];
        E_ref += 0.5 * charges[i] * ref.potentials[i];
    }
    double rel_E = std::abs(E_dmk - E_ref) / std::abs(E_ref);
    MESSAGE("Energy: DMK=", E_dmk, " ref=", E_ref, " rel_err=", rel_E);
    CHECK(rel_E < 1e-4);
}

// ============================================================================
// Tests: rectangular cell + non-uniform
// ============================================================================

TEST_CASE("nonuniform: rectangular cell with cluster") {
    // 2:1:0.5 aspect ratio
    Mat3 h = {{{6.0, 0, 0}, {0, 3.0, 0}, {0, 0, 1.5}}};
    Lattice lat(h);

    std::vector<Vec3> pos;
    std::vector<double> charges;
    double L = std::cbrt(lat.volume());
    clustered_system(lat, lat.to_cartesian({0.5, 0.5, 0.5}), 0.03 * L,
                     200, 100, 88, pos, charges);

    int N = static_cast<int>(pos.size());

    PeriodicDMKTree tree(h, pos, 6, 30, 4);
    auto pot = tree.evaluate(charges);

    REQUIRE(pot.size() == static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        CHECK(std::isfinite(pot[i]));

    auto ref = ewald_nufft(lat, pos, charges, 5.0);
    double err = l2_rel(pot, ref.potentials);
    MESSAGE("Rectangular cluster L2 error: ", err);
    CHECK(err < 1e-4);
}

// ============================================================================
// Float precision with non-uniform (when available)
// ============================================================================

#if PDMK_ENABLE_FLOAT
TEST_CASE("nonuniform: float precision with cluster") {
    double L = 5.0;
    Mat3 h = {{{L, 0, 0}, {0, L, 0}, {0, 0, L}}};
    Lattice lat(h);

    std::vector<Vec3> pos;
    std::vector<double> charges;
    clustered_system(lat, {2.5, 2.5, 2.5}, 0.03 * L, 200, 100, 42, pos, charges);

    int N = static_cast<int>(pos.size());

    PeriodicDMKTreeT<float> tree(h, pos, 6, 20, 4);
    auto pot_f = tree.evaluate(charges);

    REQUIRE(pot_f.size() == static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        CHECK(std::isfinite(pot_f[i]));

    // Compare against Ewald (double) reference
    auto ref = ewald_nufft(lat, pos, charges, 5.0);
    double num = 0, den = 0;
    for (int i = 0; i < N; ++i) {
        double d = static_cast<double>(pot_f[i]) - ref.potentials[i];
        num += d * d;
        den += ref.potentials[i] * ref.potentials[i];
    }
    double err = std::sqrt(num / den);
    MESSAGE("Float cluster L2 error: ", err);
    CHECK(err < 1e-4);
}
#endif
