// Regression test: PeriodicDMKTreeT<double, 2> must produce accurate
// potentials even when the input cell's lattice_setup reorders axes.
//
// Background: lattice_setup sorts lattice vectors by length (longest
// first) and Gram-Schmidts them into an upper-triangular umat with a
// rotation matrix qmat. Downstream code (bin_sort, compute_grid,
// generate_images) consumes positions as if they live in the umat
// frame. Before the qmat-rotation fix, any cell with |a1| < |a2|
// triggered the sort — misaligning the position frame from umat and
// silently breaking the tree (`Total misplaced particles:` error at
// construction, rel_l2 in the 1e+17 range).
//
// PDMK 2D internally uses the +log(r) kernel convention; ewald_direct_2d
// uses the physical -log(r)/(2π) convention. This test applies the
// documented -1/(2π) rescale and mean-subtracts (both are defined only
// up to an additive gauge in neutral 2D PBC).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <pdmk/ewald_nufft.hpp>
#include <pdmk/periodic_dmk.hpp>
#include <pdmk/types.hpp>

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

using namespace pdmk;

static constexpr double DMK_2D_TO_PHYSICAL = -1.0 / (2.0 * M_PI);

static std::vector<double> mean_sub(std::vector<double> v) {
    double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    for (double& x : v) x -= m;
    return v;
}

static double rel_l2(const std::vector<double>& a,
                     const std::vector<double>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return std::sqrt(num / den);
}

static void check_against_ewald(Mat2 cell, int N, uint64_t seed,
                                double tol) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.02, 0.98);
    std::vector<Vec2> pos(N);
    std::vector<double> chg(N);
    for (int i = 0; i < N; ++i) {
        double s = U(rng), t = U(rng);
        pos[i] = {s * cell[0][0] + t * cell[1][0],
                  s * cell[0][1] + t * cell[1][1]};
        chg[i] = (i & 1) ? +1.0 : -1.0;
    }

    auto ref = mean_sub(ewald_direct_2d(cell, pos, chg, 5.5).potentials);

    PeriodicDMKTreeT<double, 2> tree(cell, pos, 6, 150, 7);
    (void)tree.evaluate(chg);
    auto raw = tree.evaluate(chg);
    std::vector<double> u_phys(raw.size());
    for (size_t i = 0; i < raw.size(); ++i)
        u_phys[i] = DMK_2D_TO_PHYSICAL * raw[i];
    u_phys = mean_sub(std::move(u_phys));

    CHECK(rel_l2(u_phys, ref) < tol);
}

TEST_CASE("2D PDMK agrees with Ewald when |a2| > |a1| (2:1 rect)") {
    // a1 = (1, 0), a2 = (0, 2). |a2|>|a1| triggers the lattice_setup sort.
    Mat2 cell{{{1.0, 0.0}, {0.0, 2.0}}};
    check_against_ewald(cell, /*N=*/500, /*seed=*/31, /*tol=*/1e-4);
}

TEST_CASE("2D PDMK agrees with Ewald when |a2| > |a1| (tri α=π/3, aspect=2)") {
    const double c = 0.5, s = 0.8660254037844387;
    Mat2 cell{{{1.0, 0.0}, {2.0 * c, 2.0 * s}}};
    check_against_ewald(cell, /*N=*/500, /*seed=*/41, /*tol=*/1e-4);
}
