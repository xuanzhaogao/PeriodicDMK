#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <pdmk/cell_list.hpp>
#include <pdmk/types.hpp>

#include <cmath>
#include <random>
#include <set>
#include <utility>
#include <vector>

using namespace pdmk;

// Brute-force unique pair count with image shells, used as reference.
static int brute_pair_count(const Mat2 &cell,
                            const std::vector<Vec2> &pos,
                            double cutoff) {
    double a0x = cell[0][0], a0y = cell[0][1];
    double a1x = cell[1][0], a1y = cell[1][1];
    double det = std::abs(a0x * a1y - a0y * a1x);
    auto norm2 = [](double x, double y) { return std::sqrt(x*x + y*y); };
    double h0 = det / norm2(a1x, a1y);
    double h1 = det / norm2(a0x, a0y);
    int n0 = (int)std::ceil(cutoff / h0) + 1;
    int n1 = (int)std::ceil(cutoff / h1) + 1;
    double r2c = cutoff * cutoff;
    int N = (int)pos.size();
    int count = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            for (int k0 = -n0; k0 <= n0; ++k0)
            for (int k1 = -n1; k1 <= n1; ++k1) {
                if (i == j && k0 == 0 && k1 == 0) continue;
                double dx = pos[j][0] + k0*a0x + k1*a1x - pos[i][0];
                double dy = pos[j][1] + k0*a0y + k1*a1y - pos[i][1];
                double r2 = dx*dx + dy*dy;
                if (r2 < r2c && r2 > 1e-30) {
                    if (i == j) count += 1;  // self-image pair: one visit
                    else count += 1;
                }
            }
        }
    }
    return count;
}

TEST_CASE("CellList2D pair count matches brute force (unit square)") {
    Mat2 cell{{{1.0, 0.0}, {0.0, 1.0}}};
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int N = 100;
    std::vector<Vec2> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = {U(rng), U(rng)};
    double cutoff = 0.1;

    CellList2D cl(cell, cutoff, pos);
    int cl_count = 0;
    cl.for_each_pair([&](int, int, const Vec2 &, double) { cl_count++; });

    int brute = brute_pair_count(cell, pos, cutoff);
    CHECK(cl_count == brute);
}

TEST_CASE("CellList2D dr is consistent with r2 and cutoff") {
    Mat2 cell{{{1.0, 0.0}, {0.0, 1.0}}};
    std::mt19937_64 rng(13);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int N = 50;
    std::vector<Vec2> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = {U(rng), U(rng)};
    double cutoff = 0.2;

    CellList2D cl(cell, cutoff, pos);
    for (int i = 0; i < N; ++i) {
        cl.for_each_neighbor(i, [&](int /*j*/, const Vec2 &dr, double r2) {
            double r2_check = dr[0]*dr[0] + dr[1]*dr[1];
            CHECK(std::abs(r2_check - r2) < 1e-14);
            CHECK(r2 <= cutoff * cutoff);
        });
    }
}

TEST_CASE("CellList2D triclinic shear cell") {
    Mat2 cell{{{1.0, 0.0}, {0.3, 0.95}}};
    std::mt19937_64 rng(19);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int N = 120;
    std::vector<Vec2> pos(N);
    for (int i = 0; i < N; ++i) {
        double s0 = U(rng), s1 = U(rng);
        pos[i] = {s0 * cell[0][0] + s1 * cell[1][0],
                  s0 * cell[0][1] + s1 * cell[1][1]};
    }
    double cutoff = 0.15;

    CellList2D cl(cell, cutoff, pos);
    int cl_count = 0;
    cl.for_each_pair([&](int, int, const Vec2 &, double) { cl_count++; });
    CHECK(cl_count == brute_pair_count(cell, pos, cutoff));
}

TEST_CASE("CellList2D image-shell fallback (cutoff > cell height)") {
    Mat2 cell{{{1.0, 0.0}, {0.0, 0.2}}};
    std::mt19937_64 rng(23);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int N = 40;
    std::vector<Vec2> pos(N);
    for (int i = 0; i < N; ++i) {
        pos[i] = {U(rng), 0.2 * U(rng)};
    }
    double cutoff = 0.4;

    CellList2D cl(cell, cutoff, pos);
    int cl_count = 0;
    cl.for_each_pair([&](int, int, const Vec2 &, double) { cl_count++; });
    CHECK(cl_count == brute_pair_count(cell, pos, cutoff));
}

TEST_CASE("CellList2D high-aspect cell (no off-by-one in nbins)") {
    Mat2 cell{{{1.0, 0.0}, {0.0, 100.0}}};
    std::mt19937_64 rng(29);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int N = 1000;
    std::vector<Vec2> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = {U(rng), 100.0 * U(rng)};
    double cutoff = 0.1;

    CellList2D cl(cell, cutoff, pos);
    int cl_count = 0;
    cl.for_each_pair([&](int, int, const Vec2 &, double) { cl_count++; });
    CHECK(cl_count == brute_pair_count(cell, pos, cutoff));
}
