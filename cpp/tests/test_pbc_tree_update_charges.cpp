#include <vector>
#include <random>
#include <cmath>
#include <dmk.h>
#include <dmk/pbc.hpp>
#include <dmk/pbc_tree.hpp>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("PBCTree::update_charges reuses tree with new charges") {
    using Real = double;
    const int DIM = 3;
    const int N = 64;

    std::vector<Real> avec = {2, 0, 0, 0, 2, 0, 0, 0, 2};
    std::mt19937 rng(42);
    std::uniform_real_distribution<Real> U(0.05, 1.95);
    std::vector<Real> r_src(DIM * N);
    for (auto& x : r_src) x = U(rng);

    std::vector<Real> q1(N), q2(N);
    std::normal_distribution<Real> G(0, 1);
    for (int i = 0; i < N; ++i) { q1[i] = G(rng); q2[i] = G(rng); }

    pdmk_params p;
    p.n_dim = 3; p.eps = 1e-3; p.kernel = DMK_LAPLACE;
    p.pgh_src = (dmk_pgh)0; p.pgh_trg = DMK_POTENTIAL;
    p.use_periodic = 1; p.n_per_leaf = 16; p.log_level = 6;

    const Real rc = dmk::pbc::single_level_optimal_rc<Real, DIM>(p, avec.data(), N);

    // Build combined sources for charge set 1
    std::vector<Real> srcall1, chgall1;
    int nsall = dmk::pbc::build_combined_sources<Real, DIM>(
        avec.data(), rc, N, r_src.data(), 1, q1.data(),
        srcall1, chgall1);

    dmk::PBCTree<Real, DIM> tree(p, avec.data(), rc,
                                  nsall, srcall1.data(),
                                  N, r_src.data(), chgall1.data());
    tree.eval();
    tree.apply_target_self_correction(q1.data());
    std::vector<Real> pot1_src(N, 0), pot1_trg(N, 0);
    tree.desort_potentials(pot1_src.data(), pot1_trg.data());

    // Now rebuild combined charges for q2 (same positions → same images),
    // call update_charges, re-eval, and compare against a fresh tree.
    std::vector<Real> srcall2_dummy, chgall2_full;
    dmk::pbc::build_combined_sources<Real, DIM>(
        avec.data(), rc, N, r_src.data(), 1, q2.data(),
        srcall2_dummy, chgall2_full);

    tree.update_charges(chgall2_full.data());
    tree.eval();
    tree.apply_target_self_correction(q2.data());
    std::vector<Real> pot2a_src(N, 0), pot2a_trg(N, 0);
    tree.desort_potentials(pot2a_src.data(), pot2a_trg.data());

    // Reference: build fresh tree for q2
    dmk::PBCTree<Real, DIM> tree2(p, avec.data(), rc,
                                   nsall, srcall2_dummy.data(),
                                   N, r_src.data(), chgall2_full.data());
    tree2.eval();
    tree2.apply_target_self_correction(q2.data());
    std::vector<Real> pot2b_src(N, 0), pot2b_trg(N, 0);
    tree2.desort_potentials(pot2b_src.data(), pot2b_trg.data());

    Real num = 0, den = 0;
    for (int i = 0; i < N; ++i) {
        Real d = pot2a_trg[i] - pot2b_trg[i];
        num += d * d;
        den += pot2b_trg[i] * pot2b_trg[i];
    }
    CHECK(std::sqrt(num / std::max(den, Real(1e-300))) < 1e-12);
}
