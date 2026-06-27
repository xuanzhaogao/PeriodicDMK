#include <pdmk/periodic_dmk.hpp>
#include <pdmk/image_layout.hpp>

#include <dmk.h>
#include <dmk/pbc.hpp>
#include <dmk/pbc_tree.hpp>
#include <dmk/util.hpp>
#include <dmk/prolate0_fun.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <optional>
#include <stdexcept>

namespace pdmk {

template <typename Real, int DIM>
struct PeriodicDMKTreeT<Real, DIM>::Impl {
    pdmk_params params;
    std::vector<Real> avec;       // DIMxDIM column-major
    std::vector<Real> umat, qmat;
    int mgrid[DIM];
    Real xgrid0[DIM];
    Real rc;
    Real beta;

    int n_src;
    std::vector<Real> r_src;      // stable buffer for NUFFT cache pointer
    std::vector<Real> srcall;     // [DIM × nsall]
    std::vector<int>  img_src_idx;
    int nsall;

    std::unique_ptr<dmk::PBCTree<Real, DIM>> tree;
    std::vector<Real> pot_src_scratch;

    std::optional<std::array<double, DIM>> bloch_;

    Impl(const MatND<DIM>& cell, const std::vector<VecND<DIM>>& positions,
          int n_digits, int n_per_leaf, int nz,
          std::optional<std::array<double, DIM>> bloch)
        : n_src((int)positions.size()), bloch_(bloch)
    {
        if (bloch.has_value() && DIM == 2)
            throw std::logic_error("PeriodicDMKTreeT: QP (bloch) is only supported for DIM==3");

        // params
        params = pdmk_params{};
        params.n_dim = DIM;
        params.eps = std::pow(10.0, -std::max(1, n_digits));
        params.kernel = DMK_LAPLACE;
        params.pgh_src = (dmk_pgh)0;
        params.pgh_trg = DMK_POTENTIAL;
        params.use_periodic = 1;
        params.n_per_leaf = n_per_leaf;
        params.log_level = 6;

        // Lattice
        avec.resize(DIM * DIM);
        // MatND uses the Lattice convention: cell[j][i] is component i of
        // lattice vector j (column-major, see include/pdmk/lattice.hpp).
        // DMK-pbc's avec is column-major: avec[j*DIM + i] = component i of a_j.
        for (int j = 0; j < DIM; ++j)
            for (int i = 0; i < DIM; ++i)
                avec[j * DIM + i] = (Real)cell[j][i];

        umat.assign(DIM * DIM, Real(0)); qmat.assign(DIM * DIM, Real(0));
        dmk::pbc::lattice_setup<Real, DIM>(avec.data(), umat.data(), qmat.data());

        // rc selection
        if (nz > 0) {
            Real min_axis = umat[0];
            for (int d = 1; d < DIM; ++d)
                min_axis = std::min(min_axis, umat[d + d * DIM]);
            rc = min_axis / nz;
        } else {
            rc = dmk::pbc::single_level_optimal_rc<Real, DIM>(
                params, avec.data(), n_src);
        }
        beta = (Real)dmk::util::calc_bandlimiting(params);

        dmk::pbc::compute_grid<Real, DIM>(umat.data(), rc, mgrid, xgrid0);

        // Positions: stable buffer.
        //
        // lattice_setup orders avec's columns by length (longest first) and
        // builds qmat as the orthogonal rotation into that sorted frame.
        // Downstream (bin_sort, compute_grid, generate_images, far_field_nufft)
        // all consume positions as if they live in that umat-aligned frame.
        // For inputs where |a1| < |a2| the sort swaps axes and the frames
        // diverge — particles get clamped to wrong bins and the tree is
        // silently broken (`Total misplaced particles:` error at construction,
        // potentials in the 1e+17 range).
        //
        // Fix: project positions onto the umat basis via qmat^T before
        // handing them downstream. When avec is already upper-triangular
        // with the longest axis first (the common 3D campaign case),
        // qmat = I and this is a no-op. Only cells that trigger the sort
        // get an actual rotation.
        r_src.resize((size_t)DIM * n_src);
        {
            std::array<Real, DIM> tmp{};
            for (int i = 0; i < n_src; ++i) {
                for (int j = 0; j < DIM; ++j) {
                    Real s = 0;
                    for (int k = 0; k < DIM; ++k)
                        s += qmat[k + j * DIM] * (Real)positions[i][k];
                    tmp[j] = s;
                }
                for (int j = 0; j < DIM; ++j)
                    r_src[(size_t)i * DIM + j] = tmp[j];
            }
        }

        // Images + layout map
        std::vector<Real> srcimg;
        generate_images_with_src_idx<Real, DIM>(
            umat.data(), rc, mgrid, xgrid0,
            n_src, r_src.data(), srcimg, img_src_idx);
        const int nsimg = (int)img_src_idx.size();

        nsall = n_src + nsimg;
        srcall.resize((size_t)DIM * nsall);
        std::copy(r_src.begin(), r_src.end(), srcall.begin());
        std::copy(srcimg.begin(), srcimg.end(),
                   srcall.begin() + (size_t)DIM * n_src);

        // Placeholder zero charges to build the tree
        std::vector<Real> chgall_zero((size_t)nsall, Real(0));
        tree = std::make_unique<dmk::PBCTree<Real, DIM>>(
            params, avec.data(), rc,
            nsall, srcall.data(),
            n_src, r_src.data(),
            chgall_zero.data());

        pot_src_scratch.assign((size_t)n_src, Real(0));
    }

    std::vector<Real> evaluate(const std::vector<double>& charges) {
        if (bloch_.has_value())
            throw std::logic_error("evaluate: bloch is set - use evaluate_complex()");
        if ((int)charges.size() != n_src)
            throw std::invalid_argument("charges size != n_src");

        // Real charges in user order
        std::vector<Real> q_real(n_src);
        for (int i = 0; i < n_src; ++i) q_real[i] = (Real)charges[i];

        if constexpr (DIM == 2) {
            // Scope-minimized path: route 2D directly through pbcdmk_adaptive.
            // The PBCTree<Real,2>::eval() end-to-end path was never exercised
            // in the upstream DMK-pbc repo and produces divergent values on
            // neutral 2D Laplace problems. The Impl constructor's tree,
            // srcall, img_src_idx, nsall are unused on this branch — wasted
            // work but harmless; not removed to keep the diff small.
            std::vector<Real> pot_trg((size_t)n_src, Real(0));
            dmk::pbc::pbcdmk_adaptive<Real, DIM>(
                params, avec.data(), rc, n_src, r_src.data(),
                q_real.data(), pot_trg.data(), nullptr);
            return pot_trg;
        } else {
            // Combined-charge buffer
            std::vector<Real> chgall((size_t)nsall);
            std::copy(q_real.begin(), q_real.end(), chgall.begin());
            for (int j = 0; j < (int)img_src_idx.size(); ++j)
                chgall[n_src + j] = q_real[img_src_idx[j]];

            tree->update_charges(chgall.data());
            tree->eval();
            tree->apply_target_self_correction(q_real.data());

            std::vector<Real> pot_trg((size_t)n_src, Real(0));
            std::fill(pot_src_scratch.begin(), pot_src_scratch.end(), Real(0));
            tree->desort_potentials(pot_src_scratch.data(), pot_trg.data());

            // Far-field NUFFT — uses the stable r_src buffer for cache key
            dmk::pbc::far_field_nufft<Real, DIM>(
                umat.data(), rc, (Real)params.eps, beta,
                n_src, r_src.data(), q_real.data(), pot_trg.data(), Real(0),
                nullptr);

            return pot_trg;
        }
    }

    std::vector<Real> evaluate_timed(const std::vector<double>& charges,
                                      EvalTiming& timing) {
        if (bloch_.has_value())
            throw std::logic_error("evaluate_timed: bloch is set - use evaluate_complex()");
        if ((int)charges.size() != n_src)
            throw std::invalid_argument("charges size != n_src");

        std::vector<Real> q_real(n_src);
        for (int i = 0; i < n_src; ++i) q_real[i] = (Real)charges[i];

        using clk = std::chrono::steady_clock;
        auto ms_between = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        if constexpr (DIM == 2) {
            // 2D path: pbcdmk_adaptive surfaces its four internal stages
            // (build / eval / self+desort / nufft) via PbcAdaptiveTiming.
            std::vector<Real> pot_trg((size_t)n_src, Real(0));
            dmk::pbc::PbcAdaptiveTiming pat;
            dmk::pbc::pbcdmk_adaptive<Real, DIM>(
                params, avec.data(), rc, n_src, r_src.data(),
                q_real.data(), pot_trg.data(), nullptr, &pat);
            timing.build_ms  = pat.build_ms;
            timing.dmk_ms    = pat.eval_ms;
            timing.direct_ms = pat.direct_ms + pat.self_desort_ms;
            timing.nufft_ms  = pat.nufft_ms;
            return pot_trg;
        } else {
            std::vector<Real> chgall((size_t)nsall);
            std::copy(q_real.begin(), q_real.end(), chgall.begin());
            for (int j = 0; j < (int)img_src_idx.size(); ++j)
                chgall[n_src + j] = q_real[img_src_idx[j]];

            auto t0 = clk::now();
            tree->update_charges(chgall.data());
            tree->upward_pass();
            tree->downward_prologue();
            tree->form_downward_expansions();
            auto t1 = clk::now();  // end of dmk_ms

            tree->evaluate_direct_interactions();
            tree->apply_target_self_correction(q_real.data());

            std::vector<Real> pot_trg((size_t)n_src, Real(0));
            std::fill(pot_src_scratch.begin(), pot_src_scratch.end(), Real(0));
            tree->desort_potentials(pot_src_scratch.data(), pot_trg.data());
            auto t2 = clk::now();  // end of direct_ms

            dmk::pbc::far_field_nufft<Real, DIM>(
                umat.data(), rc, (Real)params.eps, beta,
                n_src, r_src.data(), q_real.data(), pot_trg.data(), Real(0),
                nullptr);
            auto t3 = clk::now();  // end of nufft_ms

            timing.dmk_ms    = ms_between(t0, t1);
            timing.direct_ms = ms_between(t1, t2);
            timing.nufft_ms  = ms_between(t2, t3);

            return pot_trg;
        }
    }

    std::vector<std::complex<Real>> evaluate_complex(const std::vector<double>& charges) {
        if (!bloch_.has_value())
            throw std::logic_error("evaluate_complex: bloch not set at construction");
        if constexpr (DIM != 3) {
            (void)charges;
            throw std::logic_error("evaluate_complex: 2D QP is out of scope");
        } else {
            if ((int)charges.size() != n_src)
                throw std::invalid_argument("charges size != n_src");
            std::vector<Real> q_real((size_t)n_src);
            for (int i = 0; i < n_src; ++i) q_real[i] = (Real)charges[i];

            // Low-level writes pot[2*n_src] interleaved (Re, Im)
            std::vector<Real> pot_flat((size_t)2 * n_src, Real(0));
            std::array<Real, DIM> bloch_real{};
            for (int d = 0; d < DIM; ++d) bloch_real[d] = (Real)(*bloch_)[d];

            dmk::pbc::pbcdmk_adaptive<Real, DIM>(
                params, avec.data(), rc, n_src, r_src.data(),
                q_real.data(), pot_flat.data(), bloch_real.data());

            std::vector<std::complex<Real>> out((size_t)n_src);
            for (int i = 0; i < n_src; ++i)
                out[i] = std::complex<Real>(pot_flat[2*i+0], pot_flat[2*i+1]);
            return out;
        }
    }
};

template <typename Real, int DIM>
PeriodicDMKTreeT<Real, DIM>::PeriodicDMKTreeT(const MatND<DIM>& cell,
                                          const std::vector<VecND<DIM>>& positions,
                                          int n_digits, int n_per_leaf, int nz,
                                          std::optional<std::array<double, DIM>> bloch)
    : p_(std::make_unique<Impl>(cell, positions, n_digits, n_per_leaf, nz, bloch)),
      n_src_(p_->n_src), rc_(p_->rc) {}

template <typename Real, int DIM>
PeriodicDMKTreeT<Real, DIM>::~PeriodicDMKTreeT() = default;

template <typename Real, int DIM>
std::vector<Real> PeriodicDMKTreeT<Real, DIM>::evaluate(const std::vector<double>& charges) {
    return p_->evaluate(charges);
}

template <typename Real, int DIM>
std::vector<Real> PeriodicDMKTreeT<Real, DIM>::evaluate_timed(
    const std::vector<double>& charges, EvalTiming& timing)
{
    return p_->evaluate_timed(charges, timing);
}

template <typename Real, int DIM>
std::vector<std::complex<Real>> PeriodicDMKTreeT<Real, DIM>::evaluate_complex(
    const std::vector<double>& charges)
{
    return p_->evaluate_complex(charges);
}

template class PeriodicDMKTreeT<double, 3>;
template class PeriodicDMKTreeT<double, 2>;
#if PDMK_ENABLE_FLOAT
template class PeriodicDMKTreeT<float, 3>;
template class PeriodicDMKTreeT<float, 2>;
#endif

// ── Auto wrapper ───────────────────────────────────────────────────────────

AutoPeriodicDMKTree::AutoPeriodicDMKTree(const Mat3& cell,
                                          const std::vector<Vec3>& positions,
                                          int n_digits,
                                          int n_per_leaf,
                                          int nz)
{
#if PDMK_ENABLE_FLOAT
    if (n_digits <= 6) {
        single_precision_ = true;
        impl_ = std::make_unique<PeriodicDMKTreeT<float, 3>>(
            cell, positions, n_digits, n_per_leaf, nz);
        return;
    }
#endif
    single_precision_ = false;
    impl_ = std::make_unique<PeriodicDMKTreeT<double, 3>>(
        cell, positions, n_digits, n_per_leaf, nz);
}

bool AutoPeriodicDMKTree::is_single_precision() const { return single_precision_; }

std::variant<std::vector<float>, std::vector<double>>
AutoPeriodicDMKTree::evaluate(const std::vector<double>& charges) {
#if PDMK_ENABLE_FLOAT
    if (single_precision_)
        return std::get<std::unique_ptr<PeriodicDMKTreeT<float, 3>>>(impl_)
                    ->evaluate(charges);
#endif
    return std::get<std::unique_ptr<PeriodicDMKTreeT<double, 3>>>(impl_)
                ->evaluate(charges);
}

AutoPeriodicDMKTree make_periodic_dmk_tree(const Mat3& cell,
                                            const std::vector<Vec3>& positions,
                                            int n_digits,
                                            int n_per_leaf,
                                            int nz) {
    return AutoPeriodicDMKTree(cell, positions, n_digits, n_per_leaf, nz);
}

// ── 2D Auto wrapper ────────────────────────────────────────────────────────

AutoPeriodicDMKTree2D::AutoPeriodicDMKTree2D(const Mat2& cell,
                                              const std::vector<Vec2>& positions,
                                              int n_digits,
                                              int n_per_leaf,
                                              int nz)
{
#if PDMK_ENABLE_FLOAT
    if (n_digits <= 6) {
        single_precision_ = true;
        impl_ = std::make_unique<PeriodicDMKTreeT<float, 2>>(
            cell, positions, n_digits, n_per_leaf, nz);
        return;
    }
#endif
    single_precision_ = false;
    impl_ = std::make_unique<PeriodicDMKTreeT<double, 2>>(
        cell, positions, n_digits, n_per_leaf, nz);
}

bool AutoPeriodicDMKTree2D::is_single_precision() const { return single_precision_; }

std::variant<std::vector<float>, std::vector<double>>
AutoPeriodicDMKTree2D::evaluate(const std::vector<double>& charges) {
#if PDMK_ENABLE_FLOAT
    if (single_precision_)
        return std::get<std::unique_ptr<PeriodicDMKTreeT<float, 2>>>(impl_)
                    ->evaluate(charges);
#endif
    return std::get<std::unique_ptr<PeriodicDMKTreeT<double, 2>>>(impl_)
                ->evaluate(charges);
}

AutoPeriodicDMKTree2D make_periodic_dmk_tree_2d(const Mat2& cell,
                                                 const std::vector<Vec2>& positions,
                                                 int n_digits,
                                                 int n_per_leaf,
                                                 int nz) {
    return AutoPeriodicDMKTree2D(cell, positions, n_digits, n_per_leaf, nz);
}

} // namespace pdmk
