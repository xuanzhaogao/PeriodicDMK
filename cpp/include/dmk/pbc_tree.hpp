#ifndef PBC_TREE_HPP
#define PBC_TREE_HPP

#include <complex>
#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/logger.h>
#include <dmk/pbc.hpp>
#include <dmk/tree.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>
#include <nda/nda.hpp>
#include <sctl.hpp>

namespace dmk {

/// Multilevel tree for PBC-DMK.
///
/// Rectangular grid of rc-sized boxes at level 0, standard octree below.
/// No SCTL dependency for tree topology — uses top-down construction.
/// Eval code adapted from DMKPtTree with 3 level-0 fixups.
///
/// Single-rank only (no MPI): owned = with_halo, no ghost boxes.
template <typename Real, int DIM>
struct PBCTree {
    // ================================================================
    // Tree topology (replaces SCTL Morton IDs / NodeLists / NodeAttr)
    // ================================================================
    static constexpr int mc = 1 << DIM;            // children per box
    static constexpr int ncoll_max = sctl::pow<DIM>(3); // max colleagues per box

    std::vector<int> box_depth;                        // depth of each box
    std::vector<int> box_parent;                       // parent index (-1 at level 0)
    std::vector<int> box_nchild;                       // number of children (0 = leaf)
    std::vector<std::array<int, mc>> box_children;     // child indices (-1 = no child)
    std::vector<int> box_ncoll;                        // number of colleagues
    std::vector<std::array<int, ncoll_max>> box_colleagues; // colleague indices (-1 = invalid)

    int nboxes_ = 0;

    // ================================================================
    // Level structure (same names as DMKPtTree)
    // ================================================================
    sctl::Vector<sctl::Vector<int>> level_indices;
    sctl::Vector<Real> boxsize;
    sctl::Vector<Real> centers;

    // ================================================================
    // Particle counts and offsets per box
    // Single-rank: owned = with_halo (no ghost/halo distinction)
    // ================================================================
    sctl::Vector<int> src_counts_owned;
    sctl::Vector<int> trg_counts_owned;
    // Aliases for eval code compatibility
    sctl::Vector<int> &src_counts_with_halo = src_counts_owned;

    // Sorted particle positions and charges
    sctl::Vector<Real> r_src_sorted_owned;
    sctl::Vector<sctl::Long> r_src_offsets_owned;
    // Alias
    sctl::Vector<Real> &r_src_sorted_with_halo = r_src_sorted_owned;
    sctl::Vector<sctl::Long> r_src_offsets_with_halo;

    sctl::Vector<Real> r_trg_sorted_owned;
    sctl::Vector<sctl::Long> r_trg_offsets_owned;

    sctl::Vector<Real> charge_sorted_owned;
    sctl::Vector<sctl::Long> charge_offsets_owned;
    // Alias
    sctl::Vector<Real> &charge_sorted_with_halo = charge_sorted_owned;
    sctl::Vector<sctl::Long> charge_offsets_with_halo;

    // Output potentials (sorted order)
    sctl::Vector<Real> pot_src_sorted;
    sctl::Vector<sctl::Long> pot_src_offsets;
    sctl::Vector<Real> pot_trg_sorted;
    sctl::Vector<sctl::Long> pot_trg_offsets;

    // ================================================================
    // Proxy coefficients and plane wave data
    // ================================================================
    sctl::Vector<Real> proxy_coeffs_upward;
    sctl::Vector<sctl::Long> proxy_coeffs_offsets;
    sctl::Vector<Real> proxy_coeffs_downward;
    sctl::Vector<sctl::Long> proxy_coeffs_offsets_downward;

    sctl::Vector<std::complex<Real>> pw_out;
    sctl::Vector<sctl::Long> pw_out_offsets;

    // ================================================================
    // Flags
    // ================================================================
    sctl::Vector<bool> ifpwexp;
    sctl::Vector<bool> iftensprodeval;
    sctl::Vector<bool> is_global_leaf; // = is_leaf for single-rank

    sctl::Vector<bool> has_proxy_from_children;
    struct C2PWork {
        int src_box;
        int center_box;
        int level;
    };
    std::vector<C2PWork> charge2proxy_work;
    std::vector<int> direct_work;

    // ================================================================
    // Interaction lists
    // ================================================================
    static constexpr int nlist1_max_ = sctl::pow<DIM>(4) - sctl::pow<DIM>(2) + 1;
    std::vector<std::array<int, nlist1_max_>> list1_;
    std::vector<int> nlist1_;

    static constexpr int nlistpw_max_ = sctl::pow<DIM>(3);
    std::vector<std::array<int, nlistpw_max_>> listpw_;
    std::vector<int> nlistpw_;

    std::vector<int> proxy_down_zeroed;

    // ================================================================
    // Parameters (must be declared before expansion_constants)
    // ================================================================
    const pdmk_params params;
    // Non-const so callers can force an nd-promoted layout (e.g. nd=2 for
    // quasi-periodic Laplace, where each original density becomes a (Re, Im)
    // pair). When the caller does not override, these default to the values
    // derived from params.kernel + pgh.
    int kernel_input_dim;
    int kernel_output_dim_src;
    int kernel_output_dim_trg;
    int kernel_output_dim_max;
    int n_tables;
    const int n_digits;

    // ================================================================
    // Fourier / kernel data
    // ================================================================
    ExpansionConstants<DIM> expansion_constants;
    const int &n_order = expansion_constants.n_order;

    struct LevelFourierData {
        sctl::Vector<std::complex<Real>> poly2pw;
        sctl::Vector<std::complex<Real>> pw2poly;
        sctl::Vector<Real> radialft;
        sctl::Vector<std::complex<Real>> wpwshift;
    };
    std::vector<LevelFourierData> difference_fourier_data;
    LevelFourierData window_fourier_data;

    FourierData<Real> fourier_data;
    sctl::Vector<Real> c2p;
    sctl::Vector<Real> p2c;

    // ================================================================
    // Evaluators
    // ================================================================
    std::vector<direct_evaluator_func<Real>> evaluator_by_level_src;
    std::vector<direct_evaluator_func<Real>> evaluator_by_level_trg;

    sctl::Vector<sctl::Vector<Real>> workspaces_;

    // ================================================================
    // Sort permutations (for desort)
    // ================================================================
    std::vector<int> src_perm; // src_perm[sorted_idx] = original_idx
    std::vector<int> trg_perm;

    // Number of original sources / targets (before images)
    int n_orig_src = 0;
    int n_orig_trg = 0;

    // Logging
    std::shared_ptr<spdlog::logger> logger;

    // Fine-grained timing accumulators for DMK_PBC_PROFILE. Reset by downward_pass.
    mutable double form_eval_t_pw_ = 0;   // PW-shift (colleague → pw_in)
    mutable double form_eval_t_p2p_ = 0;  // pw → proxy_down
    mutable double form_eval_t_p2c_ = 0;  // parent → child proxy propagation
    mutable double form_eval_t_eval_ = 0; // proxy → target potentials
    mutable long   form_eval_n_shifts_ = 0;

    // ================================================================
    // Constructor
    //
    // Caller supplies combined original + image sources and charges.
    // (Image generation lives in the caller, matching the Fortran
    // pbcdmk_adaptive pattern.) For non-QP, use the pbc::generate_images
    // helper with bloch=nullptr; for QP use it with a non-null bloch and
    // pass nd_override=2.
    //
    // Parameters:
    //   n_src_all  — total sources (originals + images)
    //   r_src_all  — sources[DIM × n_src_all]
    //   charge_all — charges[nd × n_src_all] where nd = nd_override (if
    //                != -1) or get_kernel_input_dim(params)
    //   nd_override — if != -1, override kernel_input_dim (and scale
    //                 kernel_output_dim_* by the same factor). Valid for
    //                 scalar real kernels (e.g. Laplace + DMK_POTENTIAL).
    // ================================================================
    PBCTree(const pdmk_params &params_, const Real *avec, Real rc,
            int n_src_all, const Real *r_src_all, int n_trg, const Real *r_trg,
            const Real *charge_all, int nd_override = -1);

    // ================================================================
    // Accessors (same signatures as DMKPtTree)
    // ================================================================
    int n_levels() const { return level_indices.Dim(); }
    int n_boxes() const { return nboxes_; }

    std::span<const int> list1(int i) const { return {list1_[i].data(), (size_t)nlist1_[i]}; }
    std::span<const int> listpw(int i) const { return {listpw_[i].data(), (size_t)nlistpw_[i]}; }

    Real *r_src_with_halo_ptr(int i) { return &r_src_sorted_owned[r_src_offsets_with_halo[i]]; }
    Real *r_src_owned_ptr(int i) { return &r_src_sorted_owned[r_src_offsets_owned[i]]; }
    Real *r_trg_owned_ptr(int i) { return trg_counts_owned[i] ? &r_trg_sorted_owned[r_trg_offsets_owned[i]] : nullptr; }

    Real *pot_src_ptr(int i) { return &pot_src_sorted[pot_src_offsets[i]]; }
    Real *pot_trg_ptr(int i) { return &pot_trg_sorted[pot_trg_offsets[i]]; }

    Real *charge_owned_ptr(int i) { return &charge_sorted_owned[charge_offsets_owned[i]]; }
    Real *charge_with_halo_ptr(int i) { return &charge_sorted_owned[charge_offsets_with_halo[i]]; }

    Real *center_ptr(int i) { return &centers[i * DIM]; }
    const Real *center_ptr(int i) const { return &centers[i * DIM]; }
    ndview<Real, 1> center_view(int i) { return ndview<Real, 1>({DIM}, center_ptr(i)); }
    ndview<const Real, 1> center_view(int i) const { return ndview<const Real, 1>({DIM}, center_ptr(i)); }

    Real *proxy_ptr_upward(int i) { return &proxy_coeffs_upward[proxy_coeffs_offsets[i]]; }
    Real *proxy_ptr_downward(int i) { return &proxy_coeffs_downward[proxy_coeffs_offsets_downward[i]]; }

    ndview<Real, 2> r_src_with_halo_view(int i) { return {DIM == 2 ? std::array<long,2>{DIM, src_counts_owned[i]} : std::array<long,2>{DIM, src_counts_owned[i]}, r_src_with_halo_ptr(i)}; }
    ndview<Real, 2> r_src_owned_view(int i) { return {{DIM, src_counts_owned[i]}, r_src_owned_ptr(i)}; }
    ndview<Real, 2> r_trg_owned_view(int i) { return {{DIM, trg_counts_owned[i]}, r_trg_owned_ptr(i)}; }
    ndview<Real, 2> pot_src_view(int i) { return {{kernel_output_dim_src, src_counts_owned[i]}, pot_src_ptr(i)}; }
    ndview<Real, 2> pot_trg_view(int i) { return {{kernel_output_dim_trg, trg_counts_owned[i]}, pot_trg_ptr(i)}; }
    ndview<Real, 2> charge_owned_view(int i) { return {{kernel_input_dim, src_counts_owned[i]}, charge_owned_ptr(i)}; }
    ndview<Real, 2> charge_with_halo_view(int i) { return {{kernel_input_dim, src_counts_owned[i]}, charge_with_halo_ptr(i)}; }

    ndview<Real, DIM + 1> proxy_view_upward(int i);
    ndview<Real, DIM + 1> proxy_view_downward(int i);

    std::complex<Real> *pw_out_ptr(int i) { return &pw_out[pw_out_offsets[i]]; }
    ndview<std::complex<Real>, DIM + 1> pw_out_view(int i);

    // ================================================================
    // Evaluation
    // ================================================================
    void eval();
    void upward_pass();
    void downward_pass();
    /// Reset pot_*_sorted and planewave data before downward-pass work.
    /// Must run before form_downward_expansions or evaluate_direct_interactions.
    void downward_prologue();
    /// Planewave upward→downward evaluation (form_outgoing + per-level form_eval).
    /// Requires downward_prologue() to have run. Legacy name: the planewave half
    /// of downward_pass().
    void form_downward_expansions();
    void form_outgoing_expansions();
    void form_eval_expansions(const sctl::Vector<int> &boxes,
                              const sctl::Vector<std::complex<Real>> &wpwshift,
                              Real boxsize,
                              const ndview<std::complex<Real>, 2> &pw2poly_view,
                              const sctl::Vector<Real> &p2c);
    void evaluate_direct_interactions();
    void init_planewave_data();

    /// Reusable-evaluation hook: refill sorted charge buffer from a
    /// user-order combined-charge array (n_src originals followed by
    /// image slots, same order as build_combined_sources), then zero
    /// pot_src_sorted / pot_trg_sorted so the next eval() starts fresh.
    /// Positions and the tree topology are unchanged.
    void update_charges(const Real* charge_all_unsorted);

    /// Extract potentials from sorted to user order.
    void desort_potentials(Real *pot_src, Real *pot_trg);

    /// Apply self-interaction correction to target potentials.
    /// Use when targets coincide with sources (pbcdmk_adaptive): subtracts
    /// W(0, level) * trg_charge[i] per target, mirroring Fortran
    /// pbcdmk_self_corr_targ. trg_charges is in user order (same as r_trg).
    void apply_target_self_correction(const Real *trg_charges);

private:
    // ================================================================
    // Construction helpers
    // ================================================================
    void build_tree(const Real *avec, Real rc,
                    int n_src_all, const Real *r_src_all,
                    int n_trg, const Real *r_trg,
                    const Real *charge_all);
    void refine_level(int level, int ndiv);
    void build_colleague_lists_deeper(int level);
    void compute_data_offsets();
    void compute_proxy_expansion_flags();
    void compute_proxy_evaluation_flags();
    void build_plane_wave_interaction_lists();
    void build_direct_interaction_lists();
    void build_upward_pass_work_lists();
    void build_direct_work_lists();
    void allocate_proxy_coefficients();
    void precompute_window_difference_data();
    void build_evaluators();
    void generate_metadata();
};

} // namespace dmk

#endif
