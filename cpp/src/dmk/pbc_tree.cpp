#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <numeric>

#include <dmk/legeexps.hpp>
#include <dmk/pbc_tree.hpp>
#include <dmk/chebychev.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/omp_wrapper.hpp>
#include <dmk/planewave.hpp>
#include <dmk/prolate0_fun.hpp>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>

namespace dmk {

// ================================================================
// View accessors (template-dependent)
// ================================================================

template <typename Real, int DIM>
ndview<Real, DIM + 1> PBCTree<Real, DIM>::proxy_view_upward(int i) {
    if constexpr (DIM == 2)
        return ndview<Real, DIM + 1>({n_order, n_order, n_tables}, proxy_ptr_upward(i));
    else
        return ndview<Real, DIM + 1>({n_order, n_order, n_order, n_tables}, proxy_ptr_upward(i));
}

template <typename Real, int DIM>
ndview<Real, DIM + 1> PBCTree<Real, DIM>::proxy_view_downward(int i) {
    if constexpr (DIM == 2)
        return ndview<Real, DIM + 1>({n_order, n_order, n_tables}, proxy_ptr_downward(i));
    else
        return ndview<Real, DIM + 1>({n_order, n_order, n_order, n_tables}, proxy_ptr_downward(i));
}

template <typename Real, int DIM>
ndview<std::complex<Real>, DIM + 1> PBCTree<Real, DIM>::pw_out_view(int i) {
    const int n_pw = expansion_constants.n_pw_diff;
    if constexpr (DIM == 2)
        return ndview<std::complex<Real>, DIM + 1>({n_pw, (n_pw + 1) / 2, n_tables}, pw_out_ptr(i));
    else
        return ndview<std::complex<Real>, DIM + 1>({n_pw, n_pw, (n_pw + 1) / 2, n_tables}, pw_out_ptr(i));
}

// ================================================================
// Helper: child sign pattern. Child j has center = parent_center + isgn[j][d]*bsh.
// Convention: bit d of j → sign in dimension d. 0 = minus, 1 = plus.
// ================================================================
static constexpr int child_sign(int child_idx, int dim) {
    return ((child_idx >> dim) & 1) ? 1 : -1;
}

// ================================================================
// Sort particles within a parent box's range into its children.
// Updates isrc (permutation), isrcse (ranges), and the sorted position array.
// ================================================================
template <typename Real, int DIM>
static void sort_to_children(int parent, int nboxes,
                             const Real *centers_arr, // DIM * nboxes
                             const int *box_children_ptr, // mc entries for parent
                             int n_pts, // total pts in parent's range
                             int parent_start, // start index in sorted arrays
                             Real *pos_sorted, // DIM * n_total_pts (sorted positions)
                             Real *chg_sorted, // nd * n_total_pts (sorted charges, or nullptr)
                             int nd,
                             int *isrc, // permutation array
                             int *isrcse) // 2 * nboxes (start, end per box)
{
    constexpr int mc = 1 << DIM;
    if (n_pts <= 0) return;

    // Determine which child each particle belongs to
    const Real *pcen = &centers_arr[parent * DIM];
    std::vector<int> child_of(n_pts);
    for (int i = 0; i < n_pts; i++) {
        int ci = 0;
        for (int d = 0; d < DIM; d++) {
            if (pos_sorted[(parent_start + i) * DIM + d] >= pcen[d])
                ci |= (1 << d);
        }
        child_of[i] = ci;
    }

    // Count per child
    int count[mc] = {};
    for (int i = 0; i < n_pts; i++)
        count[child_of[i]]++;

    // Compute child start offsets
    int offset[mc];
    offset[0] = parent_start;
    for (int c = 1; c < mc; c++)
        offset[c] = offset[c - 1] + count[c - 1];

    // Set isrcse for children
    for (int c = 0; c < mc; c++) {
        int cbox = box_children_ptr[c];
        if (cbox >= 0) {
            isrcse[2 * cbox] = offset[c];
            isrcse[2 * cbox + 1] = offset[c] + count[c] - 1;
        }
    }

    // Rearrange: create temp copies and write back in child order
    std::vector<Real> tmp_pos(n_pts * DIM);
    std::vector<Real> tmp_chg(nd > 0 ? n_pts * nd : 0);
    std::vector<int> tmp_perm(n_pts);
    std::memcpy(tmp_pos.data(), &pos_sorted[parent_start * DIM], n_pts * DIM * sizeof(Real));
    if (nd > 0 && chg_sorted)
        std::memcpy(tmp_chg.data(), &chg_sorted[parent_start * nd], n_pts * nd * sizeof(Real));
    std::memcpy(tmp_perm.data(), &isrc[parent_start], n_pts * sizeof(int));

    int running[mc];
    std::memcpy(running, offset, sizeof(offset));

    for (int i = 0; i < n_pts; i++) {
        int c = child_of[i];
        int dst = running[c] - parent_start;
        // Actually, we need to write to the global array at position running[c]
        int gidx = running[c];
        for (int d = 0; d < DIM; d++)
            pos_sorted[gidx * DIM + d] = tmp_pos[i * DIM + d];
        if (nd > 0 && chg_sorted)
            for (int k = 0; k < nd; k++)
                chg_sorted[gidx * nd + k] = tmp_chg[i * nd + k];
        isrc[gidx] = tmp_perm[i];
        running[c]++;
    }
}

// ================================================================
// Constructor: build PBC tree
// ================================================================

template <typename Real, int DIM>
PBCTree<Real, DIM>::PBCTree(const pdmk_params &params_, const Real *avec, Real rc,
                            int n_src_all, const Real *r_src_all, int n_trg, const Real *r_trg,
                            const Real *charge_all, int nd_override)
    : params(params_),
      kernel_input_dim(get_kernel_input_dim(params.n_dim, params.kernel)),
      kernel_output_dim_src(get_kernel_output_dim(params.n_dim, params.kernel, params.pgh_src)),
      kernel_output_dim_trg(get_kernel_output_dim(params.n_dim, params.kernel, params.pgh_trg)),
      kernel_output_dim_max(std::max(kernel_output_dim_src, kernel_output_dim_trg)),
      n_tables(1),
      n_digits(std::max(1, (int)std::round(log10(1.0 / params_.eps) - 0.1))),
      expansion_constants(params),
      logger(spdlog::default_logger()) {

    // Apply nd_override if requested (e.g. nd=2 for quasi-periodic Laplace).
    // The scalar factor is applied uniformly to input and output dims — only
    // valid for scalar real kernels.
    if (nd_override > 0) {
        const int nd_default = get_kernel_input_dim(params.n_dim, params.kernel);
        if (nd_default != 1)
            throw std::runtime_error("PBCTree: nd_override only supported for scalar kernels");
        const int f = nd_override;
        kernel_input_dim      = f;
        kernel_output_dim_src = kernel_output_dim_src * f;
        kernel_output_dim_trg = kernel_output_dim_trg * f;
        kernel_output_dim_max = std::max(kernel_output_dim_src, kernel_output_dim_trg);
    }
    // n_tables = nd for proxy / PW storage (generate_metadata reads this)
    n_tables = kernel_input_dim;

    n_orig_src = n_src_all;
    n_orig_trg = n_trg;
    const int nd = kernel_input_dim;

    // ============================================
    // Step 1: Lattice setup + grid
    // ============================================
    Real umat[DIM * DIM], qmat[DIM * DIM];
    pbc::lattice_setup<Real, DIM>(avec, umat, qmat);

    int mgrid[DIM];
    Real xgrid0[DIM];
    pbc::compute_grid<Real, DIM>(umat, rc, mgrid, xgrid0);

    // ============================================
    // Step 2: Use caller-supplied combined sources/charges
    // (Image generation has been lifted out of PBCTree; matches Fortran
    // pbcdmk_adaptive pattern. Caller uses pbc::generate_images with an
    // optional Bloch vector to phase image charges for QP.)
    // ============================================
    const int nsall = n_src_all;
    // Non-owning views; we reuse the caller's buffers directly where
    // possible, but the downstream code below reorders (bin sort) into new
    // arrays, so a copy simplifies lifetime management.
    std::vector<Real> srcall(DIM * nsall), chgall(nd * nsall);
    std::memcpy(srcall.data(), r_src_all, (size_t)DIM * nsall * sizeof(Real));
    std::memcpy(chgall.data(), charge_all, (size_t)nd * nsall * sizeof(Real));

    logger->info("PBCTree: nsall={}", nsall);

    // ============================================
    // Step 4: Level-0 rectangular grid
    // ============================================
    int mle[DIM];
    Real xelo[DIM];
    for (int i = 0; i < DIM; i++) {
        mle[i] = mgrid[i] + 2;
        xelo[i] = xgrid0[i] - rc;
    }

    int nboxes0 = mle[0] * mle[1];
    if constexpr (DIM == 3) nboxes0 *= mle[2];

    // Reserve generous capacity for boxes (level 0 + refinement)
    const int max_boxes = nboxes0 + mc * ((nsall + params.n_per_leaf - 1) / params.n_per_leaf) * 50 + 100;
    box_depth.reserve(max_boxes);
    box_parent.reserve(max_boxes);
    box_nchild.reserve(max_boxes);
    box_children.reserve(max_boxes);
    box_ncoll.reserve(max_boxes);
    box_colleagues.reserve(max_boxes);

    // Temporary centers for grid setup
    std::vector<Real> grid_centers(DIM * nboxes0);
    std::vector<int> grid_colleagues(ncoll_max * nboxes0);
    pbc::rect_grid_setup<Real, DIM>(mgrid, rc, xgrid0, grid_centers.data(), grid_colleagues.data());

    // Initialize level-0 boxes
    box_depth.resize(nboxes0);
    box_parent.resize(nboxes0);
    box_nchild.resize(nboxes0, 0);
    box_children.resize(nboxes0);
    box_ncoll.resize(nboxes0);
    box_colleagues.resize(nboxes0);

    for (int i = 0; i < nboxes0; i++) {
        box_depth[i] = 0;
        box_parent[i] = -1;
        box_children[i].fill(-1);

        // Count valid colleagues
        box_ncoll[i] = 0;
        for (int j = 0; j < ncoll_max; j++) {
            box_colleagues[i][j] = grid_colleagues[i * ncoll_max + j];
            if (box_colleagues[i][j] >= 0) box_ncoll[i]++;
        }
    }
    nboxes_ = nboxes0;

    // ============================================
    // Step 5: Bin sort sources into level-0 grid
    // ============================================
    src_perm.resize(nsall);
    std::vector<int> src_se(2 * max_boxes, 0); // start/end per box
    for (int i = 0; i < 2 * max_boxes; i++) src_se[i] = (i % 2 == 0) ? 0 : -1; // empty

    pbc::bin_sort<Real, DIM>(mle, rc, xelo, nboxes0, nsall, srcall.data(),
                              src_perm.data(), src_se.data());

    // Reorder sources into sorted order
    std::vector<Real> src_sorted(DIM * nsall), chg_sorted(nd * nsall);
    for (int i = 0; i < nsall; i++) {
        int idx = src_perm[i];
        for (int d = 0; d < DIM; d++)
            src_sorted[i * DIM + d] = srcall[idx * DIM + d];
        for (int k = 0; k < nd; k++)
            chg_sorted[i * nd + k] = chgall[idx * nd + k];
    }

    // Sort targets similarly (if separate from sources)
    trg_perm.resize(n_trg);
    std::vector<int> trg_se(2 * max_boxes, 0);
    for (int i = 0; i < 2 * max_boxes; i++) trg_se[i] = (i % 2 == 0) ? 0 : -1;

    if (n_trg > 0) {
        pbc::bin_sort<Real, DIM>(mle, rc, xelo, nboxes0, n_trg, r_trg,
                                  trg_perm.data(), trg_se.data());
    }
    std::vector<Real> trg_sorted(DIM * n_trg);
    for (int i = 0; i < n_trg; i++) {
        int idx = trg_perm[i];
        for (int d = 0; d < DIM; d++)
            trg_sorted[i * DIM + d] = r_trg[idx * DIM + d];
    }

    // ============================================
    // Step 6: Top-down octree refinement
    // ============================================
    // Temporary centers array (grows with new boxes)
    std::vector<Real> all_centers(DIM * max_boxes);
    std::memcpy(all_centers.data(), grid_centers.data(), DIM * nboxes0 * sizeof(Real));

    const int ndiv = params.n_per_leaf;
    // Hysteresis factor on the split decision: don't split a box unless its
    // particle count exceeds npl_alpha * NPL. Default 1.5. Purpose: avoid
    // mixed-depth trees when the average particles/box at some level sits
    // just above NPL — a handful of "hot" boxes splitting while their
    // siblings stay as leaves roughly doubles the non-leaf population at
    // the next level, which explodes PW cost. Staying as a leaf costs O(x²)
    // extra direct work for a leaf with x particles but keeps the tree
    // uniform-depth.
    const double npl_alpha = []() {
        const char *s = std::getenv("DMK_NPL_ALPHA");
        return s ? std::atof(s) : 1.5;
    }();
    const int ndiv_effective = (int)std::ceil(ndiv * npl_alpha);
    int max_level = 0;

    for (int level = 0; level < 50; level++) {
        // Find boxes at this level that need refinement
        bool any_refined = false;
        Real bsh = rc / (Real)(1 << (level + 2)); // half child boxsize = child_boxsize / 2

        // Collect boxes at this level
        std::vector<int> boxes_at_level;
        for (int b = 0; b < nboxes_; b++)
            if (box_depth[b] == level && box_nchild[b] == 0)
                boxes_at_level.push_back(b);

        for (int b : boxes_at_level) {
            int ist = src_se[2 * b];
            int ien = src_se[2 * b + 1];
            int npts = ien - ist + 1;
            if (npts <= ndiv_effective) continue;

            // Refine: create mc children
            any_refined = true;
            box_nchild[b] = mc;

            for (int c = 0; c < mc; c++) {
                int child_id = nboxes_;
                box_children[b][c] = child_id;

                // Grow arrays
                box_depth.push_back(level + 1);
                box_parent.push_back(b);
                box_nchild.push_back(0);
                box_children.push_back({});
                box_children.back().fill(-1);
                box_ncoll.push_back(0);
                box_colleagues.push_back({});
                box_colleagues.back().fill(-1);

                // Child center
                for (int d = 0; d < DIM; d++)
                    all_centers[child_id * DIM + d] = all_centers[b * DIM + d] + child_sign(c, d) * bsh;

                // Initialize ranges as empty
                if (2 * child_id + 1 < (int)src_se.size()) {
                    src_se[2 * child_id] = 0;
                    src_se[2 * child_id + 1] = -1;
                }
                if (2 * child_id + 1 < (int)trg_se.size()) {
                    trg_se[2 * child_id] = 0;
                    trg_se[2 * child_id + 1] = -1;
                }

                nboxes_++;
            }

            // Sort parent's sources into children
            sort_to_children<Real, DIM>(b, nboxes_, all_centers.data(),
                                        box_children[b].data(),
                                        npts, ist,
                                        src_sorted.data(), chg_sorted.data(), nd,
                                        src_perm.data(), src_se.data());

            // Sort parent's targets into children
            int tist = trg_se[2 * b];
            int tien = trg_se[2 * b + 1];
            int tnpts = tien - tist + 1;
            if (tnpts > 0) {
                sort_to_children<Real, DIM>(b, nboxes_, all_centers.data(),
                                            box_children[b].data(),
                                            tnpts, tist,
                                            trg_sorted.data(), nullptr, 0,
                                            trg_perm.data(), trg_se.data());
            }
        }

        if (!any_refined) break;
        max_level = level + 1;

        // Build colleague lists for the new level
        // For each child: colleagues are children of parent's colleagues that are adjacent
        Real child_bs = rc / (Real)(1 << (level + 1));
        Real dp1 = 1.05 * child_bs;

        for (int b = 0; b < nboxes_; b++) {
            if (box_depth[b] != level + 1) continue;
            int dad = box_parent[b];

            box_ncoll[b] = 0;
            box_colleagues[b].fill(-1);

            // Iterate ALL stencil positions of parent's colleagues
            for (int ic = 0; ic < ncoll_max; ic++) {
                int jbox = box_colleagues[dad][ic];
                if (jbox < 0 || box_nchild[jbox] == 0) continue;

                // Non-leaf colleague: check each child for adjacency
                for (int jc = 0; jc < mc; jc++) {
                    int kbox = box_children[jbox][jc];
                    if (kbox < 0) continue;

                    bool adjacent = true;
                    for (int d = 0; d < DIM; d++) {
                        Real dist = std::abs(all_centers[b * DIM + d] - all_centers[kbox * DIM + d]);
                        if (dist > dp1) { adjacent = false; break; }
                    }
                    if (!adjacent) continue;

                    // Compute stencil position from spatial relationship
                    // stencil index = (dz+1)*9 + (dy+1)*3 + (dx+1) for 3D
                    //               = (dy+1)*3 + (dx+1) for 2D
                    int stencil_idx = 0;
                    int stride = 1;
                    for (int d = 0; d < DIM; d++) {
                        Real diff = all_centers[kbox * DIM + d] - all_centers[b * DIM + d];
                        int di = (int)std::round(diff / child_bs);
                        stencil_idx += (di + 1) * stride;
                        stride *= 3;
                    }
                    if (stencil_idx >= 0 && stencil_idx < ncoll_max) {
                        box_colleagues[b][stencil_idx] = kbox;
                        box_ncoll[b]++;
                    }
                }
            }
        }
    }

    // ============================================
    // Step 6b: 2:1 (level-restricted) tree balancing.
    // A leaf at level L must not have any colleague (at level L) whose subtree
    // contains boxes at level > L+1 adjacent to the leaf.  If it does, refine
    // the leaf to level L+1.  Iterate until no more forced refinements are
    // needed.  This matches free-space DMKPtTree's `balance21=true` and
    // Fortran pts_tree_rgrid_build's 2:1 pass.
    // ============================================
    {
        bool balance_changed = true;
        int balance_iter = 0;
        while (balance_changed && balance_iter++ < 50) {
            balance_changed = false;

            // Iterate shallowest-to-deepest so we process a leaf at level L
            // before checking whether refining it triggers further level-L+1
            // violations in the SAME balance iteration.  Those will be caught
            // on the next outer iteration.
            for (int L = 0; L <= max_level; ++L) {
                const Real lvl_bs = rc / (Real)(1 << L);
                const Real child_bs_L = rc / (Real)(1 << (L + 1));
                const Real adj_tol = 1.05 * 0.5 * (lvl_bs + child_bs_L);

                // Snapshot leaves at level L (box list may grow inside the loop).
                std::vector<int> leaves_L;
                for (int b = 0; b < nboxes_; b++)
                    if (box_depth[b] == L && box_nchild[b] == 0 &&
                        (src_se[2*b + 1] - src_se[2*b] + 1) > 0)
                        leaves_L.push_back(b);

                for (int b : leaves_L) {
                    if (box_nchild[b] != 0) continue; // already refined this pass

                    // Check: does any colleague's subtree contain a non-leaf grandchild
                    // adjacent to b (meaning level L+2 exists and touches b)?
                    bool violate = false;
                    for (int ic = 0; ic < ncoll_max && !violate; ic++) {
                        int cc = box_colleagues[b][ic];
                        if (cc < 0 || cc == b) continue;
                        if (box_nchild[cc] == 0) continue;
                        for (int jc = 0; jc < mc; jc++) {
                            int gc = box_children[cc][jc];
                            if (gc < 0) continue;
                            if (box_nchild[gc] == 0) continue; // gc is leaf at L+1 — fine
                            // gc is non-leaf at level L+1: its children at L+2 are adjacent to some
                            // neighborhood.  Check if gc itself is adjacent to b (shares a face/edge/corner).
                            bool adj = true;
                            for (int d = 0; d < DIM; d++) {
                                Real dist = std::abs(all_centers[b * DIM + d] - all_centers[gc * DIM + d]);
                                if (dist > adj_tol) { adj = false; break; }
                            }
                            if (adj) { violate = true; break; }
                        }
                    }
                    if (!violate) continue;

                    // Refine b to level L+1.
                    int ist = src_se[2 * b];
                    int ien = src_se[2 * b + 1];
                    int npts = ien - ist + 1;

                    Real bsh = rc / (Real)(1 << (L + 2));
                    box_nchild[b] = mc;
                    for (int c = 0; c < mc; c++) {
                        int child_id = nboxes_;
                        box_children[b][c] = child_id;
                        box_depth.push_back(L + 1);
                        box_parent.push_back(b);
                        box_nchild.push_back(0);
                        box_children.push_back({});
                        box_children.back().fill(-1);
                        box_ncoll.push_back(0);
                        box_colleagues.push_back({});
                        box_colleagues.back().fill(-1);
                        for (int d = 0; d < DIM; d++)
                            all_centers[child_id * DIM + d] =
                                all_centers[b * DIM + d] + child_sign(c, d) * bsh;
                        if (2 * child_id + 1 < (int)src_se.size()) {
                            src_se[2 * child_id] = 0;
                            src_se[2 * child_id + 1] = -1;
                        }
                        if (2 * child_id + 1 < (int)trg_se.size()) {
                            trg_se[2 * child_id] = 0;
                            trg_se[2 * child_id + 1] = -1;
                        }
                        nboxes_++;
                    }
                    if (npts > 0)
                        sort_to_children<Real, DIM>(b, nboxes_, all_centers.data(),
                                                    box_children[b].data(),
                                                    npts, ist,
                                                    src_sorted.data(), chg_sorted.data(), nd,
                                                    src_perm.data(), src_se.data());
                    int tist = trg_se[2 * b], tien = trg_se[2 * b + 1];
                    int tnpts = tien - tist + 1;
                    if (tnpts > 0)
                        sort_to_children<Real, DIM>(b, nboxes_, all_centers.data(),
                                                    box_children[b].data(),
                                                    tnpts, tist,
                                                    trg_sorted.data(), nullptr, 0,
                                                    trg_perm.data(), trg_se.data());

                    // Build colleague lists for b's new children at level L+1.
                    Real child_bs = child_bs_L;
                    Real cdp1 = 1.05 * child_bs;
                    for (int c = 0; c < mc; c++) {
                        int nb = box_children[b][c];
                        box_ncoll[nb] = 0;
                        box_colleagues[nb].fill(-1);
                        for (int ic = 0; ic < ncoll_max; ic++) {
                            int jbox = box_colleagues[b][ic]; // b is parent of nb
                            if (jbox < 0 || box_nchild[jbox] == 0) continue;
                            for (int jc = 0; jc < mc; jc++) {
                                int kbox = box_children[jbox][jc];
                                if (kbox < 0) continue;
                                bool adj = true;
                                for (int d = 0; d < DIM; d++) {
                                    Real dist = std::abs(all_centers[nb * DIM + d] - all_centers[kbox * DIM + d]);
                                    if (dist > cdp1) { adj = false; break; }
                                }
                                if (!adj) continue;
                                int sidx = 0, stride = 1;
                                for (int d = 0; d < DIM; d++) {
                                    Real diff = all_centers[kbox * DIM + d] - all_centers[nb * DIM + d];
                                    int di = (int)std::round(diff / child_bs);
                                    sidx += (di + 1) * stride;
                                    stride *= 3;
                                }
                                if (sidx >= 0 && sidx < ncoll_max) {
                                    box_colleagues[nb][sidx] = kbox;
                                    box_ncoll[nb]++;
                                }
                            }
                        }
                    }
                    // The new children may in turn make *existing* level-L+1 boxes
                    // (siblings or colleagues-of-siblings) see grandchildren of b's
                    // now-non-leaf siblings.  We must add b's new level-L+1 children
                    // as colleagues of any level-L+1 box adjacent to them, and also
                    // reverse-link.
                    for (int c = 0; c < mc; c++) {
                        int nb = box_children[b][c];
                        for (int ic = 0; ic < ncoll_max; ic++) {
                            int kbox = box_colleagues[nb][ic];
                            if (kbox < 0 || kbox == nb) continue;
                            // Make sure kbox also lists nb as a colleague (at the
                            // mirror stencil position).
                            int sidx_rev = 0, stride = 1;
                            for (int d = 0; d < DIM; d++) {
                                Real diff = all_centers[nb * DIM + d] - all_centers[kbox * DIM + d];
                                int di = (int)std::round(diff / child_bs);
                                sidx_rev += (di + 1) * stride;
                                stride *= 3;
                            }
                            if (sidx_rev >= 0 && sidx_rev < ncoll_max &&
                                box_colleagues[kbox][sidx_rev] != nb) {
                                if (box_colleagues[kbox][sidx_rev] < 0) box_ncoll[kbox]++;
                                box_colleagues[kbox][sidx_rev] = nb;
                            }
                        }
                    }

                    balance_changed = true;
                    if (L + 1 > max_level) max_level = L + 1;
                }
            }
        }
        logger->info("PBCTree: 2:1 balance converged in {} iterations", balance_iter);
    }

    logger->info("PBCTree: {} boxes, {} levels (level 0: {} boxes)", nboxes_, max_level + 1, nboxes0);

    // ============================================
    // Step 7: Finalize tree data structures
    // ============================================

    // Copy centers to sctl::Vector
    centers.ReInit(DIM * nboxes_);
    std::memcpy(&centers[0], all_centers.data(), DIM * nboxes_ * sizeof(Real));

    // Build level_indices and boxsize
    int n_lev = max_level + 1;
    level_indices.ReInit(n_lev);
    for (int b = 0; b < nboxes_; b++)
        level_indices[box_depth[b]].PushBack(b);

    boxsize.ReInit(n_lev + 1); // +1 for virtual level used in self-interaction
    boxsize[0] = rc;
    for (int l = 1; l <= n_lev; l++)
        boxsize[l] = boxsize[l - 1] / Real(2);

    // is_global_leaf (single-rank: leaf = no children)
    is_global_leaf.ReInit(nboxes_);
    for (int b = 0; b < nboxes_; b++)
        is_global_leaf[b] = (box_nchild[b] == 0);

    // Source/target counts per box (leaf counts from sort, subtree accumulated)
    src_counts_owned.ReInit(nboxes_);
    trg_counts_owned.ReInit(nboxes_);
    src_counts_owned.SetZero();
    trg_counts_owned.SetZero();

    // Leaf particle counts
    for (int b = 0; b < nboxes_; b++) {
        if (is_global_leaf[b]) {
            int ist = src_se[2 * b], ien = src_se[2 * b + 1];
            src_counts_owned[b] = (ist <= ien) ? ien - ist + 1 : 0;
            int tist = trg_se[2 * b], tien = trg_se[2 * b + 1];
            trg_counts_owned[b] = (tist <= tien) ? tien - tist + 1 : 0;
        }
    }

    // Accumulate subtree counts bottom-up
    for (int l = n_lev - 1; l >= 0; l--) {
        for (auto b : level_indices[l]) {
            int dad = box_parent[b];
            if (dad >= 0) {
                src_counts_owned[dad] += src_counts_owned[b];
                trg_counts_owned[dad] += trg_counts_owned[b];
            }
        }
    }

    // Compute data offsets
    // Source offsets: each box's sorted source range
    r_src_offsets_owned.ReInit(nboxes_);
    r_src_offsets_with_halo.ReInit(nboxes_);
    charge_offsets_owned.ReInit(nboxes_);
    charge_offsets_with_halo.ReInit(nboxes_);
    r_trg_offsets_owned.ReInit(nboxes_);
    pot_src_offsets.ReInit(nboxes_);
    pot_trg_offsets.ReInit(nboxes_);

    for (int b = 0; b < nboxes_; b++) {
        int ist = src_se[2 * b];
        r_src_offsets_owned[b] = ist * DIM;
        r_src_offsets_with_halo[b] = ist * DIM;
        charge_offsets_owned[b] = ist * nd;
        charge_offsets_with_halo[b] = ist * nd;

        int tist = trg_se[2 * b];
        r_trg_offsets_owned[b] = tist * DIM;
    }

    // Copy sorted data into sctl::Vectors
    r_src_sorted_owned.ReInit(DIM * nsall);
    std::memcpy(&r_src_sorted_owned[0], src_sorted.data(), DIM * nsall * sizeof(Real));

    charge_sorted_owned.ReInit(nd * nsall);
    std::memcpy(&charge_sorted_owned[0], chg_sorted.data(), nd * nsall * sizeof(Real));

    r_trg_sorted_owned.ReInit(DIM * n_trg);
    if (n_trg > 0)
        std::memcpy(&r_trg_sorted_owned[0], trg_sorted.data(), DIM * n_trg * sizeof(Real));

    // Allocate output potential arrays
    pot_src_sorted.ReInit(kernel_output_dim_src * nsall);
    pot_src_sorted.SetZero();
    pot_trg_sorted.ReInit(kernel_output_dim_trg * n_trg);
    pot_trg_sorted.SetZero();

    // Compute pot offsets from src_se / trg_se
    for (int b = 0; b < nboxes_; b++) {
        int ist = src_se[2 * b];
        pot_src_offsets[b] = ist * kernel_output_dim_src;
        int tist = trg_se[2 * b];
        pot_trg_offsets[b] = tist * kernel_output_dim_trg;
    }

    // Verify particles are within their assigned boxes
    {
        int n_misplaced = 0;
        for (int b = 0; b < nboxes_; b++) {
            if (!is_global_leaf[b]) continue;
            int ist = src_se[2 * b], ien = src_se[2 * b + 1];
            Real bs = boxsize[box_depth[b]] / Real(2);
            for (int i = ist; i <= ien; i++) {
                for (int d = 0; d < DIM; d++) {
                    Real diff = std::abs(src_sorted[i * DIM + d] - all_centers[b * DIM + d]);
                    if (diff > bs * 1.01) {
                        if (n_misplaced < 5)
                            logger->error("Box {} (depth {}, center={:.4f}): particle {} at {:.4f}, diff={:.4f} > bs/2={:.4f}",
                                          b, box_depth[b], all_centers[b * DIM + d], i, src_sorted[i * DIM + d], diff, bs);
                        n_misplaced++;
                    }
                }
            }
        }
        if (n_misplaced > 0) logger->error("Total misplaced particles: {}", n_misplaced);
    }

    // Verify bounds before metadata generation
    for (int b = 0; b < nboxes_; b++) {
        int ist = src_se[2 * b], ien = src_se[2 * b + 1];
        if (ist < 0 || (ien >= nsall && is_global_leaf[b])) {
            logger->error("Box {} (depth {}): src_se=[{},{}] OOB (nsall={})", b, box_depth[b], ist, ien, nsall);
        }
        int tist = trg_se[2 * b], tien = trg_se[2 * b + 1];
        if (tist < 0 || (tien >= n_trg && is_global_leaf[b] && n_trg > 0)) {
            logger->error("Box {} (depth {}): trg_se=[{},{}] OOB (n_trg={})", b, box_depth[b], tist, tien, n_trg);
        }
    }
    logger->info("Bounds check passed");

    // ============================================
    // Step 8: Generate DMK metadata
    // ============================================
    generate_metadata();

    logger->info("PBCTree construction complete");
}

// ================================================================
// generate_metadata: ifpwexp, interaction lists, Fourier data, evaluators
// ================================================================

template <typename Real, int DIM>
void PBCTree<Real, DIM>::generate_metadata() {
    logger->info("metadata: proxy flags (nboxes={})", nboxes_);
    compute_proxy_expansion_flags();
    compute_proxy_evaluation_flags();
    logger->info("metadata: lists");
    build_plane_wave_interaction_lists();
    build_direct_interaction_lists();
    build_upward_pass_work_lists();
    build_direct_work_lists();
    logger->info("metadata: proxy alloc");
    allocate_proxy_coefficients();

    logger->info("metadata: FourierData ({} levels, boxsize.Dim={})", n_levels(), boxsize.Dim());
    fourier_data = FourierData<Real>(params.kernel, DIM, params.eps, expansion_constants.n_pw_win,
                                     params.fparam, expansion_constants.beta, boxsize);
    logger->info("metadata: FourierData done, precompute (n_pw={})...", fourier_data.n_pw());
    precompute_window_difference_data();
    logger->info("metadata: precompute done");
    logger->info("metadata: evaluators");
    build_evaluators();

    // c2p / p2c matrices
    std::tie(c2p, p2c) = dmk::chebyshev::get_c2p_p2c_matrices<Real>(DIM, n_order);
}

// ================================================================
// Metadata sub-steps
// ================================================================

template <typename Real, int DIM>
void PBCTree<Real, DIM>::compute_proxy_expansion_flags() {
    ifpwexp.ReInit(nboxes_);
    ifpwexp.SetZero();

    // PBC fixup A: ALL level-0 boxes get ifpwexp=true.  This differs from
    // Fortran pdmkmain_pbc (which leaves ifpwexp=0 for level-0 leaves with
    // all-leaf colleagues), but it is needed to avoid a latent bug in
    // get_self_interaction_constant: at i_level=0 that function assumes the
    // correction is for a level-0 box "as if ifpwexp=1", using 0.5*boxsize.
    // With ifpwexp=1 forced everywhere at level 0, the correction always
    // uses w0[depth=level+1], which is correctly sized regardless.
    for (auto b : level_indices[0])
        ifpwexp[b] = true;

    // Standard logic for deeper levels
    for (int l = 1; l < n_levels(); l++) {
        for (auto b : level_indices[l]) {
            if (box_nchild[b] > 0) {
                ifpwexp[b] = true;
                continue;
            }
            // Leaf: check if any colleague is non-leaf.
            // Note: box_colleagues is indexed by stencil position (0..ncoll_max-1),
            // with -1 for out-of-grid positions. Iterate the full stencil.
            for (int ic = 0; ic < ncoll_max; ic++) {
                int nb = box_colleagues[b][ic];
                if (nb >= 0 && box_nchild[nb] > 0) {
                    ifpwexp[b] = true;
                    break;
                }
            }
        }
    }

    // Clear for empty subtrees
    for (int b = 0; b < nboxes_; b++)
        if (src_counts_owned[b] == 0 && trg_counts_owned[b] == 0)
            ifpwexp[b] = false;
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::compute_proxy_evaluation_flags() {
    iftensprodeval.ReInit(nboxes_);
    iftensprodeval.SetZero();

    // A leaf with particles needs its proxy_down evaluated at its particles
    // regardless of ifpwexp: the PW contribution reaches it either through
    // its own PW (ifpwexp=1) or through ancestors' propagated proxy_down.
    // If we skip eval here the particles miss the entire PW contribution.
    for (int b = 0; b < nboxes_; b++) {
        if (!is_global_leaf[b]) continue;
        if (src_counts_owned[b] + trg_counts_owned[b] == 0) continue;
        iftensprodeval[b] = true;
    }
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::build_plane_wave_interaction_lists() {
    listpw_.resize(nboxes_);
    nlistpw_.resize(nboxes_, 0);

    for (int b = 0; b < nboxes_; b++) {
        nlistpw_[b] = 0;
        for (int ic = 0; ic < ncoll_max; ic++) {
            int nb = box_colleagues[b][ic];
            if (nb < 0) continue;
            if (is_global_leaf[b] || is_global_leaf[nb]) {
                listpw_[b][nlistpw_[b]++] = nb;
            }
        }
    }
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::build_direct_interaction_lists() {
    list1_.resize(nboxes_);
    nlist1_.resize(nboxes_, 0);

    for (int l = 0; l < n_levels(); l++) {
        for (auto box : level_indices[l]) {
            if (!is_global_leaf[box]) continue;
            nlist1_[box] = 0;

            // (a) and (b): colleagues and their children
            for (int ic = 0; ic < ncoll_max; ic++) {
                int nb = box_colleagues[box][ic];
                if (nb < 0) continue;

                if (is_global_leaf[nb] && src_counts_owned[nb] > 0) {
                    // (a) Leaf colleague
                    list1_[box][nlist1_[box]++] = nb;
                } else if (box_nchild[nb] > 0 && l + 1 < n_levels()) {
                    // (b) Non-leaf colleague: add close children
                    Real cutoff = 1.05 * 0.75 * boxsize[l];
                    for (int jc = 0; jc < mc; jc++) {
                        int kbox = box_children[nb][jc];
                        if (kbox < 0 || src_counts_owned[kbox] == 0) continue;
                        if (!is_global_leaf[kbox]) continue;

                        bool inrange = true;
                        for (int d = 0; d < DIM; d++) {
                            if (std::abs(center_ptr(box)[d] - center_ptr(kbox)[d]) > cutoff) {
                                inrange = false; break;
                            }
                        }
                        if (inrange) list1_[box][nlist1_[box]++] = kbox;
                    }
                }
            }

            // (c) Leaf colleagues of parent — skip for level 0 (parent = -1)
            if (l == 0) continue;
            int dad = box_parent[box];
            if (dad < 0) continue;

            Real cutoff2 = 1.5 * 1.05 * boxsize[l];
            for (int ic = 0; ic < ncoll_max; ic++) {
                int nb = box_colleagues[dad][ic];
                if (nb < 0 || !is_global_leaf[nb] || src_counts_owned[nb] == 0) continue;

                bool inrange = true;
                for (int d = 0; d < DIM; d++) {
                    if (std::abs(center_ptr(box)[d] - center_ptr(nb)[d]) > cutoff2) {
                        inrange = false; break;
                    }
                }
                if (inrange) list1_[box][nlist1_[box]++] = nb;
            }
        }
    }
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::build_upward_pass_work_lists() {
    has_proxy_from_children.ReInit(nboxes_);
    has_proxy_from_children.SetZero();
    charge2proxy_work.clear();

    for (int l = n_levels() - 1; l >= 0; l--) {
        for (auto b : level_indices[l]) {
            if (!ifpwexp[b] || src_counts_owned[b] == 0) continue;

            bool child_has_proxy = false;
            for (int c = 0; c < mc; c++) {
                int ch = box_children[b][c];
                if (ch >= 0 && ifpwexp[ch] && src_counts_owned[ch] > 0) {
                    child_has_proxy = true;
                    break;
                }
            }

            if (child_has_proxy) {
                has_proxy_from_children[b] = true;
                // Add work for children without proxy.  Scale is PARENT's (l),
                // not child's (l+1): proxy is formed in parent b's basis using
                // child ch's source positions.
                for (int c = 0; c < mc; c++) {
                    int ch = box_children[b][c];
                    if (ch >= 0 && src_counts_owned[ch] > 0 && !ifpwexp[ch])
                        charge2proxy_work.push_back({ch, b, l});
                }
            } else {
                charge2proxy_work.push_back({b, b, l});
            }
        }
    }
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::build_direct_work_lists() {
    direct_work.clear();
    for (int b = 0; b < nboxes_; b++) {
        if (!is_global_leaf[b] || nlist1_[b] == 0) continue;
        if (src_counts_owned[b] == 0 && trg_counts_owned[b] == 0) continue;
        direct_work.push_back(b);
    }
    // Sort by descending work estimate
    std::sort(direct_work.begin(), direct_work.end(), [&](int a, int b_) {
        int wa = (src_counts_owned[a] + trg_counts_owned[a]);
        int wb = (src_counts_owned[b_] + trg_counts_owned[b_]);
        return wa > wb;
    });
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::allocate_proxy_coefficients() {
    const int n_coeffs = n_tables * sctl::pow<DIM>(n_order);
    const int n_pw = expansion_constants.n_pw_diff;
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * n_tables;

    // Upward proxy coefficients
    proxy_coeffs_offsets.ReInit(nboxes_);
    sctl::Long total_up = 0;
    for (int b = 0; b < nboxes_; b++) {
        if (ifpwexp[b] && src_counts_owned[b] > 0) {
            proxy_coeffs_offsets[b] = total_up;
            total_up += n_coeffs;
        } else {
            proxy_coeffs_offsets[b] = -1;
        }
    }
    proxy_coeffs_upward.ReInit(total_up);
    proxy_coeffs_upward.SetZero();

    // Downward proxy coefficients
    proxy_coeffs_offsets_downward.ReInit(nboxes_);
    sctl::Long total_down = 0;
    for (int b = 0; b < nboxes_; b++) {
        if ((ifpwexp[b] || iftensprodeval[b]) && (src_counts_owned[b] + trg_counts_owned[b] > 0)) {
            proxy_coeffs_offsets_downward[b] = total_down;
            total_down += n_coeffs;
        } else {
            proxy_coeffs_offsets_downward[b] = -1;
        }
    }
    proxy_coeffs_downward.ReInit(total_down);
    proxy_coeffs_downward.SetZero();

    // Plane wave outgoing expansion
    pw_out_offsets.ReInit(nboxes_);
    sctl::Long total_pw = 0;
    for (int b = 0; b < nboxes_; b++) {
        if (ifpwexp[b] && src_counts_owned[b] > 0) {
            pw_out_offsets[b] = total_pw;
            total_pw += n_pw_per_box;
        } else {
            pw_out_offsets[b] = -1;
        }
    }
    pw_out.ReInit(total_pw);
    pw_out.SetZero();

    proxy_down_zeroed.assign(nboxes_, 0);
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::precompute_window_difference_data() {
    const int n_pw = fourier_data.n_pw();
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);

    difference_fourier_data.resize(n_levels());

    // Windowed kernel (root level — still precomputed even if skipped for PBC)
    {
        sctl::Vector<Real> kernel_ft;
        get_windowed_kernel_ft<Real, DIM>(params.kernel, &params.fparam, fourier_data.beta(), n_pw,
                                           boxsize[0], fourier_data.prolate0_fun, kernel_ft);
        window_fourier_data.radialft.ReInit(n_pw_modes);
        window_fourier_data.poly2pw.ReInit(n_order * n_pw);
        window_fourier_data.pw2poly.ReInit(n_order * n_pw);
        util::mk_tensor_product_fourier_transform(DIM, n_pw,
            ndview<Real, 1>({kernel_ft.Dim()}, &kernel_ft[0]),
            ndview<Real, 1>({n_pw_modes}, &window_fourier_data.radialft[0]));
        fourier_data.calc_planewave_coeff_matrices(-1, n_order, window_fourier_data.poly2pw,
                                                    window_fourier_data.pw2poly);
    }

    // DEBUG: report radialft magnitudes per level
    if (getenv("DMK_PBC_CHECK_NAN")) {
        logger->warn("windowed radialft: max={:.3e}",
                     [&]() { double m = 0;
                         for (int i = 0; i < window_fourier_data.radialft.Dim(); i++)
                             m = std::max(m, (double)std::abs(window_fourier_data.radialft[i]));
                         return m; }());
    }

    // Difference kernels per level.  kernel_ft is hoisted outside the loop so
    // that get_difference_kernel_ft's scale-factor branch (for l > 0 Laplace/
    // SQRT_LAPLACE) can read and multiply the level-0 values in place.  Without
    // this hoist, levels > 0 would see an empty kernel_ft and radialft would be
    // incorrectly zeroed — a latent bug that doesn't affect 2-level tests where
    // ifpwexp=false at level 1, but breaks 3+ level trees with level>0 PW.
    sctl::Vector<Real> kernel_ft;
    for (int l = 0; l < n_levels(); l++) {
        auto &lfd = difference_fourier_data[l];
        bool is_root = (l == 0);

        lfd.radialft.ReInit(n_pw_modes);
        lfd.wpwshift.ReInit(n_pw_modes * sctl::pow<DIM>(3));
        lfd.poly2pw.ReInit(n_order * n_pw);
        lfd.pw2poly.ReInit(n_order * n_pw);

        get_difference_kernel_ft<Real, DIM>(is_root, params.kernel, &params.fparam, fourier_data.beta(), n_pw,
                                             boxsize[l], fourier_data.prolate0_fun, kernel_ft);
        util::mk_tensor_product_fourier_transform(DIM, n_pw,
            ndview<Real, 1>({kernel_ft.Dim()}, &kernel_ft[0]),
            ndview<Real, 1>({n_pw_modes}, &lfd.radialft[0]));
        fourier_data.calc_planewave_coeff_matrices(l, n_order, lfd.poly2pw, lfd.pw2poly);
        dmk::calc_planewave_translation_matrix<DIM>(1, boxsize[l], n_pw,
                                                     fourier_data.difference_kernel(l).hpw, lfd.wpwshift);

        // Fuse kernel_ft (radialft) into wpwshift: each of the ncoll_max=3^DIM
        // translation stencils is pre-multiplied by the radial kernel FT. This
        // lets form_outgoing_expansions skip the per-box multiply_kernelFT pass
        // (~40 ms at 1M/1e-3) — the fused shift folds kernel × translation
        // into a single complex multiply per mode during shift_planewave.
        //
        // wpwshift layout (per calc_planewave_translation_matrix): per stencil
        // block, real parts then imag parts (SoA), each of length n_pw_modes.
        // radialft is real-valued so it scales both real and imag halves.
        {
            const int n_stencils = sctl::pow<DIM>(3);
            for (int s = 0; s < n_stencils; ++s) {
                Real *shift_r = reinterpret_cast<Real *>(&lfd.wpwshift[s * n_pw_modes]);
                Real *shift_i = shift_r + n_pw_modes;
                for (int m = 0; m < n_pw_modes; ++m) {
                    shift_r[m] *= lfd.radialft[m];
                    shift_i[m] *= lfd.radialft[m];
                }
            }
        }
        if (getenv("DMK_PBC_CHECK_NAN")) {
            double mrft = 0, mp2pw = 0, mpw2p = 0, mshift = 0;
            for (int i = 0; i < lfd.radialft.Dim(); i++)
                mrft = std::max(mrft, (double)std::abs(lfd.radialft[i]));
            for (int i = 0; i < lfd.poly2pw.Dim(); i++)
                mp2pw = std::max(mp2pw, (double)std::abs(lfd.poly2pw[i]));
            for (int i = 0; i < lfd.pw2poly.Dim(); i++)
                mpw2p = std::max(mpw2p, (double)std::abs(lfd.pw2poly[i]));
            for (int i = 0; i < lfd.wpwshift.Dim(); i++)
                mshift = std::max(mshift, (double)std::abs(lfd.wpwshift[i]));
            logger->warn("level {} boxsize={:.4e} hpw={:.4e} radialft max={:.3e} poly2pw max={:.3e} pw2poly max={:.3e} wpwshift max={:.3e}",
                         l, boxsize[l], fourier_data.difference_kernel(l).hpw, mrft, mp2pw, mpw2p, mshift);
        }
    }
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::build_evaluators() {
    auto eval = make_evaluator_aot<Real>(params.kernel, params.pgh_src, DIM, n_digits, 3);
    auto eval_trg = make_evaluator_aot<Real>(params.kernel, params.pgh_trg, DIM, n_digits, 3);
#ifdef DMK_USE_JIT
    if (!util::env_is_set("DMK_DEBUG_FORCE_AOT")) {
        eval = make_evaluator_jit<Real>(params.kernel, params.pgh_src, DIM, n_digits,
                                         expansion_constants.beta, 3);
        eval_trg = make_evaluator_jit<Real>(params.kernel, params.pgh_trg, DIM, n_digits,
                                             expansion_constants.beta, 3);
    }
#endif
    // +1 because self-interaction at leaf level with ifpwexp uses level+1
    evaluator_by_level_src.assign(n_levels() + 1, eval);
    evaluator_by_level_trg.assign(n_levels() + 1, eval_trg);

    // Workspaces for OMP threads
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    workspaces_.ReInit(n_threads);
}

// ================================================================
// Eval methods (adapted from DMKPtTree with SCTL → stored arrays)
// ================================================================

template <typename Real, int DIM>
void PBCTree<Real, DIM>::init_planewave_data() {
    // Already allocated in allocate_proxy_coefficients; just zero pw_out
    pw_out.SetZero();
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::upward_pass() {
    const std::size_t n_coeffs = n_tables * sctl::pow<DIM>(n_order);
    logger->info("upward pass started");

#pragma omp parallel
#pragma omp single
    workspaces_.ReInit(MY_OMP_GET_NUM_THREADS());

    proxy_coeffs_upward.SetZero();

    constexpr int n_children = 1u << DIM;

    // charge2proxycharge
#pragma omp parallel
    {
        sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];
#pragma omp for schedule(static)
        for (int i = 0; i < (int)charge2proxy_work.size(); ++i) {
            const auto &w = charge2proxy_work[i];
            proxy::charge2proxycharge<Real, DIM>(r_src_owned_view(w.src_box), charge_owned_view(w.src_box),
                                                  this->center_view(w.center_box), 2.0 / boxsize[w.level],
                                                  proxy_view_upward(w.center_box), workspace);
        }
    }

    // tensorprod::transform (child → parent aggregation, bottom-up)
#pragma omp parallel
    {
        sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];
        for (int i_level = n_levels() - 1; i_level >= 0; --i_level) {
#pragma omp for schedule(static)
            for (int idx = 0; idx < level_indices[i_level].Dim(); ++idx) {
                const int i_box = level_indices[i_level][idx];
                if (!has_proxy_from_children[i_box]) continue;

                for (int ic = 0; ic < n_children; ++ic) {
                    const int cb = box_children[i_box][ic];
                    if (cb < 0 || !(src_counts_owned[cb] > 0 && ifpwexp[cb])) continue;

                    const ndview<Real, 2> c2p_view({n_order, DIM}, &c2p[ic * DIM * n_order * n_order]);
                    tensorprod::transform<Real, DIM>(n_tables, true, proxy_view_upward(cb), c2p_view,
                                                     proxy_view_upward(i_box), workspace);
                }
            }
        }
    }

    // No MPI broadcast needed (single-rank)
    logger->info("upward pass finished");
    if (getenv("DMK_PBC_CHECK_NAN")) {
        const int npbox = sctl::pow<DIM>(n_order);
        for (int b = 0; b < nboxes_; b++) {
            if (proxy_coeffs_offsets[b] < 0) continue;
            double maxv = 0;
            bool nan = false;
            for (int k = 0; k < npbox; k++) {
                Real v = proxy_coeffs_upward[proxy_coeffs_offsets[b] + k];
                if (!std::isfinite(v)) { nan = true; break; }
                maxv = std::max(maxv, (double)std::abs(v));
            }
            if (nan) logger->error("proxy_up NaN at box={} lev={}", b, box_depth[b]);
            else if (maxv > 1e3)
                logger->warn("proxy_up HUGE box={} lev={} nch={} max={:.3e}",
                             b, box_depth[b], box_nchild[b], maxv);
        }
    }
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::form_outgoing_expansions() {
    const int n_pw = fourier_data.n_pw();
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * n_tables;

    // PBC: skip windowed kernel at root — NUFFT handles it (params.use_periodic==1).

#pragma omp parallel
    {
        sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];
#pragma omp for schedule(dynamic)
        for (int i_box = 0; i_box < n_boxes(); ++i_box) {
            if (ifpwexp[i_box] && proxy_coeffs_offsets[i_box] != -1) {
                const int level = box_depth[i_box];
                auto &dfd = difference_fourier_data[level];
                const ndview<std::complex<Real>, 2> poly2pw_view({n_pw, n_order}, &dfd.poly2pw[0]);

                dmk::proxy::proxycharge2pw<Real, DIM>(proxy_view_upward(i_box), poly2pw_view, pw_out_view(i_box),
                                                       workspace);
                // kernel_ft (dfd.radialft) is now fused into dfd.wpwshift in
                // precompute_window_difference_data, so no separate multiply here.
                if (getenv("DMK_PBC_CHECK_NAN")) {
                    const int n_pw_modes = sctl::pow<DIM-1>(n_pw) * ((n_pw+1)/2);
                    double maxv = 0;
                    for (int k = 0; k < n_pw_modes; k++) {
                        auto v = pw_out[pw_out_offsets[i_box] + k];
                        maxv = std::max(maxv, (double)std::abs(v));
                        if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) {
                            logger->error("pw_out NaN at box={} lev={}", i_box, level);
                            break;
                        }
                    }
                    if (maxv > 1e3)
                        logger->warn("pw_out HUGE box={} lev={} max={:.3e}", i_box, level, maxv);
                }
            } else if (pw_out_offsets[i_box] != -1) {
                pw_out_view(i_box) = 0;
            }
        }
    }
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::form_eval_expansions(const sctl::Vector<int> &boxes,
                                               const sctl::Vector<std::complex<Real>> &wpwshift,
                                               Real bsize,
                                               const ndview<std::complex<Real>, 2> &pw2poly_view,
                                               const sctl::Vector<Real> &p2c_mat) {
    const int n_pw = fourier_data.n_pw();
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * n_tables;
    const Real sc = 2.0 / bsize;
    const int nd = n_tables;

    const bool dbg_prof = (std::getenv("DMK_PBC_PROFILE") != nullptr);

#pragma omp parallel
    {
        sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];
        sctl::Vector<std::complex<Real>> pw_in(n_pw_per_box);

        double t_pw_local = 0, t_p2p_local = 0, t_p2c_local = 0, t_eval_local = 0;
        long n_shift_calls_local = 0;

        auto pw_in_view = [this, &pw_in, n_pw]() {
            if constexpr (DIM == 2)
                return ndview<std::complex<Real>, DIM + 1>({n_pw, (n_pw + 1) / 2, n_tables}, &pw_in[0]);
            else
                return ndview<std::complex<Real>, DIM + 1>({n_pw, n_pw, (n_pw + 1) / 2, n_tables}, &pw_in[0]);
        }();

#pragma omp for schedule(dynamic)
        for (auto box : boxes) {
            // Skip boxes with no downstream output. trg_counts_owned and
            // src_counts_owned are bottom-up subtree sums, so this also skips
            // non-leaf ancestors whose descendants are all empty (outer padding
            // cubes in PBC's rectangular grid, which hold only image sources —
            // their pw_out is still computed in form_outgoing for colleagues
            // to shift from, but they never emit any potential).
            const bool has_output =
                (kernel_output_dim_src > 0 && src_counts_owned[box] > 0) ||
                (kernel_output_dim_trg > 0 && trg_counts_owned[box] > 0);
            const int nboxpts = src_counts_owned[box] + trg_counts_owned[box];

            double _tA = dbg_prof ? MY_OMP_GET_WTIME() : 0.0;
            if (ifpwexp[box] && nboxpts && has_output) {
                // Self-shift. In nD the self stencil (all zero offsets) sits at
                // index (1,1,...) in the 3^DIM lattice: 13 for DIM=3, 4 for DIM=2.
                // That entry has imag=0 and its real part equals kernel_ft after
                // fusion, so this is a compact complex×real multiply.
                constexpr int self_stencil = [] {
                    int s = 0, stride = 1;
                    for (int d = 0; d < DIM; ++d) { s += stride; stride *= 3; }
                    return s;
                }();
                {
                    const Real *sr = reinterpret_cast<const Real *>(&wpwshift[self_stencil * n_pw_modes]);
                    const Real *po = reinterpret_cast<const Real *>(pw_out_ptr(box));
                    Real *pi = reinterpret_cast<Real *>(&pw_in[0]);
                    for (int d = 0; d < n_tables; ++d) {
                        const Real *po_d = po + d * 2 * n_pw_modes;
                        Real *pi_d = pi + d * 2 * n_pw_modes;
                        for (int m = 0; m < n_pw_modes; ++m) {
                            pi_d[2 * m]     = po_d[2 * m]     * sr[m];
                            pi_d[2 * m + 1] = po_d[2 * m + 1] * sr[m];
                        }
                    }
                }

                // Accumulate incoming expansions from colleagues.
                // Both SCTL nbr and rect_grid_setup use natural stencil order
                // (x fastest: ind = dx + dy*3 + dz*9), same as wpwshift.
                // Compute shift index from box-neighbor center offset.
                // wpwshift[t] is built with k[d] in {0,1,2} meaning shift
                // (k[d]-1)*bsize; t = k[0]*9 + k[1]*3 + k[2] where k[0]
                // indexes the first pw_expansion extent (z_half_freq in
                // calc_planewave_translation_matrix's outermost loop), k[2]
                // the innermost (x_freq). So physical (x,y,z) offsets map to
                // (k2, k1, k0) respectively.
                for (int ind = 0; ind < ncoll_max; ind++) {
                    int neighbor = box_colleagues[box][ind];
                    if (neighbor < 0 || neighbor == box) continue;
                    if (is_global_leaf[box] && is_global_leaf[neighbor]) continue;
                    if (pw_out_offsets[neighbor] == -1) continue;
                    // Stencil index = Σ_d (k_d + 1) * 3^d. Generic over DIM.
                    int shift_ind = 0;
                    int stride = 1;
                    for (int d = 0; d < DIM; ++d) {
                        int kd = (int)std::lround((center_ptr(box)[d] - center_ptr(neighbor)[d]) / bsize) + 1;
                        shift_ind += kd * stride;
                        stride *= 3;
                    }
                    // wpwshift has n_pw_modes complex entries per stencil,
                    // shared across densities; shift_planewave reads nd from
                    // pwexp.extent(DIM) and internally iterates.
                    ndview<const std::complex<Real>, 1> wpwshift_view({n_pw_modes},
                                                                      &wpwshift[n_pw_modes * shift_ind]);
                    shift_planewave<std::complex<Real>, DIM>(pw_out_view(neighbor), pw_in_view, wpwshift_view);
                    if (dbg_prof) n_shift_calls_local++;
                }

                // DEBUG: check pw_in for NaN/Inf
                if (getenv("DMK_PBC_CHECK_NAN")) {
                    for (int k = 0; k < n_pw_per_box; k++) {
                        if (!std::isfinite(pw_in[k].real()) || !std::isfinite(pw_in[k].imag())) {
                            logger->error("pw_in NaN/Inf at box={} lev={} k={}", box, box_depth[box], k);
                            break;
                        }
                    }
                }

                double _tB = dbg_prof ? MY_OMP_GET_WTIME() : 0.0;
                if (dbg_prof) t_pw_local += (_tB - _tA) * 1000.0;

                // Convert incoming PW to local expansion
                if (!proxy_down_zeroed[box]) {
                    proxy_view_downward(box) = 0;
                    proxy_down_zeroed[box] = true;
                }
                dmk::planewave_to_proxy_potential<Real, DIM>(pw_in_view, pw2poly_view, proxy_view_downward(box),
                                                              workspace);
                double _tC = dbg_prof ? MY_OMP_GET_WTIME() : 0.0;
                if (dbg_prof) t_p2p_local += (_tC - _tB) * 1000.0;
                if (getenv("DMK_PBC_CHECK_NAN")) {
                    const int npbox = sctl::pow<DIM>(n_order);
                    double maxv = 0;
                    for (int k = 0; k < npbox; k++) {
                        Real v = proxy_coeffs_downward[proxy_coeffs_offsets_downward[box] + k];
                        if (!std::isfinite(v)) {
                            logger->error("proxy_down NaN/Inf at box={} lev={} k={}", box, box_depth[box], k);
                            break;
                        }
                        maxv = std::max(maxv, (double)std::abs(v));
                    }
                    if (maxv > 1e6) {
                        logger->error("proxy_down HUGE at box={} lev={} max={}", box, box_depth[box], maxv);
                    }
                }

                if (!iftensprodeval[box]) {
                    // Propagate to children
                    constexpr int n_children = 1u << DIM;
                    for (int i_child = 0; i_child < n_children; ++i_child) {
                        const int child = box_children[box][i_child];
                        if (child < 0 || !(src_counts_owned[child] + trg_counts_owned[child])) continue;
                        const ndview<Real, 2> p2c_view({n_order, DIM},
                                                        const_cast<Real *>(&p2c_mat[i_child * DIM * n_order * n_order]));
                        tensorprod::transform<Real, DIM>(nd, proxy_down_zeroed[child], proxy_view_downward(box),
                                                          p2c_view, proxy_view_downward(child), workspace);
                        proxy_down_zeroed[child] = true;
                    }
                }
                double _tD = dbg_prof ? MY_OMP_GET_WTIME() : 0.0;
                if (dbg_prof) t_p2c_local += (_tD - _tC) * 1000.0;
            }

            double _tE = dbg_prof ? MY_OMP_GET_WTIME() : 0.0;
            if (iftensprodeval[box]) {
                if (src_counts_owned[box]) {
                    if (params.pgh_src == DMK_POTENTIAL)
                        proxy::eval_targets<Real, DIM, 1>(proxy_view_downward(box), r_src_owned_view(box),
                                                           this->center_view(box), sc, pot_src_view(box), workspace);
                    else if (params.pgh_src == DMK_POTENTIAL_GRAD)
                        proxy::eval_targets<Real, DIM, 2>(proxy_view_downward(box), r_src_owned_view(box),
                                                           this->center_view(box), sc, pot_src_view(box), workspace);
                }
                if (trg_counts_owned[box]) {
                    if (params.pgh_trg == DMK_POTENTIAL)
                        proxy::eval_targets<Real, DIM, 1>(proxy_view_downward(box), r_trg_owned_view(box),
                                                           this->center_view(box), sc, pot_trg_view(box), workspace);
                    else if (params.pgh_trg == DMK_POTENTIAL_GRAD)
                        proxy::eval_targets<Real, DIM, 2>(proxy_view_downward(box), r_trg_owned_view(box),
                                                           this->center_view(box), sc, pot_trg_view(box), workspace);
                }
            }
            double _tF = dbg_prof ? MY_OMP_GET_WTIME() : 0.0;
            if (dbg_prof) t_eval_local += (_tF - _tE) * 1000.0;
        }
        if (dbg_prof) {
#pragma omp atomic
            form_eval_t_pw_ += t_pw_local;
#pragma omp atomic
            form_eval_t_p2p_ += t_p2p_local;
#pragma omp atomic
            form_eval_t_p2c_ += t_p2c_local;
#pragma omp atomic
            form_eval_t_eval_ += t_eval_local;
#pragma omp atomic
            form_eval_n_shifts_ += n_shift_calls_local;
        }
    }
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::evaluate_direct_interactions() {
    Real w0[64]; // max depth + 1
    for (int i_level = 0; i_level < n_levels(); ++i_level)
        w0[i_level] = get_self_interaction_constant<Real, DIM>(fourier_data, params.kernel, i_level, boxsize[i_level]);
    // Extra level for self-interaction when ifpwexp[leaf] is true (uses level+1)
    if (n_levels() < 64) {
        Real bs_extra = boxsize[n_levels() - 1] / Real(2);
        w0[n_levels()] = get_self_interaction_constant<Real, DIM>(fourier_data, params.kernel, n_levels(), bs_extra);
    }
    std::atomic<long> n_pairs_tot{0}, n_src_pairs_tot{0}, n_trg_pairs_tot{0};
    std::atomic<long> n_trg_pairs_actual{0};  // post-filter (matches PeriodicDMK's "pairs=")
    const bool debug_profile = getenv("DMK_PBC_PROFILE");
    // Same-depth ContactGeometry cull threshold. -1 disables.
    const int filter_trg_min_ = []() {
        const char *s = std::getenv("DMK_FILTER_NTRG_MIN");
        return s ? std::atoi(s) : 64;
    }();

    // Per-target-box dump: one line per direct_work target box with its list1.
    // DMK_PBC_DUMP_LIST1=<path> or "1" (→ /tmp/pbc_list1.txt).
    std::FILE *dump_fp = nullptr;
    if (const char *path = std::getenv("DMK_PBC_DUMP_LIST1")) {
        const char *outpath = (path[0] == '1' && path[1] == 0) ? "/tmp/pbc_list1.txt" : path;
        dump_fp = std::fopen(outpath, "w");
    }

    // src_counts_owned aliases src_counts_with_halo (see pbc_tree.hpp); for
    // large image extents (nz>>1) per-box halo-inclusive counts can exceed
    // n_per_leaf, so size the per-thread filter scratch buffers to the actual
    // tree-wide maximum to avoid overflow into adjacent stack/heap.
    int max_box_count = params.n_per_leaf;
    for (int b = 0; b < nboxes_; ++b) {
        if (src_counts_owned[b] > max_box_count) max_box_count = src_counts_owned[b];
        if (trg_counts_owned[b] > max_box_count) max_box_count = trg_counts_owned[b];
    }

#pragma omp parallel
    {
        constexpr int MAX_PTS = 1000;
        util::StackOrHeapBuffer<Real, DIM * MAX_PTS> r_buf(DIM * max_box_count);
        util::StackOrHeapBuffer<Real, 3 * MAX_PTS> charge_buf(kernel_input_dim * max_box_count);
        util::StackOrHeapBuffer<Real, DIM * MAX_PTS> r_trg_buf(DIM * max_box_count);
        util::StackOrHeapBuffer<Real, 9 * MAX_PTS> pot_buf(kernel_output_dim_max * max_box_count);
        util::StackOrHeapBuffer<int, MAX_PTS> index_map(max_box_count);

        long n_pairs_local = 0, n_src_pairs_local = 0, n_trg_pairs_local = 0;
        long n_trg_pairs_actual_local = 0;
#pragma omp for schedule(dynamic)
        for (int idx = 0; idx < (int)direct_work.size(); ++idx) {
            const int trg_box = direct_work[idx];
            const int trg_level = box_depth[trg_box];

            long dump_sum_src = 0;
            for (auto src_box : list1(trg_box)) {
                if (debug_profile) {
                    n_pairs_local++;
                    if (kernel_output_dim_src > 0)
                        n_src_pairs_local += (long)src_counts_owned[src_box] * (long)src_counts_owned[trg_box];
                    if (kernel_output_dim_trg > 0)
                        n_trg_pairs_local += (long)src_counts_owned[src_box] * (long)trg_counts_owned[trg_box];
                }
                if (dump_fp) dump_sum_src += src_counts_owned[src_box];
                int src_level = box_depth[src_box];
                Real bsize = boxsize[src_level];

                if (ifpwexp[src_box] && src_box == trg_box) {
                    bsize /= Real{2.0};
                    src_level = src_level + 1;
                } else if (src_level < trg_level) {
                    bsize = boxsize[trg_level];
                    src_level = trg_level;
                }

                const Real d2max = bsize * bsize;
                const Real bsizeinv = Real{1} / bsize;

                Real rsc = 2 * bsizeinv;
                Real cen = -bsize / Real{2};

                if ((params.kernel == DMK_SQRT_LAPLACE && DIM == 3) || (params.kernel == DMK_LAPLACE && DIM == 2)) {
                    rsc = 2 * bsizeinv * bsizeinv;
                    cen = Real{-1.0};
                } else if (params.kernel == DMK_YUKAWA)
                    cen = Real{-1.0};

                // Determine filtering need
                const bool src_larger = box_depth[src_box] < box_depth[trg_box];
                const bool trg_larger = box_depth[src_box] > box_depth[trg_box];

                // Box corners for contact geometry
                std::array<Real, DIM> corner_a, corner_b;
                Real size_a = boxsize[box_depth[src_box]];
                Real size_b = boxsize[box_depth[trg_box]];
                for (int d = 0; d < DIM; d++) {
                    corner_a[d] = center_ptr(src_box)[d] - size_a / 2;
                    corner_b[d] = center_ptr(trg_box)[d] - size_b / 2;
                }

                int n_src = src_counts_owned[src_box];
                const Real *r_src_ptr = r_src_with_halo_ptr(src_box);
                const Real *charge_ptr = charge_with_halo_ptr(src_box);

                // Extend the ContactGeometry cull to same-depth pairs when
                // there are enough targets in the target box to amortize the
                // per-source geometry test. Match PeriodicDMK's threshold by
                // using only the target-side eval count: max of src_counts
                // (used when pgh_src>0) and trg_counts (pgh_trg>0). At low eps
                // with small leaves, this falls below 64 and the filter is
                // skipped — the SIMD kernel masks out-of-range pairs instead.
                const int n_eval_trg = std::max(
                    kernel_output_dim_src > 0 ? src_counts_owned[trg_box] : 0,
                    kernel_output_dim_trg > 0 ? trg_counts_owned[trg_box] : 0);
                const bool filter_same = (!src_larger && !trg_larger) &&
                                         (filter_trg_min_ >= 0 && n_eval_trg >= filter_trg_min_);
                if (src_larger || filter_same) {
                    ContactGeometry<Real, DIM> geom(corner_a.data(), corner_b.data(), size_a, size_b, d2max);
                    n_src = filter_sources(geom, n_src, r_src_ptr, charge_ptr, kernel_input_dim, r_buf.data(),
                                            charge_buf.data());
                    r_src_ptr = r_buf.data();
                    charge_ptr = charge_buf.data();
                }
                if (debug_profile && kernel_output_dim_trg > 0)
                    n_trg_pairs_actual_local += (long)n_src * (long)trg_counts_owned[trg_box];
                if (!n_src) continue;

                // Wrapper around the AOT evaluator that supports nd>1 (e.g. QP
                // Laplace treated as two independent real densities). The AOT
                // evaluator itself is nd=1; we stride-extract each density
                // from the interleaved `charge_local` buffer, call the kernel,
                // and accumulate into the correct slot of `pot_out`. nd=1
                // takes the direct path (no scratch, zero overhead).
                const int nd_eval = kernel_input_dim;
                auto call_eval_nd = [&](const direct_evaluator_func<Real> &evaluator,
                                        int n_src_local, const Real *r_src_local,
                                        const Real *charge_local, int n_eval_trg_local,
                                        const Real *r_trg_local, Real *pot_out,
                                        int k_out_dim) {
                    if (nd_eval == 1) {
                        evaluator(rsc, cen, d2max, 1e-30, n_src_local, r_src_local,
                                  charge_local, n_eval_trg_local, r_trg_local, pot_out);
                        return;
                    }
                    std::vector<Real> scratch_chg((std::size_t)n_src_local);
                    std::vector<Real> scratch_pot((std::size_t)n_eval_trg_local);
                    for (int k = 0; k < nd_eval; k++) {
                        for (int j = 0; j < n_src_local; j++)
                            scratch_chg[j] = charge_local[j * nd_eval + k];
                        std::memset(scratch_pot.data(), 0, n_eval_trg_local * sizeof(Real));
                        evaluator(rsc, cen, d2max, 1e-30, n_src_local, r_src_local,
                                  scratch_chg.data(), n_eval_trg_local, r_trg_local,
                                  scratch_pot.data());
                        for (int i = 0; i < n_eval_trg_local; i++)
                            pot_out[i * k_out_dim + k] += scratch_pot[i];
                    }
                };

                // Evaluate at owned source points in target box
                if (kernel_output_dim_src > 0 && src_counts_owned[trg_box]) {
                    int n_eval_trg = src_counts_owned[trg_box];
                    Real *eval_r_trg = r_src_owned_ptr(trg_box);
                    Real *eval_pot = pot_src_ptr(trg_box);

                    if (trg_larger) {
                        ContactGeometry<Real, DIM> geom(corner_a.data(), corner_b.data(), size_a, size_b, d2max);
                        n_eval_trg = filter_targets(geom, n_eval_trg, eval_r_trg, r_trg_buf.data(), index_map.data());
                        std::memset(pot_buf.data(), 0, n_eval_trg * kernel_output_dim_src * sizeof(Real));
                        eval_r_trg = r_trg_buf.data();
                        eval_pot = pot_buf.data();
                    }
                    if (n_eval_trg > 0) {
                        if (evaluator_by_level_src[src_level])
                            call_eval_nd(evaluator_by_level_src[src_level], n_src, r_src_ptr, charge_ptr,
                                         n_eval_trg, eval_r_trg, eval_pot, kernel_output_dim_src);
                        if (trg_larger)
                            scatter_add_potential(pot_buf.data(), pot_src_ptr(trg_box), index_map.data(), n_eval_trg,
                                                  kernel_output_dim_src);
                    }
                }

                // Evaluate at owned target points in target box
                if (kernel_output_dim_trg > 0 && trg_counts_owned[trg_box]) {
                    int n_eval_trg = trg_counts_owned[trg_box];
                    Real *eval_r_trg = r_trg_owned_ptr(trg_box);
                    Real *eval_pot = pot_trg_ptr(trg_box);

                    if (trg_larger) {
                        ContactGeometry<Real, DIM> geom(corner_a.data(), corner_b.data(), size_a, size_b, d2max);
                        n_eval_trg = filter_targets(geom, n_eval_trg, eval_r_trg, r_trg_buf.data(), index_map.data());
                        std::memset(pot_buf.data(), 0, n_eval_trg * kernel_output_dim_trg * sizeof(Real));
                        eval_r_trg = r_trg_buf.data();
                        eval_pot = pot_buf.data();
                    }
                    if (n_eval_trg > 0) {
                        call_eval_nd(evaluator_by_level_trg[src_level], n_src, r_src_ptr, charge_ptr,
                                     n_eval_trg, eval_r_trg, eval_pot, kernel_output_dim_trg);
                        if (trg_larger)
                            scatter_add_potential(pot_buf.data(), pot_trg_ptr(trg_box), index_map.data(), n_eval_trg,
                                                  kernel_output_dim_trg);
                    }
                }
            }

            if (dump_fp) {
                const Real *c = center_ptr(trg_box);
                #pragma omp critical
                std::fprintf(dump_fp, "%.6f %.6f %.6f %d %d %d %ld\n",
                             (double)c[0], (double)c[1], (double)c[2],
                             trg_level, trg_counts_owned[trg_box], nlist1_[trg_box], dump_sum_src);
            }

            // Source-side self-correction (skipped when pgh_src=0).
            if (kernel_output_dim_src > 0 && src_counts_owned[trg_box]) {
                auto pot = pot_src_view(trg_box);
                auto chg = charge_with_halo_view(trg_box);
                const int depth = box_depth[trg_box] + ifpwexp[trg_box];
                const auto correction_factor = w0[depth];
                for (int i_src = 0; i_src < src_counts_owned[trg_box]; ++i_src)
                    for (int i = 0; i < kernel_input_dim; ++i)
                        pot(i, i_src) -= correction_factor * chg(i, i_src);
            }
        }
        if (debug_profile) {
            n_pairs_tot += n_pairs_local;
            n_src_pairs_tot += n_src_pairs_local;
            n_trg_pairs_tot += n_trg_pairs_local;
            n_trg_pairs_actual += n_trg_pairs_actual_local;
        }
    }
    if (debug_profile) {
        long nominal = (long)n_trg_pairs_tot.load();
        long actual = (long)n_trg_pairs_actual.load();
        double cull = nominal > 0 ? 100.0 * (1.0 - (double)actual / (double)nominal) : 0.0;
        logger->warn("  [prof] direct: box_pairs={}  src_pairs={}  trg_pairs_nominal={}  trg_pairs_actual={}  (culled {:.1f}%)",
                     (long)n_pairs_tot.load(),
                     (long)n_src_pairs_tot.load(),
                     nominal, actual, cull);
    }
    if (dump_fp) std::fclose(dump_fp);
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::downward_prologue() {
    pot_src_sorted.SetZero();
    pot_trg_sorted.SetZero();
    init_planewave_data();
    std::fill(proxy_down_zeroed.begin(), proxy_down_zeroed.end(), 0);
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::form_downward_expansions() {
    bool debug_skip_pw = (getenv("DMK_PBC_SKIP_PW") != nullptr);
    bool debug_profile = (getenv("DMK_PBC_PROFILE") != nullptr);

    auto t0 = std::chrono::steady_clock::now();
    if (debug_profile) {
        form_eval_t_pw_ = form_eval_t_p2p_ = form_eval_t_p2c_ = form_eval_t_eval_ = 0;
        form_eval_n_shifts_ = 0;
    }
    if (!debug_skip_pw) {
        form_outgoing_expansions();
        auto t1 = std::chrono::steady_clock::now();
        const int n_pw = fourier_data.n_pw();
        for (int i_level = 0; i_level < n_levels(); ++i_level) {
            auto &dfd = difference_fourier_data[i_level];
            const ndview<std::complex<Real>, 2> pw2p({n_pw, n_order}, &dfd.pw2poly[0]);
            form_eval_expansions(level_indices[i_level], dfd.wpwshift, boxsize[i_level], pw2p, p2c);
        }
        auto t2 = std::chrono::steady_clock::now();
        if (debug_profile) {
            double t_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double t_eval = std::chrono::duration<double, std::milli>(t2 - t1).count();
            int nth = 1;
#ifdef _OPENMP
            nth = omp_get_max_threads();
#endif
            logger->warn("  [prof] form_outgoing={:.1f} form_eval_all={:.1f} ms  "
                         "[ pw_shift={:.1f} ({} calls)  pw_to_proxy={:.1f}  p2c={:.1f}  proxy_eval={:.1f} ] (omp_sum/{} threads)",
                         t_out, t_eval,
                         form_eval_t_pw_ / nth, form_eval_n_shifts_,
                         form_eval_t_p2p_ / nth,
                         form_eval_t_p2c_ / nth, form_eval_t_eval_ / nth, nth);
        }
    } else {
        logger->info("  SKIPPING planewave step (DMK_PBC_SKIP_PW)");
    }
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::downward_pass() {
    logger->info("downward pass started");
    downward_prologue();
    form_downward_expansions();

    bool debug_skip_direct = (getenv("DMK_PBC_SKIP_DIRECT") != nullptr);
    bool debug_profile = (getenv("DMK_PBC_PROFILE") != nullptr);
    auto t_pw = std::chrono::steady_clock::now();
    if (!debug_skip_direct)
        evaluate_direct_interactions();
    else
        logger->info("  SKIPPING direct step (DMK_PBC_SKIP_DIRECT)");
    auto t_dir = std::chrono::steady_clock::now();
    if (debug_profile) {
        double t_direct_ms = std::chrono::duration<double, std::milli>(t_dir - t_pw).count();
        logger->warn("  [prof] direct={:.1f} ms", t_direct_ms);
    }
    logger->info("downward pass completed");
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::eval() {
    logger->info("PBCTree::eval() started");
    upward_pass();
    downward_pass();
    logger->info("PBCTree::eval() completed");
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::apply_target_self_correction(const Real *trg_charges) {
    // Mirrors Fortran pbcdmk_self_corr_targ: pot_trg[i] -= w0(depth) * q[i].
    // Targets are assumed to coincide with the user's 1:1 source set.
    if (kernel_output_dim_trg == 0) return;

    Real w0[64];
    for (int i_level = 0; i_level < n_levels(); ++i_level)
        w0[i_level] = get_self_interaction_constant<Real, DIM>(fourier_data, params.kernel, i_level, boxsize[i_level]);
    if (n_levels() < 64) {
        Real bs_extra = boxsize[n_levels() - 1] / Real(2);
        w0[n_levels()] = get_self_interaction_constant<Real, DIM>(fourier_data, params.kernel, n_levels(), bs_extra);
    }

    // First sorted target index of each box, derived from the byte offset.
    for (int b = 0; b < nboxes_; ++b) {
        if (!is_global_leaf[b] || trg_counts_owned[b] == 0) continue;

        const int depth = box_depth[b] + ifpwexp[b];
        const Real sc = w0[depth];
        const sctl::Long t_start = r_trg_offsets_owned[b] / DIM;

        for (int i = 0; i < trg_counts_owned[b]; ++i) {
            const int sorted_idx = (int)(t_start + i);
            const int orig_idx = trg_perm[sorted_idx];
            for (int k = 0; k < kernel_input_dim; ++k)
                pot_trg_sorted[sorted_idx * kernel_output_dim_trg + k]
                    -= sc * trg_charges[orig_idx * kernel_input_dim + k];
        }
    }
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::desort_potentials(Real *pot_src, Real *pot_trg) {
    // Unsort source potentials
    for (int i = 0; i < (int)src_perm.size(); i++) {
        int orig = src_perm[i];
        for (int k = 0; k < kernel_output_dim_src; k++)
            pot_src[orig * kernel_output_dim_src + k] = pot_src_sorted[i * kernel_output_dim_src + k];
    }
    // Unsort target potentials
    for (int i = 0; i < (int)trg_perm.size(); i++) {
        int orig = trg_perm[i];
        for (int k = 0; k < kernel_output_dim_trg; k++)
            pot_trg[orig * kernel_output_dim_trg + k] = pot_trg_sorted[i * kernel_output_dim_trg + k];
    }
}

template <typename Real, int DIM>
void PBCTree<Real, DIM>::update_charges(const Real* charge_all_unsorted) {
    const int nd = kernel_input_dim;
    const sctl::Long nsall = src_perm.size();
    // src_perm is the sorted -> user map built during PBCTree construction
    // (see bin_sort call above; chg_sorted[i*nd+k] = chgall[src_perm[i]*nd+k]).
    for (sctl::Long i = 0; i < nsall; ++i) {
        const int j = src_perm[i];
        for (int k = 0; k < nd; ++k)
            charge_sorted_owned[i * nd + k] =
                charge_all_unsorted[(sctl::Long)j * nd + k];
    }
    for (auto& v : pot_src_sorted) v = 0;
    for (auto& v : pot_trg_sorted) v = 0;
}

// ================================================================
// Explicit instantiations
// ================================================================

template struct PBCTree<float, 2>;
template struct PBCTree<float, 3>;
template struct PBCTree<double, 2>;
template struct PBCTree<double, 3>;

} // namespace dmk
