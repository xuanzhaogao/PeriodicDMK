#pragma once

#include <pdmk/types.hpp>
#include <pdmk/lattice.hpp>
#include <pdmk/eval_timing.hpp>
#include <array>
#include <complex>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace dmk { template <typename Real, int DIM> struct PBCTree; }

namespace pdmk {

template <typename Real = double, int DIM = 3>
class PeriodicDMKTreeT {
public:
    using scalar_type = Real;

    PeriodicDMKTreeT(const MatND<DIM>& cell_matrix,
                      const std::vector<VecND<DIM>>& positions,
                      int n_digits = 6,
                      int n_per_leaf = 300,
                      int nz = 4,
                      std::optional<std::array<double, DIM>> bloch = std::nullopt);

    ~PeriodicDMKTreeT();

    PeriodicDMKTreeT(const PeriodicDMKTreeT&) = delete;
    PeriodicDMKTreeT& operator=(const PeriodicDMKTreeT&) = delete;
    PeriodicDMKTreeT(PeriodicDMKTreeT&&) = delete;
    PeriodicDMKTreeT& operator=(PeriodicDMKTreeT&&) = delete;

    /// Evaluate periodic Coulomb potentials for `charges` at the stored
    /// source positions. Must be called from the same thread as prior
    /// `evaluate()` calls for NUFFT plan reuse.
    std::vector<Real> evaluate(const std::vector<double>& charges);

    /// Same as evaluate(), but reports a three-way wall-time breakdown
    /// in `timing`. Must be called from the same OS thread as any prior
    /// evaluate()/evaluate_timed() calls — the NUFFT plan cache is
    /// thread_local.
    std::vector<Real> evaluate_timed(const std::vector<double>& charges,
                                      EvalTiming& timing);

    /// Quasi-periodic evaluation (3D only, requires bloch set at construction).
    /// Returns complex potentials of size n_src.
    std::vector<std::complex<Real>> evaluate_complex(const std::vector<double>& charges);

    int n_points() const { return n_src_; }
    double cube_size() const { return rc_; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    int n_src_;
    double rc_;
};

using PeriodicDMKTree = PeriodicDMKTreeT<double, 3>;

// ── Auto (float/double selection) ──────────────────────────────────────────
#if PDMK_ENABLE_FLOAT
using AnyPDMKTree = std::variant<std::unique_ptr<PeriodicDMKTreeT<float, 3>>,
                                  std::unique_ptr<PeriodicDMKTreeT<double, 3>>>;
#else
using AnyPDMKTree = std::variant<std::unique_ptr<PeriodicDMKTreeT<double, 3>>>;
#endif

#if PDMK_ENABLE_FLOAT
using AnyPDMKTree2D = std::variant<std::unique_ptr<PeriodicDMKTreeT<float, 2>>,
                                    std::unique_ptr<PeriodicDMKTreeT<double, 2>>>;
#else
using AnyPDMKTree2D = std::variant<std::unique_ptr<PeriodicDMKTreeT<double, 2>>>;
#endif

class AutoPeriodicDMKTree {
public:
    AutoPeriodicDMKTree(const Mat3& cell,
                         const std::vector<Vec3>& positions,
                         int n_digits,
                         int n_per_leaf = 300,
                         int nz = 4);

    AutoPeriodicDMKTree(const AutoPeriodicDMKTree&) = delete;
    AutoPeriodicDMKTree& operator=(const AutoPeriodicDMKTree&) = delete;
    AutoPeriodicDMKTree(AutoPeriodicDMKTree&&) = default;
    AutoPeriodicDMKTree& operator=(AutoPeriodicDMKTree&&) = default;

    bool is_single_precision() const;

    std::variant<std::vector<float>, std::vector<double>>
    evaluate(const std::vector<double>& charges);

private:
    AnyPDMKTree impl_;
    bool single_precision_;
};

AutoPeriodicDMKTree make_periodic_dmk_tree(const Mat3& cell,
                                            const std::vector<Vec3>& positions,
                                            int n_digits,
                                            int n_per_leaf = 300,
                                            int nz = 4);

class AutoPeriodicDMKTree2D {
public:
    AutoPeriodicDMKTree2D(const Mat2& cell,
                          const std::vector<Vec2>& positions,
                          int n_digits,
                          int n_per_leaf = 300,
                          int nz = 4);

    AutoPeriodicDMKTree2D(const AutoPeriodicDMKTree2D&) = delete;
    AutoPeriodicDMKTree2D& operator=(const AutoPeriodicDMKTree2D&) = delete;
    AutoPeriodicDMKTree2D(AutoPeriodicDMKTree2D&&) = default;
    AutoPeriodicDMKTree2D& operator=(AutoPeriodicDMKTree2D&&) = default;

    bool is_single_precision() const;

    std::variant<std::vector<float>, std::vector<double>>
    evaluate(const std::vector<double>& charges);

private:
    AnyPDMKTree2D impl_;
    bool single_precision_;
};

AutoPeriodicDMKTree2D make_periodic_dmk_tree_2d(const Mat2& cell,
                                                 const std::vector<Vec2>& positions,
                                                 int n_digits,
                                                 int n_per_leaf = 300,
                                                 int nz = 4);

extern template class PeriodicDMKTreeT<double, 3>;
extern template class PeriodicDMKTreeT<double, 2>;
#if PDMK_ENABLE_FLOAT
extern template class PeriodicDMKTreeT<float, 3>;
extern template class PeriodicDMKTreeT<float, 2>;
#endif

} // namespace pdmk
