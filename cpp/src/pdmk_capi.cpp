/// C API implementation for PeriodicDMK.

#include <pdmk/pdmk_capi.h>
#include <pdmk/periodic_dmk.hpp>
#include <pdmk/ewald_nufft.hpp>
#include <pdmk/cubic_cover_optimizer.hpp>
#include <pdmk/types.hpp>
#include <pdmk/lattice.hpp>

#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

using pdmk::PeriodicDMKTreeT;
using pdmk::Mat3;
using pdmk::Vec3;
using pdmk::Lattice;

namespace {

/// Column-major 3×3 → Mat3.
///
/// Every `Mat3` in pdmk (`Lattice`, `PeriodicDMKTreeT`, …) now follows the
/// column-major `h[col][row]` convention documented in
/// `include/pdmk/lattice.hpp`: `h[j][i]` is component `i` of lattice vector
/// `j`. Julia matrices are column-major in memory, so Julia element
/// `(i,j)` lives at `cell_3x3[i + 3*j]` — a direct match.
Mat3 to_mat3(const double* cell_3x3) {
    Mat3 h;
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            h[j][i] = cell_3x3[i + 3 * j];
    return h;
}

/// Column-major 3×N → vector<Vec3>: positions_3xN[d + 3*i] = coord d of particle i.
std::vector<Vec3> to_positions(const double* pos_3xN, int N) {
    std::vector<Vec3> positions(N);
    for (int i = 0; i < N; ++i) {
        positions[i][0] = pos_3xN[0 + 3 * i];
        positions[i][1] = pos_3xN[1 + 3 * i];
        positions[i][2] = pos_3xN[2 + 3 * i];
    }
    return positions;
}

template <typename Real>
void* create_impl(const double* cell_3x3, int N, const double* positions_3xN,
                   int n_digits, int n_per_leaf, int nz) {
    try {
        Mat3 h = to_mat3(cell_3x3);
        auto pos = to_positions(positions_3xN, N);
        return new PeriodicDMKTreeT<Real>(h, pos, n_digits, n_per_leaf, nz);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pdmk_create: %s\n", e.what());
        return nullptr;
    }
}

template <typename Real, typename Out>
int evaluate_impl(void* handle, int /*N_in*/,
                   const double* /*positions_unused*/, const double* charges,
                   Out* pot_out) {
    if (!handle) return -1;
    try {
        auto* tree = static_cast<PeriodicDMKTreeT<Real>*>(handle);
        const int N = tree->n_points();
        std::vector<double> q(charges, charges + N);
        auto pot = tree->evaluate(q);
        for (int i = 0; i < N; ++i) pot_out[i] = static_cast<Out>(pot[i]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pdmk_evaluate: %s\n", e.what());
        return -1;
    }
}

template <typename Real>
void destroy_impl(void* handle) {
    delete static_cast<PeriodicDMKTreeT<Real>*>(handle);
}

} // namespace

extern "C" {

int pdmk_ewald_nufft(const double* cell_3x3, int N,
                      const double* positions_3xN,
                      const double* charges,
                      double s,
                      double* potentials_out,
                      double* energy_out) {
    try {
        Mat3 h = to_mat3(cell_3x3);
        Lattice lat(h);
        auto positions = to_positions(positions_3xN, N);
        std::vector<double> q(charges, charges + N);
        auto result = pdmk::ewald_nufft(lat, positions, q, s);
        std::memcpy(potentials_out, result.potentials.data(), N * sizeof(double));
        if (energy_out) *energy_out = result.energy;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pdmk_ewald_nufft: %s\n", e.what());
        return -1;
    }
}

int pdmk_optimize_cubic_cover(const double* cell_3x3, int nz,
                               int* m_out, int* n_out, int* p_out,
                               double* e1_out, double* e2_out, double* e3_out,
                               double* s_out, double* cost_out) {
    try {
        if (nz <= 0) return 1;
        Mat3 h = to_mat3(cell_3x3);
        Lattice lat(h);
        auto res = pdmk::optimize_cubic_cover(lat, nz);
        *m_out = res.m; *n_out = res.n; *p_out = res.p;
        for (int i = 0; i < 3; ++i) {
            e1_out[i] = res.e1[i];
            e2_out[i] = res.e2[i];
            e3_out[i] = res.e3[i];
        }
        *s_out = res.s;
        *cost_out = res.cost;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pdmk_optimize_cubic_cover: %s\n", e.what());
        return -1;
    }
}

// ── Double ─────────────────────────────────────────────────────────────────

void* pdmk_create_double(const double* cell_3x3, int N,
                          const double* positions_3xN,
                          int n_digits, int n_per_leaf, int nz) {
    return create_impl<double>(cell_3x3, N, positions_3xN, n_digits, n_per_leaf, nz);
}

int pdmk_evaluate_double(void* h, int N,
                          const double* positions_3xN, const double* charges,
                          double* pot_out) {
    return evaluate_impl<double, double>(h, N, positions_3xN, charges, pot_out);
}

void pdmk_destroy_double(void* h) { destroy_impl<double>(h); }

// ── Float ──────────────────────────────────────────────────────────────────
#if PDMK_ENABLE_FLOAT
void* pdmk_create_float(const double* cell_3x3, int N,
                         const double* positions_3xN,
                         int n_digits, int n_per_leaf, int nz) {
    return create_impl<float>(cell_3x3, N, positions_3xN, n_digits, n_per_leaf, nz);
}

int pdmk_evaluate_float(void* h, int N,
                         const double* positions_3xN, const double* charges,
                         float* pot_out) {
    return evaluate_impl<float, float>(h, N, positions_3xN, charges, pot_out);
}

void pdmk_destroy_float(void* h) { destroy_impl<float>(h); }
#endif

} // extern "C"
