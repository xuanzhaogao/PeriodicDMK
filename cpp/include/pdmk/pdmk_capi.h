/// C API for PeriodicDMK — thin wrappers for Julia/Python FFI.
///
/// Conventions:
///   - cell_3x3: column-major 3×3 (cell[:,j] = lattice vector j)
///   - positions_3xN: column-major 3×N (positions[:,i] = particle i)
///   - Return 0 on success, nonzero on error (for int-returning functions).

#ifndef PDMK_CAPI_H
#define PDMK_CAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Ewald NUFFT reference ───────────────────────────────────────────────────

/// O(N log N) Ewald summation. potentials_out is length N, energy_out is scalar.
int pdmk_ewald_nufft(const double* cell_3x3, int N,
                      const double* positions_3xN,
                      const double* charges,
                      double s,
                      double* potentials_out,
                      double* energy_out);

/// Optimize the cubic cover for a 3×3 cell.
/// Outputs: m/n/p integers, e1/e2/e3 each length-3 vectors, s and cost scalars.
/// Returns 0 on success, nonzero on error.
int pdmk_optimize_cubic_cover(const double* cell_3x3, int nz,
                              int* m_out, int* n_out, int* p_out,
                              double* e1_out, double* e2_out, double* e3_out,
                              double* s_out, double* cost_out);

// ── Reusable-tree API ──────────────────────────────────────────────────────

void* pdmk_create_double(const double* cell_3x3, int N,
                          const double* positions_3xN,
                          int n_digits, int n_per_leaf, int nz);
int   pdmk_evaluate_double(void* h, int N,
                            const double* positions_3xN, const double* charges,
                            double* pot_out);
void  pdmk_destroy_double(void* h);

void* pdmk_create_float(const double* cell_3x3, int N,
                         const double* positions_3xN,
                         int n_digits, int n_per_leaf, int nz);
int   pdmk_evaluate_float(void* h, int N,
                           const double* positions_3xN, const double* charges,
                           float* pot_out);
void  pdmk_destroy_float(void* h);

#ifdef __cplusplus
}
#endif

#endif // PDMK_CAPI_H
