# PeriodicDMK

Fast periodic Coulomb solver using dual-space multilevel kernel (DMK)
splitting with Prolate Spheroidal Wave Functions (PSWFs). Computes

```
u(x_i) = sum'_{j,n} q_j / |x_i - x_j - R_n|
```

for `N` charges in an arbitrary (triclinic) periodic cell, where `R_n` are
the lattice translations. The pipeline splits the `1/r` kernel into a
compactly-supported near-field residual (evaluated over a rectangular cover
and its colleague neighbourhood) and a smooth windowed kernel applied via a
NUFFT far field (DUCC backend), plus a PSWF self-correction. The solver is
exposed as a build-once / evaluate-many tree with a C++ and a Julia API.

Main contributors: Shidong Jiang (sjiang@flatironinstitute.org) and Xuanzhao Gao (xgao@flatironinstitute.org).

For benchmark scripts and data, see https://github.com/xuanzhaogao/PeriodicDMK_Benchmark/.

## Layout

```
cpp/
  include/pdmk/   Public C++ API (periodic_dmk.hpp, pdmk_capi.h, ...)
  include/dmk/    DMK-pbc pipeline headers
  src/            Library sources (periodic_dmk.cpp, pdmk_capi.cpp, dmk/*)
  tests/          doctest test suite (20 tests)
  extern/         Vendored deps: SCTL, nda, spdlog (header-only)
  CMakeLists.txt
julia/
  PeriodicDMK/    Julia package (ccall bindings over libpdmk_jl)
```

## Dependencies

- C++20 compiler (GCC 13+ recommended)
- CMake ≥ 3.20, BLAS/LAPACK (OpenBLAS), FFTW3, OpenMP
- Vendored header-only libs under `cpp/extern/`: **SCTL**, **nda**, **spdlog**
- Auto-fetched at configure time via CMake `FetchContent` (needs network):
  **doctest**, **DUCC** (NUFFT + Ewald reference), **itertools**
- Julia ≥ 1.9 (for the Julia interface)

## Build & test — C++

On the Flatiron cluster the module set is captured in `setenv.sh`
(gcc/14, cmake, openblas/single, flexiblas, openmpi, …):

```bash
source setenv.sh                 # or: module load gcc cmake openblas fftw openmpi

cd cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPDMK_ENABLE_FLOAT=1
cmake --build . -j
ctest --output-on-failure        # 20 tests
```

`-DPDMK_ENABLE_FLOAT=1` builds the single-precision instantiation (used for
`n_digits ≤ 6`); omit it for a double-only build.

### Compile flags

Most flags are baked into `cpp/CMakeLists.txt`, not passed on the command
line. The effective compile line for the library is approximately:

```
-std=c++20 -march=native -O3 -DNDEBUG -fopenmp \
  -DSCTL_MAX_DEPTH=62 -DSCTL_GLOBAL_MEM_BUFF=4096 -DSCTL_HAVE_LIBMVEC \
  -DDMK_H_SKIP_ONESHOT -DPDMK_ENABLE_FLOAT=1
```

| Flag | Source | Purpose |
|------|--------|---------|
| `-std=c++20` | `CMAKE_CXX_STANDARD 20`, extensions off | Strict C++20 (not `gnu++20`). |
| `-march=native` | unconditional `add_compile_options` | Tunes to the **build host** ISA (AVX2/AVX-512). Best perf, but see portability note below. |
| `-O3 -DNDEBUG` | `CMAKE_BUILD_TYPE=Release` | Full opt, asserts off. **No default build type** — without it the build is unoptimized. |
| `-fopenmp` | `find_package(OpenMP)` | Pipeline is OpenMP-parallel; threads via `OMP_NUM_THREADS`. |
| `-DSCTL_MAX_DEPTH=62` | global | SCTL octree max depth. |
| `-DSCTL_GLOBAL_MEM_BUFF=4096` | global | SCTL global scratch buffer (MB). |
| `-DSCTL_HAVE_LIBMVEC` | global | Use glibc `libmvec` vectorized transcendentals in SCTL. |
| `-DDMK_H_SKIP_ONESHOT` | global | Skip vendored global `pdmk()` decls that clash with `namespace pdmk`. |
| `-DPDMK_ENABLE_FLOAT=1` | option | Build the `float` instantiation. |

**Portability of `-march=native`.** The binary is tuned to the CPU it was
built on; running it on a node with a narrower instruction set (e.g. an older
partition without AVX-512) can fault with `SIGILL`. Build where you run, or
pin a portable baseline by editing the `add_compile_options(-march=native)`
line — `-march=x86-64-v4` (portable AVX-512) or `-march=x86-64-v3` (AVX2).
Note this can't be overridden via `-DCMAKE_CXX_FLAGS`: `add_compile_options`
is appended after `CMAKE_CXX_FLAGS` and the last `-march` wins.

**Threading.** The library is OpenMP-parallel *and* calls BLAS — use a
single-threaded BLAS (the `openblas/single` module, or `OPENBLAS_NUM_THREADS=1`)
so the two don't oversubscribe cores. For profiling, use
`-DCMAKE_BUILD_TYPE=RelWithDebInfo` (`-O2 -g`) to keep debug symbols.

### Optional: FINUFFT backend

The NUFFT far field uses DUCC by default. To link FINUFFT as the primary
backend instead (DUCC stays as the fallback), configure with
`-DPDMK_ENABLE_FINUFFT=ON` and point `-DFINUFFT_ROOT=<path>` at a tree
containing `build/libfinufft.a` and `include/finufft.h`.

## Build & test — Julia

```bash
cd julia/PeriodicDMK
julia --project=. -e 'using Pkg; Pkg.build()'   # builds libpdmk_jl via cmake
julia --project=. -e 'using Pkg; Pkg.test()'    # 20 tests
```

`Pkg.build()` invokes CMake to compile `libpdmk_jl` and writes `deps/deps.jl`
pointing the package at the shared library.

## C++ interface

```cpp
#include <pdmk/periodic_dmk.hpp>
using namespace pdmk;

// cell is column-major: columns are the lattice vectors a1 | a2 | a3
Mat3 h = {{{L, 0, 0}, {0, L, 0}, {0, 0, L}}};

// Positions are baked into the tree at construction (build once).
PeriodicDMKTreeT<double> solver(h, positions, /*n_digits=*/6);

// Re-evaluate with new charges on the same tree (evaluate many):
std::vector<double> u = solver.evaluate(charges);   // size N
```

`PeriodicDMKTreeT<Real, DIM>` (DIM = 2 or 3) constructor arguments:

| Argument | Meaning |
|----------|---------|
| `cell_matrix` | `MatND<DIM>`, columns = lattice vectors |
| `positions` | `std::vector<VecND<DIM>>` source coordinates |
| `n_digits` | requested accuracy: 3, 6, 9, or 12 (default 6) |
| `n_per_leaf` | target points per leaf box (default 300) |
| `nz` | cube layers along the shortest lattice direction (default 4) |
| `bloch` | optional Bloch phase for quasi-periodic `evaluate_complex` |

Key methods: `evaluate(charges)`, `evaluate_timed(charges, EvalTiming&)`,
`evaluate_complex(charges)` (quasi-periodic). For runtime precision selection
use `make_periodic_dmk_tree(cell, positions, n_digits)` (and the `_2d` variant),
which returns an `AutoPeriodicDMKTree` that picks float vs. double automatically.
A C ABI is also available in `pdmk/pdmk_capi.h`.

## Julia interface

```julia
using PeriodicDMK

cell = L * [1.0 0 0; 0 1 0; 0 0 1]   # 3×3, columns = lattice vectors
positions = rand(3, N)               # 3×N Cartesian coordinates

solver = PDMKSolver(cell, positions; n_digits=6)   # tree built once
u = evaluate(solver, charges)                       # charges length N

# O(N log N) Ewald reference (per-particle potentials + total energy):
u_ref, energy = ewald_nufft(cell, positions, charges)
```

`PDMKSolver(cell, positions; n_digits=6, n_per_leaf=300, nz=4, precision=:auto)`
builds the tree once; `evaluate(solver, charges)` reuses it. `precision` is
`:auto` (float for `n_digits ≤ 6`), `:float`, or `:double`. The standalone
cover optimizer `optimize_cubic_cover(cell; nz=4)` is also exported.
