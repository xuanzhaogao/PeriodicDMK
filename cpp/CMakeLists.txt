cmake_minimum_required(VERSION 3.18)
project(PeriodicDMK LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler flags ──────────────────────────────────────────────────────────
add_compile_options(-march=native)
add_compile_definitions(SCTL_MAX_DEPTH=62)
add_compile_definitions(SCTL_GLOBAL_MEM_BUFF=4096)

# Silence DMK-pbc's MPI path — we compile single-rank only.
add_compile_definitions(SCTL_HAVE_LIBMVEC)

# The vendored dmk.h declares convenience C functions `pdmk()` and `pdmkf()`
# at global scope, which clash with our `namespace pdmk { ... }`. We don't
# use those one-shot entry points; skip their declarations everywhere.
add_compile_definitions(DMK_H_SKIP_ONESHOT)

# ── SCTL (header-only, git submodule) ──────────────────────────────────────
set(SCTL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/SCTL/include")
if(NOT EXISTS "${SCTL_INCLUDE_DIR}/sctl.hpp")
    message(FATAL_ERROR "SCTL not found. Run: git submodule update --init")
endif()

# ── FINUFFT (preferred NUFFT backend) ──────────────────────────────────────
# FINUFFT is the runtime default when compiled in (handles anisotropic cells
# where DUCC trips the per-axis "oversampled grid too large" assertion).
# Turn on with -DPDMK_ENABLE_FINUFFT=ON (needs a prebuilt libfinufft.a under
# FINUFFT_ROOT/build). DUCC is the backup: it is always built and is used
# automatically when FINUFFT is not linked in, or selectable at runtime with
# PDMK_NUFFT_BACKEND=ducc.
option(PDMK_ENABLE_FINUFFT "Link FINUFFT as the primary NUFFT backend (DUCC is the backup)" OFF)
set(FINUFFT_ROOT "$ENV{HOME}/codes/finufft" CACHE PATH "Path to FINUFFT source tree (expects build/libfinufft.a and include/finufft.h)")
if(PDMK_ENABLE_FINUFFT)
    if(NOT EXISTS "${FINUFFT_ROOT}/build/libfinufft.a")
        message(FATAL_ERROR "PDMK_ENABLE_FINUFFT=ON but ${FINUFFT_ROOT}/build/libfinufft.a not found")
    endif()
    if(NOT EXISTS "${FINUFFT_ROOT}/include/finufft.h")
        message(FATAL_ERROR "PDMK_ENABLE_FINUFFT=ON but ${FINUFFT_ROOT}/include/finufft.h not found")
    endif()
    add_library(finufft STATIC IMPORTED)
    set_target_properties(finufft PROPERTIES
        IMPORTED_LOCATION "${FINUFFT_ROOT}/build/libfinufft.a")
    target_include_directories(finufft INTERFACE "${FINUFFT_ROOT}/include")
endif()

# ── System dependencies ─────────────────────────────────────────────────────
find_package(FFTW3 QUIET)
if(NOT TARGET FFTW3::fftw3)
    # Fall back to manual search for FFTW
    find_library(FFTW3_LIB NAMES fftw3)
    find_library(FFTW3F_LIB NAMES fftw3f)
    find_library(FFTW3_OMP_LIB NAMES fftw3_omp)
    find_library(FFTW3F_OMP_LIB NAMES fftw3f_omp)
    find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h)
    if(FFTW3_LIB)
        add_library(fftw3_ext INTERFACE)
        target_link_libraries(fftw3_ext INTERFACE ${FFTW3_LIB})
        if(FFTW3F_LIB)
            target_link_libraries(fftw3_ext INTERFACE ${FFTW3F_LIB})
        endif()
        if(FFTW3_OMP_LIB)
            target_link_libraries(fftw3_ext INTERFACE ${FFTW3_OMP_LIB})
        endif()
        if(FFTW3F_OMP_LIB)
            target_link_libraries(fftw3_ext INTERFACE ${FFTW3F_OMP_LIB})
        endif()
        if(FFTW3_INCLUDE_DIR)
            target_include_directories(fftw3_ext INTERFACE ${FFTW3_INCLUDE_DIR})
        endif()
    else()
        message(FATAL_ERROR "FFTW3 not found")
    endif()
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP)

# ── External fetched dependencies ──────────────────────────────────────────
include(FetchContent)

# polyfit (polynomial fitting for PSWF residual kernel)
FetchContent_Declare(
    polyfit
    GIT_REPOSITORY https://github.com/DiamonDinoia/polyfit.git
    GIT_TAG        a2994771155ed4cd22c2b792b463c42eefca3067
)
set(Build_Tests OFF CACHE BOOL "" FORCE)
set(MONOFIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MONOFIT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(polyfit)

# doctest (testing framework)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
)
FetchContent_MakeAvailable(doctest)

# DUCC NUFFT backend (download only — we compile a small subset below).
option(PDMK_ENABLE_FLOAT "Build single-precision instantiation" ON)
FetchContent_Declare(
    ducc
    GIT_REPOSITORY https://gitlab.mpcdf.mpg.de/mtr/ducc.git
    GIT_TAG        1da2c31dbcdc3a542ae3e9f723385acc8bcb9ec2
)
FetchContent_GetProperties(ducc)
if(NOT ducc_POPULATED)
    FetchContent_Populate(ducc)
endif()

# ── DMK-pbc vendored deps ───────────────────────────────────────────────────
# spdlog: header-only (use the header-only target to avoid a link dep)
add_library(spdlog_h INTERFACE)
target_include_directories(spdlog_h INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/spdlog/include)
target_compile_definitions(spdlog_h INTERFACE SPDLOG_HEADER_ONLY=1)

# itertools (required header-only dep of nda)
FetchContent_Declare(
    itertools
    GIT_REPOSITORY https://github.com/TRIQS/itertools.git
    GIT_TAG        unstable
)
FetchContent_GetProperties(itertools)
if(NOT itertools_POPULATED)
    FetchContent_Populate(itertools)
endif()
add_library(itertools_h INTERFACE)
target_include_directories(itertools_h INTERFACE
    ${itertools_SOURCE_DIR}/c++)

# nda: mostly header-only, but its BLAS/LAPACK Fortran wrappers live in
# two .cpp files we need to compile.
add_library(nda_blas_lapack STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/nda/c++/nda/blas/interface/cxx_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/nda/c++/nda/lapack/interface/cxx_interface.cpp)
set_target_properties(nda_blas_lapack PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nda_blas_lapack PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/nda/c++)
target_link_libraries(nda_blas_lapack PUBLIC itertools_h BLAS::BLAS LAPACK::LAPACK)

add_library(nda_h INTERFACE)
target_include_directories(nda_h INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/nda/c++)
target_link_libraries(nda_h INTERFACE itertools_h nda_blas_lapack)

# DMK-pbc pbc-path sources (vendored under src/dmk).
file(GLOB DMK_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/dmk/*.cpp)

# ── DUCC nufft backend ─────────────────────────────────────────────────────
# DUCC is not strictly header-only: a handful of .cc files provide threading,
# FFT, and NUFFT spreadinterp instantiations. We compile just those.
set(DUCC_SOURCES
    ${ducc_SOURCE_DIR}/src/ducc0/infra/threading.cc
    ${ducc_SOURCE_DIR}/src/ducc0/infra/mav.cc
    ${ducc_SOURCE_DIR}/src/ducc0/infra/string_utils.cc
    ${ducc_SOURCE_DIR}/src/ducc0/infra/types.cc
    ${ducc_SOURCE_DIR}/src/ducc0/infra/system.cc
    ${ducc_SOURCE_DIR}/src/ducc0/math/gridding_kernel.cc
    ${ducc_SOURCE_DIR}/src/ducc0/math/pointing.cc
    ${ducc_SOURCE_DIR}/src/ducc0/math/geom_utils.cc
    ${ducc_SOURCE_DIR}/src/ducc0/math/gl_integrator.cc
    ${ducc_SOURCE_DIR}/src/ducc0/math/space_filling.cc
    ${ducc_SOURCE_DIR}/src/ducc0/nufft/spreadinterp_inst1.cc
    ${ducc_SOURCE_DIR}/src/ducc0/nufft/spreadinterp_inst2.cc
    ${ducc_SOURCE_DIR}/src/ducc0/fft/fft_inst1.cc
    ${ducc_SOURCE_DIR}/src/ducc0/fft/fft_inst2.cc
)
add_library(ducc0_nufft STATIC ${DUCC_SOURCES})
target_include_directories(ducc0_nufft PUBLIC ${ducc_SOURCE_DIR}/src)
target_compile_features(ducc0_nufft PUBLIC cxx_std_17)
set_target_properties(ducc0_nufft PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ducc0_nufft PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── dmk_vendored: scratch static lib holding only the DMK-pbc sources ─────
# Used during Phase 1 port to verify the vendored pipeline compiles in
# isolation. Eventually folded into pdmk.
add_library(dmk_vendored STATIC ${DMK_SRC})
set_target_properties(dmk_vendored PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(dmk_vendored PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SCTL_INCLUDE_DIR})
target_link_libraries(dmk_vendored PUBLIC
    spdlog_h nda_h ducc0_nufft
    polyfit::polyfit
    doctest::doctest
    BLAS::BLAS LAPACK::LAPACK
    $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>)
if(TARGET FFTW3::fftw3)
    target_link_libraries(dmk_vendored PUBLIC FFTW3::fftw3)
else()
    target_link_libraries(dmk_vendored PUBLIC fftw3_ext)
endif()
if(PDMK_ENABLE_FINUFFT)
    target_link_libraries(dmk_vendored PUBLIC finufft)
    target_compile_definitions(dmk_vendored PUBLIC PDMK_HAVE_FINUFFT=1)
endif()

# ── PeriodicDMK library ────────────────────────────────────────────────────
# Phase 1: periodic_dmk.cpp and pdmk_capi.cpp are temporarily omitted because
# they still reference headers that were deleted when vendoring DMK-pbc.
# They are rewritten on top of the new pipeline in Phases 2/3.
set(PDMK_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ewald_nufft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cubic_cover.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cubic_cover_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lattice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/periodic_dmk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pdmk_capi.cpp
)

add_library(pdmk STATIC ${PDMK_SRC} ${DMK_SRC})
set_target_properties(pdmk PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Disable doctest macros inside pdmk's own TU compilations (headers like
# chebychev.hpp include <doctest/doctest.h> and expand TEST_CASE blocks at
# file scope when included by a non-test TU). Test executables define
# DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN themselves and thus re-enable macros.
target_compile_definitions(pdmk PRIVATE DOCTEST_CONFIG_DISABLE)
if(PDMK_ENABLE_FLOAT)
    target_compile_definitions(pdmk PUBLIC PDMK_ENABLE_FLOAT=1)
endif()
target_include_directories(pdmk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SCTL_INCLUDE_DIR})
target_link_libraries(pdmk PUBLIC
    spdlog_h nda_h ducc0_nufft
    polyfit::polyfit
    doctest::doctest
    BLAS::BLAS LAPACK::LAPACK
    $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
)
if(TARGET FFTW3::fftw3)
    target_link_libraries(pdmk PUBLIC FFTW3::fftw3)
else()
    target_link_libraries(pdmk PUBLIC fftw3_ext)
endif()
if(PDMK_ENABLE_FINUFFT)
    target_link_libraries(pdmk PUBLIC finufft)
    target_compile_definitions(pdmk PUBLIC PDMK_HAVE_FINUFFT=1)
endif()

add_library(pdmk_jl SHARED ${PDMK_SRC} ${DMK_SRC})
set_target_properties(pdmk_jl PROPERTIES
    OUTPUT_NAME pdmk_jl
    POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(pdmk_jl PRIVATE DOCTEST_CONFIG_DISABLE)
if(PDMK_ENABLE_FLOAT)
    target_compile_definitions(pdmk_jl PUBLIC PDMK_ENABLE_FLOAT=1)
endif()
target_include_directories(pdmk_jl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SCTL_INCLUDE_DIR})
target_link_libraries(pdmk_jl PUBLIC
    spdlog_h nda_h ducc0_nufft
    polyfit::polyfit
    doctest::doctest
    BLAS::BLAS LAPACK::LAPACK
    $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
)
if(TARGET FFTW3::fftw3)
    target_link_libraries(pdmk_jl PUBLIC FFTW3::fftw3)
else()
    target_link_libraries(pdmk_jl PUBLIC fftw3_ext)
endif()
if(PDMK_ENABLE_FINUFFT)
    target_link_libraries(pdmk_jl PUBLIC finufft)
    target_compile_definitions(pdmk_jl PUBLIC PDMK_HAVE_FINUFFT=1)
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()

# Phase 1 smoke test: validates the vendored DMK-pbc pipeline compiles and
# produces finite output.
add_executable(test_dmk_pbc_smoke tests/test_dmk_pbc_smoke.cpp)
target_link_libraries(test_dmk_pbc_smoke PRIVATE pdmk doctest::doctest)
add_test(NAME test_dmk_pbc_smoke COMMAND test_dmk_pbc_smoke)

# Phase 2.1: PBCTree::update_charges reusable-evaluation hook.
add_executable(test_pbc_tree_update_charges tests/test_pbc_tree_update_charges.cpp)
target_link_libraries(test_pbc_tree_update_charges PRIVATE pdmk doctest::doctest)
add_test(NAME test_pbc_tree_update_charges COMMAND test_pbc_tree_update_charges)

# Phase 2.2: image_layout helper (source-index-tracking image enumeration).
add_executable(test_image_layout tests/test_image_layout.cpp)
target_link_libraries(test_image_layout PRIVATE pdmk doctest::doctest)
add_test(NAME test_image_layout COMMAND test_image_layout)

add_executable(test_image_layout_2d tests/test_image_layout_2d.cpp)
target_link_libraries(test_image_layout_2d PRIVATE pdmk doctest::doctest)
add_test(NAME test_image_layout_2d COMMAND test_image_layout_2d)

add_executable(test_far_field_nufft_2d tests/test_far_field_nufft_2d.cpp)
target_link_libraries(test_far_field_nufft_2d PRIVATE pdmk doctest::doctest)
add_test(NAME test_far_field_nufft_2d COMMAND test_far_field_nufft_2d)

add_executable(test_periodic_dmk_2d tests/test_periodic_dmk_2d.cpp)
target_link_libraries(test_periodic_dmk_2d PRIVATE pdmk doctest::doctest)
add_test(NAME test_periodic_dmk_2d COMMAND test_periodic_dmk_2d)

add_executable(test_ewald_nufft_2d tests/test_ewald_nufft_2d.cpp)
target_link_libraries(test_ewald_nufft_2d PRIVATE pdmk doctest::doctest)
add_test(NAME test_ewald_nufft_2d COMMAND test_ewald_nufft_2d)

add_executable(test_lattice_setup_qmat tests/test_lattice_setup_qmat.cpp)
target_link_libraries(test_lattice_setup_qmat PRIVATE pdmk doctest::doctest)
add_test(NAME test_lattice_setup_qmat COMMAND test_lattice_setup_qmat)

add_executable(test_cell_list_2d tests/test_cell_list_2d.cpp)
target_link_libraries(test_cell_list_2d PRIVATE pdmk doctest::doctest)
add_test(NAME test_cell_list_2d COMMAND test_cell_list_2d)

# Phase 2.3: PeriodicDMKTreeT rewired on PBCTree (round-trip vs ewald_nufft).
add_executable(test_periodic_dmk_tree tests/test_periodic_dmk_tree.cpp)
target_link_libraries(test_periodic_dmk_tree PRIVATE pdmk doctest::doctest)
add_test(NAME test_periodic_dmk_tree COMMAND test_periodic_dmk_tree)

# Public-API kept tests.
add_executable(test_periodic_dmk tests/test_periodic_dmk.cpp)
target_link_libraries(test_periodic_dmk PRIVATE pdmk doctest::doctest)
add_test(NAME test_periodic_dmk COMMAND test_periodic_dmk)

add_executable(test_reusable_tree tests/test_reusable_tree.cpp)
target_link_libraries(test_reusable_tree PRIVATE pdmk doctest::doctest)
add_test(NAME test_reusable_tree COMMAND test_reusable_tree)

add_executable(test_nonuniform tests/test_nonuniform.cpp)
target_link_libraries(test_nonuniform PRIVATE pdmk doctest::doctest)
add_test(NAME test_nonuniform COMMAND test_nonuniform)

add_executable(test_ewald_nufft tests/test_ewald_nufft.cpp)
target_link_libraries(test_ewald_nufft PRIVATE pdmk doctest::doctest)
add_test(NAME test_ewald_nufft COMMAND test_ewald_nufft)

add_executable(test_cubic_cover tests/test_cubic_cover.cpp)
target_link_libraries(test_cubic_cover PRIVATE pdmk doctest::doctest)
add_test(NAME test_cubic_cover COMMAND test_cubic_cover)

add_executable(test_cubic_cover_optimizer tests/test_cubic_cover_optimizer.cpp)
target_link_libraries(test_cubic_cover_optimizer PRIVATE pdmk doctest::doctest)
add_test(NAME test_cubic_cover_optimizer COMMAND test_cubic_cover_optimizer)

add_executable(test_lattice tests/test_lattice.cpp)
target_link_libraries(test_lattice PRIVATE pdmk doctest::doctest)
add_test(NAME test_lattice COMMAND test_lattice)

add_executable(test_evaluate_timed tests/test_evaluate_timed.cpp)
target_link_libraries(test_evaluate_timed PRIVATE pdmk doctest::doctest)
add_test(NAME test_evaluate_timed COMMAND test_evaluate_timed)

add_executable(test_campaign_ref_io tests/test_campaign_ref_io.cpp)
target_link_libraries(test_campaign_ref_io PRIVATE pdmk doctest::doctest)
add_test(NAME test_campaign_ref_io COMMAND test_campaign_ref_io)

add_executable(test_periodic_dmk_qp tests/test_periodic_dmk_qp.cpp)
target_link_libraries(test_periodic_dmk_qp PRIVATE pdmk doctest::doctest)
add_test(NAME test_periodic_dmk_qp COMMAND test_periodic_dmk_qp)
