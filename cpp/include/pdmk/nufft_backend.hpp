// pdmk/nufft_backend.hpp
//
// Runtime selection of NUFFT backend (DUCC vs FINUFFT) used by the
// periodic DMK NUFFT path (pbc.cpp) and the Ewald reference NUFFT path
// (ewald_nufft.cpp).
//
// Default: Finufft when PDMK_HAVE_FINUFFT is set at compile time, else
// Ducc. Override at runtime with PDMK_NUFFT_BACKEND={ducc,finufft}.

#pragma once

#include <cstdlib>
#include <cstring>

namespace pdmk {

enum class NufftBackend { Ducc, Finufft };

inline NufftBackend pick_nufft_backend() {
#if PDMK_HAVE_FINUFFT
    NufftBackend def = NufftBackend::Finufft;
#else
    NufftBackend def = NufftBackend::Ducc;
#endif
    if (const char *e = std::getenv("PDMK_NUFFT_BACKEND")) {
        if (std::strcmp(e, "ducc") == 0)    return NufftBackend::Ducc;
        if (std::strcmp(e, "finufft") == 0) return NufftBackend::Finufft;
    }
    return def;
}

} // namespace pdmk
