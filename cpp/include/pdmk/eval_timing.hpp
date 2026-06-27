// cpp/include/pdmk/eval_timing.hpp
#pragma once

namespace pdmk {

/// Per-evaluation wall-time breakdown reported by
/// PeriodicDMKTreeT::evaluate_timed(). By construction,
/// nufft_ms + dmk_ms + direct_ms equals the total wall time of
/// evaluate_timed() within steady-clock resolution.
struct EvalTiming {
    double nufft_ms  = 0.0;   ///< far_field_nufft call
    double dmk_ms    = 0.0;   ///< 3D: update_charges + upward_pass + downward_prologue
                              ///<     + form_downward_expansions (proxy/MLI work only)
                              ///< 2D: same — upward_pass + downward_prologue
                              ///<     + form_downward_expansions (proxy/MLI work only)
    double direct_ms = 0.0;   ///< Both 2D and 3D: evaluate_direct_interactions
                              ///<     + apply_target_self_correction + desort_potentials
    double build_ms  = 0.0;   ///< 2D only: image generation + PBCTree construction
                              ///< (per-eval cost; 0 for 3D since the tree is reused)
};

} // namespace pdmk
