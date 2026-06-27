#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace pdmk {

/// Like dmk::pbc::generate_images but also records the originating
/// source index for each emitted image. Non-QP only.
///
/// Mirrors the loop order of pbc.hpp::generate_images so that
/// (srcimg_out, img_src_idx_out) align with DMK-pbc's image layout.
template <typename Real, int DIM>
inline int generate_images_with_src_idx(
        const Real* umat, Real rc, const int* mgrid, const Real* xgrid0,
        int ns, const Real* sources,
        std::vector<Real>& srcimg_out, std::vector<int>& img_src_idx_out)
{
    Real xextlo[3], xexthi[3], xlo[3], xhi[3];
    for (int i = 0; i < DIM; ++i) {
        xextlo[i] = xgrid0[i] - rc;
        xexthi[i] = xgrid0[i] + (mgrid[i] + 1) * rc;
        xlo[i] = xgrid0[i];
        Real w = umat[i + i * DIM];
        for (int j = i + 1; j < DIM; ++j) w += std::abs(umat[i + j * DIM]);
        xhi[i] = xgrid0[i] + w;
    }

    srcimg_out.clear();
    img_src_idx_out.clear();

    if constexpr (DIM == 2) {
        int n2max = (int)((mgrid[1] + 1) * rc / umat[1 + DIM]) + 1;
        for (int n2 = -n2max; n2 <= n2max; ++n2) {
            Real shift2 = umat[1 + DIM] * n2;
            if (xlo[1] + shift2 > xexthi[1] || xhi[1] + shift2 < xextlo[1]) continue;
            Real off12 = umat[DIM] * n2;
            int n1max = (int)(((mgrid[0] + 1) * rc + std::abs(off12)) / umat[0]) + 1;
            for (int n1 = -n1max; n1 <= n1max; ++n1) {
                if (n1 == 0 && n2 == 0) continue;
                Real shift1 = umat[0] * n1 + off12;
                if (xlo[0] + shift1 > xexthi[0] || xhi[0] + shift1 < xextlo[0]) continue;
                for (int j = 0; j < ns; ++j) {
                    Real x0 = sources[j * DIM] + shift1;
                    Real x1 = sources[j * DIM + 1] + shift2;
                    if (x0 < xextlo[0] || x0 > xexthi[0] ||
                        x1 < xextlo[1] || x1 > xexthi[1]) continue;
                    srcimg_out.push_back(x0);
                    srcimg_out.push_back(x1);
                    img_src_idx_out.push_back(j);
                }
            }
        }
    } else if constexpr (DIM == 3) {
        int n3max = (int)((mgrid[2] + 1) * rc / umat[2 + 2 * DIM]) + 1;
        for (int n3 = -n3max; n3 <= n3max; ++n3) {
            Real shift3 = umat[2 + 2 * DIM] * n3;
            if (xlo[2] + shift3 > xexthi[2] || xhi[2] + shift3 < xextlo[2]) continue;
            int n2max = (int)(((mgrid[1] + 1) * rc + std::abs(umat[1 + 2 * DIM] * n3))
                              / umat[1 + DIM]) + 1;
            for (int n2 = -n2max; n2 <= n2max; ++n2) {
                Real shift2 = umat[1 + DIM] * n2 + umat[1 + 2 * DIM] * n3;
                if (xlo[1] + shift2 > xexthi[1] || xhi[1] + shift2 < xextlo[1]) continue;
                Real off12 = umat[DIM] * n2 + umat[2 * DIM] * n3;
                int n1max = (int)(((mgrid[0] + 1) * rc + std::abs(off12)) / umat[0]) + 1;
                for (int n1 = -n1max; n1 <= n1max; ++n1) {
                    if (n1 == 0 && n2 == 0 && n3 == 0) continue;
                    Real shift1 = umat[0] * n1 + off12;
                    if (xlo[0] + shift1 > xexthi[0] || xhi[0] + shift1 < xextlo[0]) continue;
                    for (int j = 0; j < ns; ++j) {
                        Real x0 = sources[j * DIM] + shift1;
                        Real x1 = sources[j * DIM + 1] + shift2;
                        Real x2 = sources[j * DIM + 2] + shift3;
                        if (x0 < xextlo[0] || x0 > xexthi[0] ||
                            x1 < xextlo[1] || x1 > xexthi[1] ||
                            x2 < xextlo[2] || x2 > xexthi[2]) continue;
                        srcimg_out.push_back(x0);
                        srcimg_out.push_back(x1);
                        srcimg_out.push_back(x2);
                        img_src_idx_out.push_back(j);
                    }
                }
            }
        }
    }
    return (int)img_src_idx_out.size();
}

} // namespace pdmk
