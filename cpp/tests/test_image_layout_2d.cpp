#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <pdmk/image_layout.hpp>
#include <dmk/pbc.hpp>
#include <vector>
#include <cmath>

TEST_CASE("generate_images_with_src_idx 2D matches dmk::pbc::generate_images 2D") {
    constexpr int DIM = 2;
    using Real = double;
    // column-major upper-triangular 2x2: a1=(1,0), a2=(0,1)
    Real umat[4] = {1.0, 0.0,
                    0.0, 1.0};
    Real rc = 0.25;
    int mgrid[2] = {4, 4};
    Real xgrid0[2] = {0.0, 0.0};
    int ns = 3;
    std::vector<Real> src = {0.1, 0.1, 0.5, 0.5, 0.9, 0.9};

    // Reference: dmk::pbc::generate_images (count-only pass; ns=3, nd=1)
    int nref = dmk::pbc::generate_images<Real, DIM>(
        umat, rc, mgrid, xgrid0, ns, src.data(), 1, nullptr,
        nullptr, nullptr);

    // Under test
    std::vector<Real> srcimg;
    std::vector<int> idx;
    int n = pdmk::generate_images_with_src_idx<Real, DIM>(
        umat, rc, mgrid, xgrid0, ns, src.data(), srcimg, idx);

    CHECK(n == nref);
    CHECK((int)idx.size() == nref);
    CHECK((int)srcimg.size() == nref * DIM);

    // Every image coordinate must correspond to its parent plus some lattice shift
    for (int k = 0; k < n; ++k) {
        int j = idx[k];
        REQUIRE(j >= 0);
        REQUIRE(j < ns);
        Real dx = srcimg[k*DIM + 0] - src[j*DIM + 0];
        Real dy = srcimg[k*DIM + 1] - src[j*DIM + 1];
        Real n1 = dx;  // unit cell => integer lattice index
        Real n2 = dy;
        CHECK(std::abs(n1 - std::round(n1)) < 1e-9);
        CHECK(std::abs(n2 - std::round(n2)) < 1e-9);
        bool is_identity = ((int)std::round(n1) == 0 && (int)std::round(n2) == 0);
        CHECK_FALSE(is_identity);
    }
}
