#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vector>
#include <cmath>
#include <pdmk/image_layout.hpp>
#include <dmk/pbc.hpp>

TEST_CASE("generate_images_with_src_idx matches DMK-pbc image ordering") {
    using Real = double;
    std::vector<Real> avec = {2, 0, 0, 0, 2, 0, 0, 0, 2};
    std::vector<Real> r_src = {0.1, 0.2, 0.3,  1.7, 1.1, 0.9};
    const int n_src = 2;
    const Real rc = 0.5;

    // DMK-pbc reference: just image positions
    Real umat[9], qmat[9];
    dmk::pbc::lattice_setup<Real, 3>(avec.data(), umat, qmat);
    int mgrid[3];
    Real xgrid0[3];
    dmk::pbc::compute_grid<Real, 3>(umat, rc, mgrid, xgrid0);
    int nsimg = dmk::pbc::generate_images<Real, 3>(
        umat, rc, mgrid, xgrid0, n_src, r_src.data(), 1, nullptr,
        nullptr, nullptr, nullptr);
    std::vector<Real> srcimg(3 * nsimg), chgimg(nsimg);
    std::vector<Real> chg = {1.0, 1.0};
    dmk::pbc::generate_images<Real, 3>(
        umat, rc, mgrid, xgrid0, n_src, r_src.data(), 1, chg.data(),
        srcimg.data(), chgimg.data(), nullptr);

    // Our version: positions AND source indices
    std::vector<Real> srcimg2;
    std::vector<int>  img_src_idx;
    pdmk::generate_images_with_src_idx<Real, 3>(
        umat, rc, mgrid, xgrid0, n_src, r_src.data(),
        srcimg2, img_src_idx);

    REQUIRE(srcimg2.size() == srcimg.size());
    for (size_t i = 0; i < srcimg.size(); ++i) CHECK(srcimg2[i] == srcimg[i]);
    REQUIRE((int)img_src_idx.size() == nsimg);
    // Each idx must be valid
    for (int i : img_src_idx) CHECK((i == 0 || i == 1));
}
