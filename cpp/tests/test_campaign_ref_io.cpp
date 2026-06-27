#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <pdmk/campaign_ref_io.hpp>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace pdmk::campaign;

TEST_CASE("SHA-256 of empty string matches known digest") {
    uint8_t out[32];
    sha256(reinterpret_cast<const uint8_t*>(""), 0, out);
    // e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    const uint8_t expected[32] = {
        0xe3,0xb0,0xc4,0x42,0x98,0xfc,0x1c,0x14,0x9a,0xfb,0xf4,0xc8,0x99,0x6f,0xb9,0x24,
        0x27,0xae,0x41,0xe4,0x64,0x9b,0x93,0x4c,0xa4,0x95,0x99,0x1b,0x78,0x52,0xb8,0x55};
    for (int i = 0; i < 32; ++i) CHECK(out[i] == expected[i]);
}

TEST_CASE("RefFile round-trip") {
    RefData in{};
    in.header.config_id = 7;
    in.header.N = 50;
    in.header.digits = 6;
    in.header.prec = 1;
    in.header.seed = 0xDEADBEEFCAFEBABEull;

    for (int i = 0; i < 9; ++i) in.cell[i] = (i == 0 || i == 4 || i == 8) ? 1.0 : 0.0;

    in.ref_idx.resize(DEFAULT_M);
    in.ref_u.resize(DEFAULT_M);
    for (uint32_t i = 0; i < DEFAULT_M; ++i) {
        in.ref_idx[i] = i % in.header.N;
        in.ref_u[i] = double(i) * 1e-3;
    }
    in.pos.resize(3 * in.header.N);
    in.chg.resize(in.header.N);
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (auto &x : in.pos) x = U(rng);
    for (auto &x : in.chg) x = U(rng);

    std::string path = "/tmp/pdmk_ref_test.bin";
    write_ref(path, in);
    auto out = read_ref(path);

    CHECK(out.header.magic      == MAGIC);
    CHECK(out.header.version    == VERSION);
    CHECK(out.header.endian_tag == ENDIAN_TAG);
    CHECK(out.header.config_id  == 7);
    CHECK(out.header.N          == 50);
    CHECK(out.header.M          == DEFAULT_M);
    CHECK(out.header.seed       == 0xDEADBEEFCAFEBABEull);
    CHECK(out.ref_u.size()      == DEFAULT_M);
    CHECK(out.pos.size()        == size_t(3 * 50));
    for (int i = 0; i < 3 * 50; ++i) CHECK(out.pos[i] == doctest::Approx(in.pos[i]));

    std::remove(path.c_str());
}

TEST_CASE("RefFile rejects tampered body") {
    RefData in{};
    in.header.N = 10; in.header.digits = 6; in.header.prec = 1;
    in.pos.assign(30, 0.0); in.chg.assign(10, 0.0);
    in.ref_idx.assign(DEFAULT_M, 0); in.ref_u.assign(DEFAULT_M, 0.0);
    std::string path = "/tmp/pdmk_ref_tamper.bin";
    write_ref(path, in);

    FILE *fp = std::fopen(path.c_str(), "rb+");
    std::fseek(fp, 80, SEEK_SET);  // somewhere in the body
    uint8_t tamper = 0xFF;
    std::fwrite(&tamper, 1, 1, fp);
    std::fclose(fp);

    CHECK_THROWS_AS(read_ref(path), std::runtime_error);
    std::remove(path.c_str());
}
