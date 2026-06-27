// cpp/include/pdmk/campaign_ref_io.hpp
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace pdmk::campaign {

// Magic = 'PDMC' little-endian = 0x43 4D 44 50.
constexpr uint32_t MAGIC = 0x50444D43u;
constexpr uint32_t VERSION = 1u;
constexpr uint32_t ENDIAN_TAG = 0x01020304u;
constexpr uint32_t HEADER_LEN = 80u;
constexpr uint32_t DEFAULT_M = 1000u;

#pragma pack(push, 1)
struct RefHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t endian_tag;
    uint32_t header_len;
    uint32_t config_id;
    uint32_t N;
    uint8_t  digits;
    uint8_t  prec;
    uint8_t  _pad[2];
    uint32_t M;
    uint64_t seed;
    uint64_t body_len;
    uint8_t  body_sha256[32];
};
#pragma pack(pop)
static_assert(sizeof(RefHeader) == 80, "RefHeader must be 80 bytes");

struct RefData {
    RefHeader header;
    double cell[9];                 // column-major
    std::vector<uint32_t> ref_idx;  // size M
    std::vector<double> ref_u;      // size M
    std::vector<double> pos;        // size 3N
    std::vector<double> chg;        // size N
};

namespace detail {
    inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

    inline constexpr uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
}

inline void sha256(const uint8_t *data, size_t len, uint8_t out[32]) {
    using detail::rotr;
    using detail::K;
    uint32_t H[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                     0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    auto process = [&](const uint8_t *chunk) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(chunk[4*i])<<24) | (uint32_t(chunk[4*i+1])<<16)
                 | (uint32_t(chunk[4*i+2])<<8) | uint32_t(chunk[4*i+3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15]>>3);
            uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19)  ^ (w[i-2]>>10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e,6)^rotr(e,11)^rotr(e,25);
            uint32_t ch = (e&f) ^ (~e & g);
            uint32_t t1 = h + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a,2)^rotr(a,13)^rotr(a,22);
            uint32_t mj = (a&b) ^ (a&c) ^ (b&c);
            uint32_t t2 = S0 + mj;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        H[0]+=a;H[1]+=b;H[2]+=c;H[3]+=d;H[4]+=e;H[5]+=f;H[6]+=g;H[7]+=h;
    };
    size_t full = len / 64;
    for (size_t i = 0; i < full; ++i) process(data + 64*i);
    uint8_t tail[128] = {};
    size_t rem = len % 64;
    std::memcpy(tail, data + 64*full, rem);
    tail[rem] = 0x80;
    size_t pad_end = (rem < 56) ? 64 : 128;
    uint64_t bits = uint64_t(len) * 8;
    for (int i = 0; i < 8; ++i) tail[pad_end - 1 - i] = uint8_t(bits >> (8*i));
    process(tail);
    if (pad_end == 128) process(tail + 64);
    for (int i = 0; i < 8; ++i) {
        out[4*i+0] = uint8_t(H[i] >> 24);
        out[4*i+1] = uint8_t(H[i] >> 16);
        out[4*i+2] = uint8_t(H[i] >> 8);
        out[4*i+3] = uint8_t(H[i]);
    }
}

inline std::vector<uint8_t> pack_body(const RefData &d) {
    size_t cell_bytes = sizeof(d.cell);
    size_t idx_bytes  = d.ref_idx.size() * sizeof(uint32_t);
    size_t u_bytes    = d.ref_u.size()   * sizeof(double);
    size_t pos_bytes  = d.pos.size()     * sizeof(double);
    size_t chg_bytes  = d.chg.size()     * sizeof(double);
    std::vector<uint8_t> body(cell_bytes + idx_bytes + u_bytes + pos_bytes + chg_bytes);
    size_t off = 0;
    std::memcpy(body.data()+off, d.cell,            cell_bytes); off += cell_bytes;
    std::memcpy(body.data()+off, d.ref_idx.data(),  idx_bytes);  off += idx_bytes;
    std::memcpy(body.data()+off, d.ref_u.data(),    u_bytes);    off += u_bytes;
    std::memcpy(body.data()+off, d.pos.data(),      pos_bytes);  off += pos_bytes;
    std::memcpy(body.data()+off, d.chg.data(),      chg_bytes);
    return body;
}

inline void write_ref(const std::string &path, const RefData &d_in) {
    RefData d = d_in;
    d.header.magic      = MAGIC;
    d.header.version    = VERSION;
    d.header.endian_tag = ENDIAN_TAG;
    d.header.header_len = HEADER_LEN;
    d.header.M          = (uint32_t)d.ref_idx.size();
    auto body = pack_body(d);
    d.header.body_len = body.size();
    sha256(body.data(), body.size(), d.header.body_sha256);

    FILE *fp = std::fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("cannot open for write: " + path);
    if (std::fwrite(&d.header, sizeof(RefHeader), 1, fp) != 1) {
        std::fclose(fp); throw std::runtime_error("header write failed: " + path);
    }
    if (std::fwrite(body.data(), 1, body.size(), fp) != body.size()) {
        std::fclose(fp); throw std::runtime_error("body write failed: " + path);
    }
    std::fclose(fp);
}

inline RefData read_ref(const std::string &path) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("cannot open for read: " + path);
    RefData d;
    if (std::fread(&d.header, sizeof(RefHeader), 1, fp) != 1) {
        std::fclose(fp); throw std::runtime_error("header short read: " + path);
    }
    if (d.header.magic != MAGIC)       { std::fclose(fp); throw std::runtime_error("bad magic: " + path); }
    if (d.header.version != VERSION)   { std::fclose(fp); throw std::runtime_error("bad version: " + path); }
    if (d.header.endian_tag != ENDIAN_TAG) { std::fclose(fp); throw std::runtime_error("bad endian tag: " + path); }
    if (d.header.header_len != HEADER_LEN) { std::fclose(fp); throw std::runtime_error("bad header_len: " + path); }

    std::vector<uint8_t> body(d.header.body_len);
    if (std::fread(body.data(), 1, body.size(), fp) != body.size()) {
        std::fclose(fp); throw std::runtime_error("body short read: " + path);
    }
    std::fclose(fp);

    uint8_t digest[32];
    sha256(body.data(), body.size(), digest);
    if (std::memcmp(digest, d.header.body_sha256, 32) != 0)
        throw std::runtime_error("sha256 mismatch: " + path);

    size_t off = 0;
    std::memcpy(d.cell, body.data()+off, sizeof(d.cell)); off += sizeof(d.cell);
    d.ref_idx.resize(d.header.M);
    std::memcpy(d.ref_idx.data(), body.data()+off, d.header.M*sizeof(uint32_t));
    off += d.header.M*sizeof(uint32_t);
    d.ref_u.resize(d.header.M);
    std::memcpy(d.ref_u.data(), body.data()+off, d.header.M*sizeof(double));
    off += d.header.M*sizeof(double);
    d.pos.resize((size_t)3 * d.header.N);
    std::memcpy(d.pos.data(), body.data()+off, d.pos.size()*sizeof(double));
    off += d.pos.size()*sizeof(double);
    d.chg.resize(d.header.N);
    std::memcpy(d.chg.data(), body.data()+off, d.chg.size()*sizeof(double));
    return d;
}

} // namespace pdmk::campaign
