#pragma once

#include <pdmk/types.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace pdmk {

template <int DIM>
class CellListT {
  public:
    CellListT(const MatND<DIM> &cell, double cutoff,
              const std::vector<VecND<DIM>> &positions);

    int size() const { return N_; }

    /// Callback: f(int j, const VecND<DIM> &dr, double r2)
    template <typename Func>
    void for_each_neighbor(int i, Func &&f) const;

    /// Callback: f(int i, int j, const VecND<DIM> &dr, double r2)
    /// Each unordered pair (including i==j with non-zero image) visited once.
    template <typename Func>
    void for_each_pair(Func &&f) const;

  private:
    // Loop order here (d = DIM-1 down to 0) must match ravel()/unravel()
    // so bin_of(frac) == bin_idx for the same particle — do not "fix"
    // one without the others.
    int bin_of(const VecND<DIM> &s) const {
        int idx = 0;
        for (int d = DIM - 1; d >= 0; --d) {
            int ix = std::min((int)(s[d] * nbins_[d]), nbins_[d] - 1);
            idx = idx * nbins_[d] + ix;
        }
        return idx;
    }
    std::array<int, DIM> unravel(int b) const {
        std::array<int, DIM> out{};
        for (int d = 0; d < DIM; ++d) {
            out[d] = b % nbins_[d];
            b /= nbins_[d];
        }
        return out;
    }
    int ravel(const std::array<int, DIM> &c) const {
        int b = 0;
        for (int d = DIM - 1; d >= 0; --d) b = b * nbins_[d] + c[d];
        return b;
    }

    static std::array<VecND<DIM>, DIM> columns_of(const MatND<DIM> &h) {
        std::array<VecND<DIM>, DIM> out{};
        for (int j = 0; j < DIM; ++j)
            for (int k = 0; k < DIM; ++k)
                out[j][k] = h[j][k];
        return out;
    }
    static VecND<DIM> to_cart(const std::array<VecND<DIM>, DIM> &cols,
                              const VecND<DIM> &s) {
        VecND<DIM> r{};
        for (int j = 0; j < DIM; ++j)
            for (int k = 0; k < DIM; ++k)
                r[k] += cols[j][k] * s[j];
        return r;
    }
    static VecND<DIM> to_frac(const MatND<DIM> &h_inv, const VecND<DIM> &r) {
        VecND<DIM> s{};
        for (int j = 0; j < DIM; ++j)
            for (int k = 0; k < DIM; ++k)
                s[k] += h_inv[j][k] * r[j];
        return s;
    }
    static std::array<double, DIM>
    heights_of(const std::array<VecND<DIM>, DIM> &a);
    static MatND<DIM> inverse_cell(const MatND<DIM> &h);

    double cutoff_, cutoff2_;
    int N_;
    int nbins_[DIM];
    int nimages_[DIM];
    std::array<VecND<DIM>, DIM> cols_;
    std::vector<VecND<DIM>> frac_;
    std::vector<VecND<DIM>> cart_;   // original Cartesian positions — avoids
                                     // rounding via to_cart(frac_[i]) in hot loops
    std::vector<int> bin_idx_;
    std::vector<int> bin_start_;
    std::vector<int> sorted_;
};

using CellList   = CellListT<3>;
using CellList2D = CellListT<2>;

// ---- template ctor ----
template <int DIM>
CellListT<DIM>::CellListT(const MatND<DIM> &cell, double cutoff,
                          const std::vector<VecND<DIM>> &positions)
    : cutoff_(cutoff), cutoff2_(cutoff * cutoff),
      N_((int)positions.size())
{
    cols_ = columns_of(cell);
    auto heights = heights_of(cols_);
    for (int d = 0; d < DIM; ++d)
        nbins_[d] = std::max(1, (int)std::floor(heights[d] / cutoff));
    for (int d = 0; d < DIM; ++d)
        nimages_[d] = (nbins_[d] == 1)
            ? (int)std::ceil(cutoff / heights[d]) : 1;
    MatND<DIM> h_inv = inverse_cell(cell);
    frac_.resize(N_);
    cart_.resize(N_);
    bin_idx_.resize(N_);
    for (int i = 0; i < N_; ++i) {
        cart_[i] = positions[i];
        VecND<DIM> s = to_frac(h_inv, positions[i]);
        for (int d = 0; d < DIM; ++d) s[d] -= std::floor(s[d]);
        frac_[i] = s;
        bin_idx_[i] = bin_of(s);
    }
    int total = 1;
    for (int d = 0; d < DIM; ++d) total *= nbins_[d];
    bin_start_.assign(total + 1, 0);
    for (int i = 0; i < N_; ++i) bin_start_[bin_idx_[i] + 1]++;
    for (int b = 0; b < total; ++b) bin_start_[b + 1] += bin_start_[b];
    sorted_.resize(N_);
    std::vector<int> count(total, 0);
    for (int i = 0; i < N_; ++i) {
        int b = bin_idx_[i];
        sorted_[bin_start_[b] + count[b]++] = i;
    }
}

// ---- neighbor iteration ----
template <int DIM>
template <typename Func>
void CellListT<DIM>::for_each_neighbor(int i, Func &&f) const {
    const VecND<DIM> &ri = cart_[i];
    int bi = bin_idx_[i];
    std::array<int, DIM> bic = unravel(bi);

    std::array<int, DIM> off;
    for (int d = 0; d < DIM; ++d) off[d] = -nimages_[d];
    while (true) {
        std::array<int, DIM> nc;
        std::array<int, DIM> shift;
        shift.fill(0);
        for (int d = 0; d < DIM; ++d) {
            nc[d] = bic[d] + off[d];
            while (nc[d] < 0)            { nc[d] += nbins_[d]; shift[d] -= 1; }
            while (nc[d] >= nbins_[d])   { nc[d] -= nbins_[d]; shift[d] += 1; }
        }
        int nb_idx = ravel(nc);
        VecND<DIM> img{};
        for (int d = 0; d < DIM; ++d)
            if (shift[d])
                for (int k = 0; k < DIM; ++k) img[k] += shift[d] * cols_[d][k];
        bool same = (nb_idx == bi);
        for (int d = 0; d < DIM; ++d) same = same && (shift[d] == 0);

        for (int jj = bin_start_[nb_idx]; jj < bin_start_[nb_idx+1]; ++jj) {
            int j = sorted_[jj];
            if (same && j == i) continue;
            const VecND<DIM> &rj = cart_[j];
            VecND<DIM> dr{};
            double r2 = 0.0;
            for (int k = 0; k < DIM; ++k) {
                dr[k] = rj[k] + img[k] - ri[k];
                r2 += dr[k] * dr[k];
            }
            if (r2 < cutoff2_ && r2 > 1e-30) f(j, dr, r2);
        }
        // Advance odometer
        int d = 0;
        for (; d < DIM; ++d) {
            off[d]++;
            if (off[d] <= nimages_[d]) break;
            off[d] = -nimages_[d];
        }
        if (d == DIM) break;
    }
}

// ---- pair iteration ----
template <int DIM>
template <typename Func>
void CellListT<DIM>::for_each_pair(Func &&f) const {
    int total = 1;
    for (int d = 0; d < DIM; ++d) total *= nbins_[d];
    for (int b = 0; b < total; ++b) {
        std::array<int, DIM> bc = unravel(b);
        std::array<int, DIM> off;
        for (int d = 0; d < DIM; ++d) off[d] = -nimages_[d];
        while (true) {
            std::array<int, DIM> nc;
            std::array<int, DIM> shift;
            shift.fill(0);
            for (int d = 0; d < DIM; ++d) {
                nc[d] = bc[d] + off[d];
                while (nc[d] < 0)          { nc[d] += nbins_[d]; shift[d] -= 1; }
                while (nc[d] >= nbins_[d]) { nc[d] -= nbins_[d]; shift[d] += 1; }
            }
            int nb_idx = ravel(nc);
            VecND<DIM> img{};
            for (int d = 0; d < DIM; ++d)
                if (shift[d])
                    for (int k = 0; k < DIM; ++k)
                        img[k] += shift[d] * cols_[d][k];
            bool all_zero_shift = true;
            for (int d = 0; d < DIM; ++d)
                if (shift[d]) { all_zero_shift = false; break; }
            bool same = (nb_idx == b) && all_zero_shift;

            bool keep;
            if (same) {
                keep = true;
            } else if (all_zero_shift) {
                keep = (nb_idx > b);
            } else {
                // Find highest d where shift != 0; keep iff positive.
                int top = DIM - 1;
                while (top >= 0 && shift[top] == 0) --top;
                keep = (top >= 0 && shift[top] > 0);
            }
            if (keep) {
                for (int ii = bin_start_[b]; ii < bin_start_[b+1]; ++ii) {
                    int i = sorted_[ii];
                    const VecND<DIM> &ri = cart_[i];
                    int jj_start = same ? ii : bin_start_[nb_idx];
                    for (int jj = jj_start; jj < bin_start_[nb_idx+1]; ++jj) {
                        int j = sorted_[jj];
                        if (same && j <= i) continue;
                        const VecND<DIM> &rj = cart_[j];
                        VecND<DIM> dr{};
                        double r2 = 0.0;
                        for (int k = 0; k < DIM; ++k) {
                            dr[k] = rj[k] + img[k] - ri[k];
                            r2 += dr[k] * dr[k];
                        }
                        if (r2 < cutoff2_ && r2 > 1e-30) f(i, j, dr, r2);
                        // For self-image pairs (i==j, non-zero image), the keep
                        // predicate drops the negative-shift direction, so we also
                        // emit the negative-image pair here.  Compute it freshly
                        // (pos[i] - img - pos[i]) rather than negating dr, so that
                        // the floating-point rounding matches what a brute-force
                        // loop over -k would produce.
                        if (!same && i == j) {
                            VecND<DIM> neg_dr{};
                            double neg_r2 = 0.0;
                            for (int k = 0; k < DIM; ++k) {
                                neg_dr[k] = cart_[j][k] - img[k] - cart_[i][k];
                                neg_r2 += neg_dr[k] * neg_dr[k];
                            }
                            if (neg_r2 < cutoff2_ && neg_r2 > 1e-30) f(i, j, neg_dr, neg_r2);
                        }
                    }
                }
            }

            // Advance odometer
            int d = 0;
            for (; d < DIM; ++d) {
                off[d]++;
                if (off[d] <= nimages_[d]) break;
                off[d] = -nimages_[d];
            }
            if (d == DIM) break;
        }
    }
}

// ---- geometry specializations ----

template <>
inline std::array<double, 2>
CellListT<2>::heights_of(const std::array<Vec2, 2> &a) {
    auto cross2 = [](const Vec2 &u, const Vec2 &v) {
        return std::abs(u[0]*v[1] - u[1]*v[0]);
    };
    auto norm = [](const Vec2 &u) { return std::sqrt(u[0]*u[0] + u[1]*u[1]); };
    double vol = cross2(a[0], a[1]);
    return { vol / norm(a[1]), vol / norm(a[0]) };
}

template <>
inline std::array<double, 3>
CellListT<3>::heights_of(const std::array<Vec3, 3> &a) {
    auto cross = [](const Vec3 &u, const Vec3 &v) {
        return Vec3{u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]};
    };
    auto dot = [](const Vec3 &u, const Vec3 &v) {
        return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
    };
    auto norm = [&](const Vec3 &u) { return std::sqrt(dot(u,u)); };
    Vec3 c12 = cross(a[1], a[2]);
    Vec3 c20 = cross(a[2], a[0]);
    Vec3 c01 = cross(a[0], a[1]);
    double vol = std::abs(dot(a[0], c12));
    return { vol / norm(c12), vol / norm(c20), vol / norm(c01) };
}

template <>
inline Mat2 CellListT<2>::inverse_cell(const Mat2 &h) {
    double det = h[0][0]*h[1][1] - h[0][1]*h[1][0];
    double inv = 1.0 / det;
    Mat2 out;
    out[0][0] =  h[1][1] * inv;
    out[0][1] = -h[0][1] * inv;
    out[1][0] = -h[1][0] * inv;
    out[1][1] =  h[0][0] * inv;
    return out;
}

template <>
inline Mat3 CellListT<3>::inverse_cell(const Mat3 &h) {
    double a00=h[0][0], a01=h[0][1], a02=h[0][2];
    double a10=h[1][0], a11=h[1][1], a12=h[1][2];
    double a20=h[2][0], a21=h[2][1], a22=h[2][2];
    double det =
          a00 * (a11 * a22 - a12 * a21)
        - a10 * (a01 * a22 - a02 * a21)
        + a20 * (a01 * a12 - a02 * a11);
    double inv = 1.0 / det;
    Mat3 out;
    out[0][0] =  (a11*a22 - a12*a21) * inv;
    out[0][1] = -(a01*a22 - a02*a21) * inv;
    out[0][2] =  (a01*a12 - a02*a11) * inv;
    out[1][0] = -(a10*a22 - a12*a20) * inv;
    out[1][1] =  (a00*a22 - a02*a20) * inv;
    out[1][2] = -(a00*a12 - a02*a10) * inv;
    out[2][0] =  (a10*a21 - a11*a20) * inv;
    out[2][1] = -(a00*a21 - a01*a20) * inv;
    out[2][2] =  (a00*a11 - a01*a10) * inv;
    return out;
}

} // namespace pdmk
