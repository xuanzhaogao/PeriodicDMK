module PeriodicDMK

export PDMKSolver, evaluate, ewald_nufft, optimize_cubic_cover, CoverResult

# Load library path from build step
const depsfile = joinpath(@__DIR__, "..", "deps", "deps.jl")
if isfile(depsfile)
    include(depsfile)
else
    error("PeriodicDMK not built. Run: using Pkg; Pkg.build(\"PeriodicDMK\")")
end

# ── CoverResult ──────────────────────────────────────────────────────────────

"""
    CoverResult

Result of `optimize_cubic_cover`. Fields:
- `m`, `n`, `p`: inner grid dimensions.
- `e1`, `e2`, `e3`: orthonormal frame columns (3-tuples of Float64).
- `s`: cube side length.
- `cost`: overhead ratio `((m+2)(n+2)(p+2) s^3) / V`.
"""
struct CoverResult
    m::Int
    n::Int
    p::Int
    e1::NTuple{3, Float64}
    e2::NTuple{3, Float64}
    e3::NTuple{3, Float64}
    s::Float64
    cost::Float64
end

"""
    optimize_cubic_cover(cell; nz=4) -> CoverResult

Compute the cubic cover for a 3×3 lattice cell (columns = lattice vectors) using
lattice reduction. `nz` sets how many cube layers divide the shortest lattice
direction after rotation; `m` and `n` are derived so all cubes share a side length.
"""
function optimize_cubic_cover(cell::AbstractMatrix{Float64}; nz::Int=4)
    size(cell) == (3, 3) || throw(ArgumentError("cell must be 3×3"))

    cell_m = Matrix{Float64}(cell)
    m_out = Ref{Cint}(0); n_out = Ref{Cint}(0); p_out = Ref{Cint}(0)
    e1_out = Vector{Float64}(undef, 3)
    e2_out = Vector{Float64}(undef, 3)
    e3_out = Vector{Float64}(undef, 3)
    s_out    = Ref{Float64}(0.0)
    cost_out = Ref{Float64}(0.0)

    ret = ccall((:pdmk_optimize_cubic_cover, libpdmk_jl), Cint,
                (Ptr{Float64}, Cint,
                 Ref{Cint}, Ref{Cint}, Ref{Cint},
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                 Ref{Float64}, Ref{Float64}),
                cell_m, nz,
                m_out, n_out, p_out,
                e1_out, e2_out, e3_out,
                s_out, cost_out)
    ret == 0 || error("pdmk_optimize_cubic_cover failed (code $ret)")

    return CoverResult(
        Int(m_out[]), Int(n_out[]), Int(p_out[]),
        (e1_out[1], e1_out[2], e1_out[3]),
        (e2_out[1], e2_out[2], e2_out[3]),
        (e3_out[1], e3_out[2], e3_out[3]),
        s_out[], cost_out[],
    )
end

# ── PDMKSolver ───────────────────────────────────────────────────────────────

"""
    PDMKSolver(cell::Matrix{Float64}, positions::Matrix{Float64};
               n_digits=6, n_per_leaf=300, nz=4, precision=:auto)

Construct a periodic Coulomb solver for the given unit cell and particle
positions. The tree is built once and can be re-evaluated with different
charges via `evaluate(solver, charges)`.

`cell` is a 3×3 matrix whose columns are the lattice vectors.
`positions` is a 3×N matrix of Cartesian coordinates.

`nz` is the number of cube layers along the shortest lattice direction after
rotation (see `optimize_cubic_cover`). `precision` can be `:double`, `:float`,
or `:auto` (uses float for `n_digits ≤ 6`).
"""
mutable struct PDMKSolver
    ptr::Ptr{Cvoid}
    is_float::Bool
    N::Int

    function PDMKSolver(cell::AbstractMatrix{Float64},
                        positions::AbstractMatrix{Float64};
                        n_digits::Int=6,
                        n_per_leaf::Int=300,
                        nz::Int=4,
                        precision::Symbol=:auto)
        size(cell) == (3, 3) || throw(ArgumentError("cell must be 3×3"))
        size(positions, 1) == 3 || throw(ArgumentError("positions must be 3×N"))
        N = size(positions, 2)

        use_float = if precision == :auto
            n_digits <= 6
        elseif precision == :float
            n_digits <= 6 || throw(ArgumentError("float requires n_digits ≤ 6"))
            true
        elseif precision == :double
            false
        else
            throw(ArgumentError("precision must be :auto, :float, or :double"))
        end

        # cell and positions are already column-major in Julia — pass directly
        cell_m = Matrix{Float64}(cell)
        pos = Matrix{Float64}(positions)
        ptr = if use_float
            ccall((:pdmk_create_float, libpdmk_jl), Ptr{Cvoid},
                  (Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint, Cint),
                  cell_m, N, pos, n_digits, n_per_leaf, nz)
        else
            ccall((:pdmk_create_double, libpdmk_jl), Ptr{Cvoid},
                  (Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint, Cint),
                  cell_m, N, pos, n_digits, n_per_leaf, nz)
        end

        ptr == C_NULL && error("Failed to create PDMK solver")

        solver = new(ptr, use_float, N)
        finalizer(solver) do s
            if s.ptr != C_NULL
                if s.is_float
                    ccall((:pdmk_destroy_float, libpdmk_jl), Cvoid, (Ptr{Cvoid},), s.ptr)
                else
                    ccall((:pdmk_destroy_double, libpdmk_jl), Cvoid, (Ptr{Cvoid},), s.ptr)
                end
                s.ptr = C_NULL
            end
        end
        return solver
    end
end

"""
    evaluate(solver::PDMKSolver, charges::Vector{Float64}) -> Vector
    evaluate(solver::PDMKSolver, positions::Matrix{Float64}, charges::Vector{Float64}) -> Vector

Compute periodic Coulomb potentials. `charges` is length N (matching the
positions passed at solver construction). The 3-arg form accepts a
`positions` argument for backward compatibility; it must have the same N as
the solver was built with (positions themselves are ignored — the tree is
built at construction time).

Returns `Vector{Float64}` (double solver) or `Vector{Float32}` (float solver).
"""
function evaluate(solver::PDMKSolver, positions::AbstractMatrix{Float64},
                  charges::AbstractVector{Float64})
    solver.ptr == C_NULL && error("Solver has been destroyed")
    size(positions, 1) == 3 || throw(ArgumentError("positions must be 3×N"))
    N = size(positions, 2)
    N == solver.N || throw(ArgumentError("positions N=$N does not match solver N=$(solver.N)"))
    length(charges) == N || throw(ArgumentError("charges length must match N"))

    # Ensure contiguous column-major layout
    pos = Matrix{Float64}(positions)
    q = Vector{Float64}(charges)

    if solver.is_float
        potentials = Vector{Float32}(undef, N)
        ret = ccall((:pdmk_evaluate_float, libpdmk_jl), Cint,
                    (Ptr{Cvoid}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float32}),
                    solver.ptr, N, pos, q, potentials)
        ret == 0 || error("pdmk_evaluate_float failed (code $ret)")
        return potentials
    else
        potentials = Vector{Float64}(undef, N)
        ret = ccall((:pdmk_evaluate_double, libpdmk_jl), Cint,
                    (Ptr{Cvoid}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                    solver.ptr, N, pos, q, potentials)
        ret == 0 || error("pdmk_evaluate_double failed (code $ret)")
        return potentials
    end
end

# 2-arg form: positions implicit (baked into solver at construction).
function evaluate(solver::PDMKSolver, charges::AbstractVector{Float64})
    solver.ptr == C_NULL && error("Solver has been destroyed")
    N = solver.N
    length(charges) == N || throw(ArgumentError("charges length must match solver N=$N"))
    q = Vector{Float64}(charges)
    # positions pointer is unused by the C API but needs to be non-null.
    dummy_pos = Matrix{Float64}(undef, 3, N)

    if solver.is_float
        potentials = Vector{Float32}(undef, N)
        ret = ccall((:pdmk_evaluate_float, libpdmk_jl), Cint,
                    (Ptr{Cvoid}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float32}),
                    solver.ptr, N, dummy_pos, q, potentials)
        ret == 0 || error("pdmk_evaluate_float failed (code $ret)")
        return potentials
    else
        potentials = Vector{Float64}(undef, N)
        ret = ccall((:pdmk_evaluate_double, libpdmk_jl), Cint,
                    (Ptr{Cvoid}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                    solver.ptr, N, dummy_pos, q, potentials)
        ret == 0 || error("pdmk_evaluate_double failed (code $ret)")
        return potentials
    end
end

# ── Ewald NUFFT reference ────────────────────────────────────────────────────

"""
    ewald_nufft(cell::Matrix{Float64}, positions::Matrix{Float64},
                charges::Vector{Float64}; s=4.8) -> (potentials, energy)

O(N log N) Ewald summation reference. Returns per-particle potentials and total energy.
"""
function ewald_nufft(cell::AbstractMatrix{Float64},
                     positions::AbstractMatrix{Float64},
                     charges::AbstractVector{Float64};
                     s::Float64=4.8)
    size(cell) == (3, 3) || throw(ArgumentError("cell must be 3×3"))
    size(positions, 1) == 3 || throw(ArgumentError("positions must be 3×N"))
    N = size(positions, 2)
    length(charges) == N || throw(ArgumentError("charges length must match N"))

    cell_m = Matrix{Float64}(cell)
    pos = Matrix{Float64}(positions)
    q = Vector{Float64}(charges)
    potentials = Vector{Float64}(undef, N)
    energy = Ref{Float64}(0.0)

    ret = ccall((:pdmk_ewald_nufft, libpdmk_jl), Cint,
                (Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64}, Float64,
                 Ptr{Float64}, Ptr{Float64}),
                cell_m, N, pos, q, s, potentials, energy)
    ret == 0 || error("pdmk_ewald_nufft failed (code $ret)")
    return potentials, energy[]
end

end # module
