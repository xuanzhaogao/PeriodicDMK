using Test
using PeriodicDMK
using Random
using LinearAlgebra

@testset "PeriodicDMK.jl" begin

    @testset "NaCl Madelung constant" begin
        a = 1.0
        cell = a * Matrix{Float64}(I, 3, 3)

        # 8-ion NaCl rock-salt conventional cell
        positions = [
            0.0    0.5a   0.5a   0.0    0.5a   0.0    0.0    0.5a;
            0.0    0.5a   0.0    0.5a   0.0    0.5a   0.0    0.5a;
            0.0    0.0    0.5a   0.5a   0.0    0.0    0.5a   0.5a
        ]
        charges = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]

        solver = PDMKSolver(cell, positions; n_digits=6, n_per_leaf=4, precision=:double)
        pot = evaluate(solver, positions, charges)

        @test length(pot) == 8
        @test all(isfinite, pot)

        energy = 0.5 * dot(charges, pot)
        M_computed = -energy * a / 8.0
        M_ref = 1.7475645946
        @test isapprox(M_computed, M_ref; rtol=0.01)
    end

    @testset "nz keyword construction" begin
        cell = 2.0 * Matrix{Float64}(I, 3, 3)
        positions = reshape([0.5, 0.5, 0.5], 3, 1)
        charges = [1.0]
        solver = PDMKSolver(cell, positions; n_digits=6, n_per_leaf=20, nz=4, precision=:double)
        pot = evaluate(solver, positions, charges)

        @test length(pot) == 1
        @test isfinite(pot[1])
    end

    @testset "optimize_cubic_cover shapes" begin
        # Rectangular diag(4,2,1), nz=4: expect (16, 8, 4) with s=0.25.
        rect = [4.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0]
        cr = optimize_cubic_cover(rect; nz=4)
        @test cr.p == 4
        @test cr.m == 16
        @test cr.n == 8
        @test isapprox(cr.s, 0.25; atol=1e-12)

        # Slender 16×2×2, nz=4: expect (32, 4, 4) with s=0.5.
        slender = [16.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
        cr2 = optimize_cubic_cover(slender; nz=4)
        @test cr2.p == 4
        @test cr2.m == 32
        @test cr2.n == 4
        @test isapprox(cr2.s, 0.5; atol=1e-12)

        # Invalid nz
        @test_throws ErrorException optimize_cubic_cover(rect; nz=0)
    end

    @testset "Random system vs Ewald reference (double)" begin
        L = 3.0
        cell = L * Matrix{Float64}(I, 3, 3)

        rng = MersenneTwister(42)
        N = 100
        positions = L .* rand(rng, 3, N) .* 0.98 .+ 0.01 * L
        charges = randn(rng, N)
        charges[1] -= sum(charges)  # charge neutral

        solver = PDMKSolver(cell, positions; n_digits=6, n_per_leaf=20, precision=:double)
        pot_dmk = evaluate(solver, positions, charges)

        pot_ref, energy_ref = ewald_nufft(cell, positions, charges; s=5.0)

        # L2 relative error
        err = norm(pot_dmk .- pot_ref) / norm(pot_ref)
        @test err < 1e-4
    end

    @testset "Random system vs Ewald reference (float)" begin
        L = 3.0
        cell = L * Matrix{Float64}(I, 3, 3)

        rng = MersenneTwister(42)
        N = 100
        positions = L .* rand(rng, 3, N) .* 0.98 .+ 0.01 * L
        charges = randn(rng, N)
        charges[1] -= sum(charges)

        solver = PDMKSolver(cell, positions; n_digits=6, n_per_leaf=20, precision=:float)
        pot_dmk = evaluate(solver, positions, charges)

        pot_ref, _ = ewald_nufft(cell, positions, charges; s=5.0)

        err = norm(Float64.(pot_dmk) .- pot_ref) / norm(pot_ref)
        @test err < 1e-4
    end

    @testset "Float vs double consistency" begin
        L = 3.0
        cell = L * Matrix{Float64}(I, 3, 3)

        rng = MersenneTwister(99)
        N = 50
        positions = L .* rand(rng, 3, N) .* 0.98 .+ 0.01 * L
        charges = randn(rng, N)
        charges[1] -= sum(charges)

        solver_d = PDMKSolver(cell, positions; n_digits=6, n_per_leaf=20, precision=:double)
        solver_f = PDMKSolver(cell, positions; n_digits=6, n_per_leaf=20, precision=:float)
        pot_d = evaluate(solver_d, positions, charges)
        pot_f = evaluate(solver_f, positions, charges)

        # Float and double should agree to ~1e-4
        err = norm(Float64.(pot_f) .- pot_d) / norm(pot_d)
        @test err < 1e-3
    end

    @testset "Triclinic cell" begin
        cell = [5.0 1.0 0.5;
                0.0 4.5 0.8;
                0.0 0.0 4.0]

        rng = MersenneTwister(77)
        N = 50
        # Generate in fractional coords, convert to Cartesian
        frac = rand(rng, 3, N) .* 0.98 .+ 0.01
        positions = cell * frac
        charges = randn(rng, N)
        charges[1] -= sum(charges)

        solver = PDMKSolver(cell, positions; n_digits=6, n_per_leaf=20, precision=:double)
        pot_dmk = evaluate(solver, positions, charges)

        pot_ref, _ = ewald_nufft(cell, positions, charges; s=5.0)

        err = norm(pot_dmk .- pot_ref) / norm(pot_ref)
        @test err < 1e-4
    end

    @testset "Ewald NUFFT energy" begin
        a = 1.0
        cell = a * Matrix{Float64}(I, 3, 3)

        positions = [
            0.0    0.5a   0.5a   0.0    0.5a   0.0    0.0    0.5a;
            0.0    0.5a   0.0    0.5a   0.0    0.5a   0.0    0.5a;
            0.0    0.0    0.5a   0.5a   0.0    0.0    0.5a   0.5a
        ]
        charges = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]

        pot, energy = ewald_nufft(cell, positions, charges; s=5.5)

        # Energy should equal 0.5 * sum(q_i * u_i)
        energy_check = 0.5 * dot(charges, pot)
        @test isapprox(energy, energy_check; rtol=1e-10)

        # Madelung constant
        M = -energy * a / 8.0
        @test isapprox(M, 1.7475645946; atol=1e-6)
    end
end
