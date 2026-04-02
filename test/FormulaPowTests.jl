# Tests for formula_pow: CSR plan construction (7bi) and hybrid dispatch (z1g)

using Oscar
using CUDA
using Metal
using KernelAbstractions
using Combinatorics

# Build an Oscar polynomial f = Σ coeffs[i]*mᵢ where mᵢ are the degree-d monomials
# in n_vars variables in with_replacement_combinations order, and return (f, f^pow).
function build_oscar_poly_and_power(n_vars, d, pow, coeffs)
    R, vars = polynomial_ring(ZZ, n_vars)
    f = zero(R)
    for (i, combo) in enumerate(with_replacement_combinations(1:n_vars, d))
        ev = zeros(Int, n_vars)
        for idx in combo; ev[idx] += 1; end
        mon = prod(vars[j]^ev[j] for j in 1:n_vars)
        f += coeffs[i] * mon
    end
    return f, f^pow
end

# Convert formula_pow output back to an Oscar polynomial.
# output[i] is the coefficient of the i-th monomial in
# with_replacement_combinations(1:n_vars, d_out) order.
function output_to_oscar_poly(output_coeffs, n_vars, d_out)
    R, vars = polynomial_ring(ZZ, n_vars)
    result = zero(R)
    for (i, combo) in enumerate(with_replacement_combinations(1:n_vars, d_out))
        c = output_coeffs[i]
        iszero(c) && continue
        ev = zeros(Int, n_vars)
        for idx in combo; ev[idx] += 1; end
        result += c * prod(vars[j]^ev[j] for j in 1:n_vars)
    end
    return result
end

@testset "formula_pow — CSR plan construction and CPU short kernel (7bi)" begin
    # n_vars=3, d=2, pow=2: all rows short (max row length well below 256)
    n_vars = 3; d = 2; pow = 2

    # Non-uniform coefficients so ordering bugs surface
    n = binomial(n_vars + d - 1, d)
    coeffs = collect(1:n)

    _, oscar_result = build_oscar_poly_and_power(n_vars, d, pow, coeffs)

    backend = KernelAbstractions.CPU()
    plan = formula_pow_plan(n_vars, d, pow, backend)

    # pow=2 with small n_vars,d → all rows should be short
    @test isempty(plan.long_rows)
    @test !isempty(plan.short_rows)

    original = collect(Int, coeffs)
    output = formula_pow(original, plan, backend)

    gpu_result = output_to_oscar_poly(output, n_vars, d * pow)
    @test gpu_result == oscar_result
end

if CUDA.functional()
    @testset "formula_pow — CUDA correctness (lin)" begin
        @testset "small pow CUDA (pow=2, n_vars=3, d=2)" begin
            n_vars = 3; d = 2; pow = 2
            n = binomial(n_vars + d - 1, d)
            coeffs = collect(Int64, 1:n)

            _, oscar_result = build_oscar_poly_and_power(n_vars, d, pow, coeffs)

            backend = CUDABackend()
            plan = formula_pow_plan(n_vars, d, pow, backend)

            @test isempty(plan.long_rows)

            original = CUDA.CuArray(coeffs)
            output = formula_pow(original, plan, backend)

            gpu_result = output_to_oscar_poly(Array(output), n_vars, d * pow)
            @test gpu_result == oscar_result
        end

        @testset "large pow CUDA (pow=6, n_vars=4, d=4)" begin
            # Use Int128 to avoid overflow: for pow=6 the multinomial coefficients
            # in term_coeffs can exceed typemax(Int64).
            n_vars = 4; d = 4; pow = 6
            n = binomial(n_vars + d - 1, d)
            coeffs = collect(Int128, 1:n)

            _, oscar_result = build_oscar_poly_and_power(n_vars, d, pow, coeffs)

            backend = CUDABackend()
            plan = formula_pow_plan(n_vars, d, pow, backend)

            @test !isempty(plan.long_rows)
            @test !isempty(plan.short_rows)

            original = CUDA.CuArray{Int128}(coeffs)
            output = formula_pow(original, plan, backend)

            gpu_result = output_to_oscar_poly(Array(output), n_vars, d * pow)
            @test gpu_result == oscar_result
        end
    end
else
    @info "Skipping CUDA formula_pow tests (no CUDA device)"
end

if Metal.functional()
    @testset "formula_pow — Metal correctness (8s3)" begin
        @testset "small pow Metal (pow=2, n_vars=3, d=2)" begin
            n_vars = 3; d = 2; pow = 2
            n = binomial(n_vars + d - 1, d)
            # Non-uniform coefficients so ordering bugs surface
            coeffs = collect(Int32, 1:n)

            _, oscar_result = build_oscar_poly_and_power(n_vars, d, pow, coeffs)

            backend = MetalBackend()
            plan = formula_pow_plan(n_vars, d, pow, backend)

            @test isempty(plan.long_rows)

            original = Metal.MtlArray(coeffs)
            output = formula_pow(original, plan, backend)

            gpu_result = output_to_oscar_poly(Array(output), n_vars, d * pow)
            @test gpu_result == oscar_result
        end

        @testset "large pow Metal (pow=6, n_vars=4, d=4)" begin
            # Metal does not support Int128; use Int64 instead.
            # Verified safe: max output coeff ≈ 1.7e16 < typemax(Int64) ≈ 9.2e18.
            n_vars = 4; d = 4; pow = 6
            n = binomial(n_vars + d - 1, d)
            coeffs = collect(Int64, 1:n)

            _, oscar_result = build_oscar_poly_and_power(n_vars, d, pow, coeffs)

            backend = MetalBackend()
            plan = formula_pow_plan(n_vars, d, pow, backend)

            @test !isempty(plan.long_rows)
            @test !isempty(plan.short_rows)

            original = Metal.MtlArray{Int64}(coeffs)
            output = formula_pow(original, plan, backend)

            gpu_result = output_to_oscar_poly(Array(output), n_vars, d * pow)
            @test gpu_result == oscar_result
        end
    end
else
    @info "Skipping Metal formula_pow tests (no Metal device)"
end

@testset "formula_pow — hybrid dispatch CPU (z1g)" begin
    @testset "small pow (pow=2, n_vars=4, d=4)" begin
        n_vars = 4; d = 4; pow = 2
        n = binomial(n_vars + d - 1, d)
        coeffs = collect(1:n)

        _, oscar_result = build_oscar_poly_and_power(n_vars, d, pow, coeffs)

        backend = KernelAbstractions.CPU()
        plan = formula_pow_plan(n_vars, d, pow, backend)

        @test isempty(plan.long_rows)

        original = collect(Int, coeffs)
        output = formula_pow(original, plan, backend)

        gpu_result = output_to_oscar_poly(output, n_vars, d * pow)
        @test gpu_result == oscar_result
    end

    @testset "large pow (pow=6, n_vars=4, d=4)" begin
        n_vars = 4; d = 4; pow = 6
        n = binomial(n_vars + d - 1, d)
        coeffs = collect(1:n)

        _, oscar_result = build_oscar_poly_and_power(n_vars, d, pow, coeffs)

        backend = KernelAbstractions.CPU()
        plan = formula_pow_plan(n_vars, d, pow, backend)

        # pow=6 should trigger long rows (max ~12652 terms per bead description)
        @test !isempty(plan.long_rows)
        @test !isempty(plan.short_rows)

        original = collect(Int, coeffs)
        output = formula_pow(original, plan, backend)

        gpu_result = output_to_oscar_poly(output, n_vars, d * pow)
        @test gpu_result == oscar_result
    end
end
