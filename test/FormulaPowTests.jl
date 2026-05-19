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

        @testset "CUDA (pow=2, n_vars=5, d=5)" begin
            n_vars = 5; d = 5; pow = 2
            n = binomial(n_vars + d - 1, d)
            coeffs = collect(Int64, 1:n)

            _, oscar_result = build_oscar_poly_and_power(n_vars, d, pow, coeffs)

            backend = CUDABackend()
            plan = formula_pow_plan(n_vars, d, pow, backend)

            original = CUDA.CuArray(coeffs)
            output = formula_pow(original, plan, backend)

            gpu_result = output_to_oscar_poly(Array(output), n_vars, d * pow)
            @test gpu_result == oscar_result
        end

        @testset "CUDA (pow=3, n_vars=4, d=8)" begin
            n_vars = 4; d = 8; pow = 3
            n = binomial(n_vars + d - 1, d)
            coeffs = collect(Int64, 1:n)

            _, oscar_result = build_oscar_poly_and_power(n_vars, d, pow, coeffs)

            backend = CUDABackend()
            plan = formula_pow_plan(n_vars, d, pow, backend)

            original = CUDA.CuArray(coeffs)
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

@testset "formula_pow — plan partition invariant (kwd)" begin
    # short_rows ∪ medium_rows ∪ long_rows must be a disjoint partition of
    # 1:num_rows. The dispatcher writes each output index from exactly one
    # tier kernel, so a complete partition lets us skip zero-init.
    backend = KernelAbstractions.CPU()
    for (n_vars, d, pow) in [(3,2,2), (4,4,2), (5,5,2), (4,4,6), (4,8,3)]
        plan = formula_pow_plan(n_vars, d, pow, backend)
        num_rows = length(plan.term_ptr) - 1
        s = Array(plan.short_rows)
        m = Array(plan.medium_rows)
        l = Array(plan.long_rows)
        combined = sort(vcat(s, m, l))
        @test combined == collect(Int32, 1:num_rows)
        @test isempty(intersect(s, m))
        @test isempty(intersect(m, l))
        @test isempty(intersect(s, l))
    end
end

@testset "formula_pow! — caller-provided output buffer (63i)" begin
    # Seed buf with sentinel garbage; formula_pow! must overwrite every index.
    # If any row escapes a tier kernel, the sentinel would leak into the result
    # and the Oscar comparison would fail.
    backend = KernelAbstractions.CPU()
    sentinel = Int(-12345)
    for (n_vars, d, pow) in [(3,2,2), (4,4,2), (4,4,6)]
        n = binomial(n_vars + d - 1, d)
        coeffs = collect(Int, 1:n)
        _, oscar_result = build_oscar_poly_and_power(n_vars, d, pow, coeffs)
        plan = formula_pow_plan(n_vars, d, pow, backend)
        num_out = length(plan.term_ptr) - 1
        buf = fill(sentinel, num_out)
        ret = formula_pow!(buf, copy(coeffs), plan, backend)
        @test ret === buf
        @test output_to_oscar_poly(buf, n_vars, d * pow) == oscar_result
        @test !any(==(sentinel), buf)
    end
end

@testset "formula_pow — tier rows sorted by length (6jj.2)" begin
    # K1 (1 thread/row, batched-only) places adjacent rows of the row-list in
    # adjacent threads of a warp. Sorting short_rows and medium_rows by row
    # length at plan time keeps each warp's lengths nearly uniform, avoiding
    # 40-60% divergence loss. K2 (warp-per-row) is indifferent to sort.
    backend = KernelAbstractions.CPU()
    for (n_vars, d, pow) in [(3,2,2), (4,4,2), (4,4,6), (4,8,3)]
        plan = formula_pow_plan(n_vars, d, pow, backend)
        tp = Array(plan.term_ptr)
        row_len(r) = tp[r+1] - tp[r]
        @test issorted(row_len.(Array(plan.short_rows)))
        @test issorted(row_len.(Array(plan.medium_rows)))
    end
end

@testset "formula_pow — batched correctness scaffolding (6jj.1)" begin
    # Test scaffolding for the future batched API (epic GPUPolynomials.jl-6jj).
    # Real assertions here are properties that hold against the current
    # single-op code and must continue to hold once the batched dispatcher
    # (6jj.5) and sort (6jj.2) land. Batched-API cases are stubbed with
    # @test_skip and activate once 6jj.3 / 6jj.5 surface the new entry points.

    backend = KernelAbstractions.CPU()

    @testset "pow=1 degenerate" begin
        # (Σ aᵢ mᵢ)^1 must round-trip the input coefficients unchanged in
        # the d_out = d basis. Future batched dispatch must preserve this.
        for (n_vars, d) in [(3, 2), (4, 4)]
            n = binomial(n_vars + d - 1, d)
            coeffs = collect(Int, 1:n)
            _, oscar_result = build_oscar_poly_and_power(n_vars, d, 1, coeffs)
            plan = formula_pow_plan(n_vars, d, 1, backend)
            output = formula_pow(coeffs, plan, backend)
            @test output_to_oscar_poly(output, n_vars, d) == oscar_result
        end
    end

    @testset "tier-row ordering invariance" begin
        # GPUPolynomials.jl-6jj.2 reorders short_rows / medium_rows by row
        # length. Output indexing is by output_row (via term_ptr[row]), not
        # by position within the tier array, so output must be identical
        # regardless of how the tier arrays are permuted.
        for (n_vars, d, pow) in [(3,2,2), (4,4,2)]
            n = binomial(n_vars + d - 1, d)
            coeffs = collect(Int, 1:n)
            plan = formula_pow_plan(n_vars, d, pow, backend)
            ref  = formula_pow(coeffs, plan, backend)
            scrambled_plan = FormulaPowPlan(
                plan.term_ptr, plan.term_coeffs,
                plan.monomial_ptr, plan.monomial_indices, plan.monomial_degrees,
                reverse(plan.short_rows),
                reverse(plan.medium_rows),
                reverse(plan.long_rows),
                plan.workgroup_size, plan.medium_workgroup_size,
            )
            scrambled = formula_pow(coeffs, scrambled_plan, backend)
            @test scrambled == ref
        end
    end

    # The cases below await the batched API surfaced by GPUPolynomials.jl-6jj.3
    # (plan with batch_size) and GPUPolynomials.jl-6jj.5 (dispatcher). Replace
    # @test_skip with real implementations once those land; the assertions
    # must hold via a degenerate batch_size=1 path through the new API.
    @testset "future batched API — awaiting 6jj.3 / 6jj.5" begin
        # B=1 batched path matches single-op output for (4,4,2), (4,4,6),
        # (4,8,3), (5,5,2).
        @test_skip false  # B=1 batched ↔ single-op parity, all four configs
        # B=2 with distinct originals; per-element output matches single-op.
        @test_skip false  # B=2 batched, distinct originals
        # B=1024 path (triggers K1 dispatch on padded layout); per-element
        # matches single-op on (4,4,2) and (4,8,3).
        @test_skip false  # B=1024 batched, K1 dispatch
        # Mixed-batch: same plan, all-different inputs across the batch.
        @test_skip false  # mixed-batch correctness
    end
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

    @testset "CPU (pow=2, n_vars=5, d=5)" begin
        n_vars = 5; d = 5; pow = 2
        n = binomial(n_vars + d - 1, d)
        coeffs = collect(1:n)

        _, oscar_result = build_oscar_poly_and_power(n_vars, d, pow, coeffs)

        backend = KernelAbstractions.CPU()
        plan = formula_pow_plan(n_vars, d, pow, backend)

        original = collect(Int, coeffs)
        output = formula_pow(original, plan, backend)

        gpu_result = output_to_oscar_poly(output, n_vars, d * pow)
        @test gpu_result == oscar_result
    end

    @testset "CPU (pow=3, n_vars=4, d=8)" begin
        n_vars = 4; d = 8; pow = 3
        n = binomial(n_vars + d - 1, d)
        coeffs = collect(1:n)

        _, oscar_result = build_oscar_poly_and_power(n_vars, d, pow, coeffs)

        backend = KernelAbstractions.CPU()
        plan = formula_pow_plan(n_vars, d, pow, backend)

        original = collect(Int, coeffs)
        output = formula_pow(original, plan, backend)

        gpu_result = output_to_oscar_poly(output, n_vars, d * pow)
        @test gpu_result == oscar_result
    end
end
