using KernelAbstractions
using KernelAbstractions: Adapt
using Combinatorics

# Module-level cache for the five raw CSR arrays keyed on (n_vars, d, pow).
# The expensive Oscar symbolic expansion is skipped on repeated calls.
# The adapt step (backend-specific) still runs each call.
const _formula_pow_cache = Dict{Tuple{Int,Int,Int},
    NamedTuple{(:term_ptr, :term_coeffs, :monomial_ptr,
                :monomial_indices, :monomial_degrees,
                :short_rows_cpu, :medium_rows_cpu, :long_rows_cpu),
               Tuple{Vector{Int32}, Vector{Int}, Vector{Int32},
                     Vector{Int32}, Vector{Int32},
                     Vector{Int32}, Vector{Int32}, Vector{Int32}}}}()

# =============================================================================
# generic_power_formula  (bead 9rr.1)
# =============================================================================
#
# Returns the symbolic polynomial (a₁ + a₂ + … + aₙ)^pow as an Oscar
# polynomial in ZZ[a₁,…,aₙ].  The aᵢ variables represent the n input
# monomials (in with_replacement_combinations(1:n_vars, d) order).
#
# The ordering contract: the i-th variable aᵢ corresponds to the i-th
# monomial in with_replacement_combinations(1:n_vars, d) order.
function generic_power_formula(n::Int, pow::Int)
    R, avars = polynomial_ring(ZZ, n)
    result = sum(avars)^pow
    return result, R
end

"""
    FormulaPowPlan

Pre-computed plan for evaluating (f₁m₁ + f₂m₂ + … + fₙmₙ)^pow, where mᵢ are
the degree-d monomials in n_vars variables.

# Ordering contract
`original[i]` passed to `formula_pow` MUST be the coefficient of the i-th monomial
in `with_replacement_combinations(1:n_vars, d)` order (lexicographic index order).
Violating this contract produces silently wrong results.

`output[i]` returned by `formula_pow` is the coefficient of the i-th monomial in
`with_replacement_combinations(1:n_vars, d*pow)` order.

# CSR layout (two-level)

Level 1: output monomial → contributing terms
  `term_ptr[i] : term_ptr[i+1]-1`  gives term indices for output monomial i
  `term_coeffs[j]`                  multinomial coefficient of term j

Level 2: term → input monomial factors
  `monomial_ptr[j] : monomial_ptr[j+1]-1`  gives factor indices for term j
  `monomial_indices[k]`                      which input monomial (1-based)
  `monomial_degrees[k]`                      to what power

Kernel computes:
  `output[i] = Σⱼ term_coeffs[j] * Πₖ original[monomial_indices[k]]^monomial_degrees[k]`
"""
struct FormulaPowPlan{V32<:AbstractVector{Int32}, VI<:AbstractVector{Int}}
    term_ptr::V32
    term_coeffs::VI
    monomial_ptr::V32
    monomial_indices::V32
    monomial_degrees::V32
    short_rows::V32
    medium_rows::V32
    long_rows::V32
    workgroup_size::Int
    medium_workgroup_size::Int
end

# =============================================================================
# expressions_from_poly  (bead 9rr.3)
# =============================================================================
#
# Given the expanded formula polynomial from generic_power_formula, group
# all terms by their corresponding output x-monomial and build flat CSR arrays.
#
# Two terms of (a₁+…+aₙ)^pow can map to the same output monomial when their
# input-monomial products are equal as x-polynomials (e.g. m₁*m₃ = m₂² when
# the x-monomial products collide).
#
# n_vars: number of original variables
# d:      degree of the input polynomial (each input monomial has degree d)
# poly:   the expanded formula in ZZ[a₁,…,aₙ]
#
# Returns (term_ptr, term_coeffs, monomial_ptr, monomial_indices, monomial_degrees)
# with output monomials in with_replacement_combinations(1:n_vars, d*pow) order.
function expressions_from_poly(poly, n_vars::Int, d::Int)
    n   = nvars(parent(poly))        # number of input monomials
    pow = total_degree(poly)         # = pow

    # Build output monomial index: x-exponent-vector → position (1-based)
    output_index = Dict{Vector{Int}, Int}()
    for (i, combo) in enumerate(with_replacement_combinations(1:n_vars, d * pow))
        ev = zeros(Int, n_vars)
        for idx in combo; ev[idx] += 1; end
        output_index[ev] = i
    end
    num_output = length(output_index)

    # Input monomial x-exponent vectors (with_replacement_combinations order)
    input_evs = Vector{Vector{Int}}(undef, n)
    for (j, combo) in enumerate(with_replacement_combinations(1:n_vars, d))
        ev = zeros(Int, n_vars)
        for idx in combo; ev[idx] += 1; end
        input_evs[j] = ev
    end

    # Accumulate (coeff, factors) lists per output monomial
    # row_data[i] = list of (coeff::Int, factors::Vector{(idx::Int, deg::Int)})
    row_data = [Tuple{Int, Vector{Tuple{Int,Int}}}[] for _ in 1:num_output]

    for term in terms(poly)
        a_ev = exponent_vector(term, 1)          # exponent in a-variables
        # Int(::ZZRingElem) throws InexactError if the value overflows Int.
        # For large pow, multinomial coefficients can exceed typemax(Int64).
        # Callers wanting large-pow support should use wider element type
        # (e.g. Int128) in the `original` array so term_coeffs stays in range.
        c    = Int(leading_coefficient(term))

        # Compute the x-monomial exponent for this term
        x_ev = zeros(Int, n_vars)
        for j in 1:n
            a_ev[j] == 0 && continue
            for k in 1:n_vars
                x_ev[k] += a_ev[j] * input_evs[j][k]
            end
        end

        out_idx = output_index[x_ev]

        factors = [(j, a_ev[j]) for j in 1:n if a_ev[j] > 0]
        push!(row_data[out_idx], (c, factors))
    end

    # Build flat CSR arrays
    term_ptr_flat     = Int32[1]
    term_coeffs_flat  = Int[]
    monomial_ptr_flat = Int32[1]
    mon_indices_flat  = Int32[]
    mon_degrees_flat  = Int32[]

    for i in 1:num_output
        for (c, factors) in row_data[i]
            push!(term_coeffs_flat, c)
            for (j, deg) in factors
                push!(mon_indices_flat, Int32(j))
                push!(mon_degrees_flat, Int32(deg))
            end
            push!(monomial_ptr_flat, Int32(length(mon_indices_flat) + 1))
        end
        push!(term_ptr_flat, Int32(length(term_coeffs_flat) + 1))
    end

    return (
        term_ptr_flat,
        term_coeffs_flat,
        monomial_ptr_flat,
        mon_indices_flat,
        mon_degrees_flat,
    )
end

# =============================================================================
# formula_pow_plan constructor  (bead w86 + rrn.2)
# =============================================================================
#
# n_vars:         number of variables in the original polynomial ring
# d:              degree of the input homogeneous polynomial
# pow:            the power to raise to
# backend:        KernelAbstractions backend (CPU(), CUDABackend(), etc.)
# workgroup_size: threshold for short/long row classification (default 256)
#
# original[i] must be the coefficient of the i-th monomial in
# with_replacement_combinations(1:n_vars, d) order.
function formula_pow_plan(n_vars::Int, d::Int, pow::Int, backend;
                          workgroup_size::Int = 256,
                          medium_workgroup_size::Int = 32)
    key = (n_vars, d, pow)

    # Compute (or retrieve from cache) the CPU-side CSR arrays.
    # The expensive Oscar symbolic expansion is skipped on cache hits.
    cached = get!(_formula_pow_cache, key) do
        n = binomial(n_vars + d - 1, d)
        poly, _ = generic_power_formula(n, pow)
        tp, tc, mp, mi, md = expressions_from_poly(poly, n_vars, d)

        num_rows    = length(tp) - 1
        row_lengths = [tp[i+1] - tp[i] for i in 1:num_rows]
        short_rows  = Int32[i for i in 1:num_rows if row_lengths[i] < medium_workgroup_size]
        medium_rows = Int32[i for i in 1:num_rows if medium_workgroup_size <= row_lengths[i] < workgroup_size]
        long_rows   = Int32[i for i in 1:num_rows if row_lengths[i] >= workgroup_size]

        # Sort tier arrays by ascending row length. The future K1 kernel
        # (1 thread/row, batched-only) places adjacent rows of the tier
        # array in adjacent threads of a warp; uniform lengths within a
        # warp avoid 40-60% divergence loss. K2/K3 (warp- or block-per-row)
        # are indifferent to this order. Output is keyed on output_row via
        # term_ptr, so sort never changes results.
        short_rows  = short_rows[sortperm(row_lengths[short_rows])]
        medium_rows = medium_rows[sortperm(row_lengths[medium_rows])]
        long_rows   = long_rows[sortperm(row_lengths[long_rows])]

        (term_ptr        = tp,
         term_coeffs     = tc,
         monomial_ptr    = mp,
         monomial_indices = mi,
         monomial_degrees = md,
         short_rows_cpu  = short_rows,
         medium_rows_cpu = medium_rows,
         long_rows_cpu   = long_rows)
    end

    return FormulaPowPlan(
        Adapt.adapt(backend, cached.term_ptr),
        Adapt.adapt(backend, cached.term_coeffs),
        Adapt.adapt(backend, cached.monomial_ptr),
        Adapt.adapt(backend, cached.monomial_indices),
        Adapt.adapt(backend, cached.monomial_degrees),
        Adapt.adapt(backend, cached.short_rows_cpu),
        Adapt.adapt(backend, cached.medium_rows_cpu),
        Adapt.adapt(backend, cached.long_rows_cpu),
        workgroup_size,
        medium_workgroup_size,
    )
end

# =============================================================================
# Long-row kernel  (bead rrn.3.2)
# =============================================================================
#
# 1 workgroup per output monomial (for rows with >= workgroup_size terms).
# ndrange = workgroup_size * length(long_rows)  (1D)
#
# Parallel tree reduction in shared memory:
#   1. Each thread accumulates its strided subset of terms into shmem[lid].
#   2. @synchronize unconditionally.
#   3. log2(WS) rounds of tree reduction; all threads hit every @synchronize.
#   4. Thread 1 writes shmem[1] to output.
#
# WS must be a power of 2.  All @synchronize calls are unconditional.
@kernel function formula_pow_long_kernel!(
        output,
        original,
        term_ptr,
        term_coeffs,
        monomial_ptr,
        monomial_indices,
        monomial_degrees,
        long_rows,
        ::Val{WS}) where WS
    # KA splits the kernel body at each @synchronize boundary into separate
    # workitem for loops.  Only @index assignments and @localmem survive across
    # phases — plain locals are scoped to their phase and lost afterward.
    # Rule: re-derive lid and row from gidx at the start of every phase.
    gidx = @index(Global, Linear)

    shmem = @localmem eltype(output) (WS,)

    # Phase 1: strided partial sum → shmem.
    lid_1  = (gidx - 1) % WS + 1
    row_1  = long_rows[(gidx - 1) ÷ WS + 1]
    rstart = term_ptr[row_1]
    rend   = term_ptr[row_1 + 1] - 1
    partial = zero(eltype(output))
    j = rstart + (lid_1 - 1)
    while j <= rend
        contrib = term_coeffs[j]
        for kk in monomial_ptr[j] : monomial_ptr[j+1]-1
            contrib *= original[monomial_indices[kk]] ^ monomial_degrees[kk]
        end
        partial += contrib
        j += WS
    end
    shmem[lid_1] = partial
    @synchronize()

    # Phase 2: tree reduction (WS must be a power of 2).
    # Re-derive lid_2 from gidx at each barrier phase inside the loop.
    # Cannot use `stride` as a variable name — it shadows Base.stride.
    for k in 1:trailing_zeros(WS)
        lid_2 = (gidx - 1) % WS + 1
        h = WS >> k
        if lid_2 <= h
            shmem[lid_2] += shmem[lid_2 + h]
        end
        @synchronize()
    end

    # Phase 3: thread 1 of each group writes the result.
    lid_3 = (gidx - 1) % WS + 1
    if lid_3 == 1
        output[long_rows[(gidx - 1) ÷ WS + 1]] = shmem[1]
    end
end

# =============================================================================
# formula_pow / formula_pow!  (bead rrn.4 + 63i)
# =============================================================================
#
# Compute the coefficients of (Σᵢ original[i]*mᵢ)^pow, where mᵢ are the
# degree-d input monomials in with_replacement_combinations order.
#
# formula_pow!(output, ...) writes into a caller-provided buffer for hot
# paths that want to reuse allocations. formula_pow(...) allocates and
# delegates to formula_pow!. Returns a backend array of length
# binomial(n_vars + d*pow - 1, d*pow) with output[i] = coefficient of the
# i-th monomial in with_replacement_combinations(1:n_vars, d*pow) order.
function formula_pow!(output, original, plan::FormulaPowPlan, backend)
    WS_M = plan.medium_workgroup_size
    WS_L = plan.workgroup_size

    # Single-op (B=1) routes short and medium tiers through the same
    # warp-per-row kernel at WS=M. The dedicated 1-thread/row short kernel
    # is latency-bound on small workloads (165–566 threads is well under
    # the in-flight capacity of a modern GPU); putting 32 threads on each
    # short row instead unlocks enough warps to hide memory stalls and
    # measured 1.8–3× faster on RTX 3070 across (4,4,2)/(4,8,3)/(4,4,6).
    # The 1-thread/row strategy is reserved for the batched path (K1),
    # where row count is large enough to be compute- rather than latency-bound.
    if !isempty(plan.short_rows)
        kernel = formula_pow_long_kernel!(backend, WS_M)
        kernel(output, original,
               plan.term_ptr, plan.term_coeffs,
               plan.monomial_ptr, plan.monomial_indices, plan.monomial_degrees,
               plan.short_rows, Val(WS_M);
               ndrange = WS_M * length(plan.short_rows))
    end

    if !isempty(plan.medium_rows)
        kernel = formula_pow_long_kernel!(backend, WS_M)
        kernel(output, original,
               plan.term_ptr, plan.term_coeffs,
               plan.monomial_ptr, plan.monomial_indices, plan.monomial_degrees,
               plan.medium_rows, Val(WS_M);
               ndrange = WS_M * length(plan.medium_rows))
    end

    if !isempty(plan.long_rows)
        kernel = formula_pow_long_kernel!(backend, WS_L)
        kernel(output, original,
               plan.term_ptr, plan.term_coeffs,
               plan.monomial_ptr, plan.monomial_indices, plan.monomial_degrees,
               plan.long_rows, Val(WS_L);
               ndrange = WS_L * length(plan.long_rows))
    end

    KernelAbstractions.synchronize(backend)
    return output
end

function formula_pow(original, plan::FormulaPowPlan, backend)
    num_out = length(plan.term_ptr) - 1
    # Uninitialized: the partition (short ∪ medium ∪ long) covers every output
    # row and each kernel writes its row once, so zero-init is redundant.
    # Invariant verified in test/FormulaPowTests.jl "plan partition invariant".
    output  = KernelAbstractions.allocate(backend, eltype(original), num_out)
    return formula_pow!(output, original, plan, backend)
end
