# Fast Int128/UInt128 division and modulo using 64-bit arithmetic.
#
# CUDA lacks the __modti3/__divti3 intrinsics for 128-bit division,
# so we implement division using Knuth's Algorithm D with 32-bit
# digits and 64-bit intermediates. This replaces the previous
# 128-iteration bit-by-bit long division with ~4 native 64-bit divs.

# ──────────────────────────────────────────────────────────────────
# Core: normalized 128÷64 division (Knuth Algorithm D, 2-digit case)
# ──────────────────────────────────────────────────────────────────

const _MASK32 = UInt64(0xFFFFFFFF)
const _BASE32 = UInt64(1) << 32

"""
    divrem_128by64_normalized(n_hi, n_lo, d) -> (quotient, remainder)

Divide the 128-bit unsigned integer `(n_hi << 64) | n_lo` by `d`.

Preconditions (caller must guarantee):
- `d` has its MSB set (i.e., `d >= 2^63`)
- `n_hi < d` (so the quotient fits in 64 bits)

Uses Knuth Algorithm D with 32-bit digits: splits `d` into two 32-bit
halves and produces two 32-bit quotient digits via 64-bit division,
each with an at-most-2-iteration correction loop.
"""
@inline function divrem_128by64_normalized(n_hi::UInt64, n_lo::UInt64, d::UInt64)
    d1 = d >>> 32           # high 32 bits of d (>= 2^31 since MSB of d is set)
    d0 = d & _MASK32        # low 32 bits of d

    # ── Round 1: compute high 32 bits of quotient ──
    # Divide (n_hi : upper32(n_lo)) by d1 to estimate q1
    q1 = n_hi ÷ d1
    r1 = n_hi - q1 * d1     # remainder, fits in 64 bits

    # Correct overestimate: q1 might be too large by at most 2
    # Check: q1 * d0 > r1 * 2^32 + upper32(n_lo)?
    while q1 >= _BASE32 || q1 * d0 > (r1 << 32) | (n_lo >>> 32)
        q1 -= UInt64(1)
        r1 += d1
        r1 >= _BASE32 && break  # r1 overflowed 32 bits, no more corrections needed
    end

    # Compute partial remainder after subtracting q1 * d from upper 96 bits
    # partial_rem = (n_hi:n_lo) - q1 * d * 2^32, keeping only the relevant 64 bits
    partial_rem = (n_hi << 32) | (n_lo >>> 32) - q1 * d

    # ── Round 2: compute low 32 bits of quotient ──
    # Divide (partial_rem : lower32(n_lo)) by d1 to estimate q0
    q0 = partial_rem ÷ d1
    r0 = partial_rem - q0 * d1

    while q0 >= _BASE32 || q0 * d0 > (r0 << 32) | (n_lo & _MASK32)
        q0 -= UInt64(1)
        r0 += d1
        r0 >= _BASE32 && break
    end

    remainder = ((partial_rem << 32) | (n_lo & _MASK32)) - q0 * d
    quotient = (q1 << 32) | q0

    return (quotient, remainder)
end

# ──────────────────────────────────────────────────────────────────
# Wrapper: general 128÷64 (normalizes, handles n_hi >= d)
# ──────────────────────────────────────────────────────────────────

"""
    udivrem_128by64(n_hi, n_lo, d) -> (q_hi, q_lo, remainder)

Divide the 128-bit unsigned integer `(n_hi << 64) | n_lo` by a nonzero
64-bit `d`. Returns the quotient as `(q_hi, q_lo)` and the remainder.
"""
@inline function udivrem_128by64(n_hi::UInt64, n_lo::UInt64, d::UInt64)
    # Fast path: high half is zero
    if n_hi == UInt64(0)
        q_lo = n_lo ÷ d
        rem = n_lo - q_lo * d
        return (UInt64(0), q_lo, rem)
    end

    # If n_hi >= d, first extract the high quotient digit
    q_hi = UInt64(0)
    if n_hi >= d
        q_hi = n_hi ÷ d
        n_hi = n_hi - q_hi * d
    end

    # Now n_hi < d. Normalize d (shift left until MSB is set).
    s = leading_zeros(d)

    if s > 0
        d_n = d << s
        # Shift the 128-bit (n_hi, n_lo) left by s bits
        n_hi_n = (n_hi << s) | (n_lo >>> (UInt64(64) - UInt64(s)))
        n_lo_n = n_lo << s
    else
        d_n = d
        n_hi_n = n_hi
        n_lo_n = n_lo
    end

    q_lo, rem_n = divrem_128by64_normalized(n_hi_n, n_lo_n, d_n)

    # Un-normalize remainder
    rem = rem_n >>> s

    return (q_hi, q_lo, rem)
end

# ──────────────────────────────────────────────────────────────────
# General: 128÷128 unsigned division
# ──────────────────────────────────────────────────────────────────

"""
    udivrem128(n, d) -> (quotient, remainder)

Divide two `UInt128` values. Returns `(n ÷ d, n % d)`.
"""
@inline function udivrem128(n::UInt128, d::UInt128)
    d_hi = UInt64(d >>> 64)
    d_lo = UInt64(d & typemax(UInt64))
    n_hi = UInt64(n >>> 64)
    n_lo = UInt64(n & typemax(UInt64))

    # Case A: divisor fits in 64 bits → use fast 128÷64
    if d_hi == UInt64(0)
        q_hi, q_lo, rem = udivrem_128by64(n_hi, n_lo, d_lo)
        quotient = (UInt128(q_hi) << 64) | UInt128(q_lo)
        return (quotient, UInt128(rem))
    end

    # Case B: divisor >= 2^64 → quotient fits in 64 bits
    # Use leading-zero shift to approximate quotient

    if n_hi == UInt64(0) && d_hi != UInt64(0)
        # n < 2^64 <= d, so quotient is 0
        return (UInt128(0), n)
    end

    s = leading_zeros(d_hi)

    if s == 0
        # d >= 2^127, quotient is 0 or 1
        q = n >= d ? UInt128(1) : UInt128(0)
        return (q, n - q * d)
    end

    # Shift both n and d right by (64 - s) so d fits in ~64 bits.
    # This gives an approximate quotient that's at most 1 too large.
    shift = UInt64(64) - UInt64(s)
    d_approx = UInt64((d >>> shift) & typemax(UInt64))

    # n >>> shift as a 128-bit value, but quotient fits in 64 bits
    # so we can use 128÷64
    _, q_lo, _ = udivrem_128by64(
        UInt64((n >>> (shift + UInt64(64))) & typemax(UInt64)),
        UInt64((n >>> shift) & typemax(UInt64)),
        d_approx
    )
    q = UInt128(q_lo)

    # q might be 1 too large — correct
    product = q * d
    if product > n
        q -= UInt128(1)
        product -= d
    end

    return (q, n - product)
end

# ──────────────────────────────────────────────────────────────────
# Public API: unchecked_mod / unchecked_div for UInt128
# ──────────────────────────────────────────────────────────────────

# Hot path: 128-bit mod 64-bit (the common case in mul_mod)
@inline function unchecked_mod(x::UInt128, m::UInt64)
    n_hi = UInt64(x >>> 64)
    n_lo = UInt64(x & typemax(UInt64))
    _, _, rem = udivrem_128by64(n_hi, n_lo, m)
    return UInt128(rem)
end

@inline function unchecked_mod(x::UInt128, m::Integer)
    _, rem = udivrem128(x, UInt128(m))
    return rem
end

@inline function unchecked_div(x::UInt128, m::UInt128)
    q, _ = udivrem128(x, m)
    return q
end

# ──────────────────────────────────────────────────────────────────
# Public API: unchecked_mod / unchecked_div for Int128
# ──────────────────────────────────────────────────────────────────

@inline function unchecked_mod(x::Int128, m::Integer)
    # Julia mod: result has sign of divisor (floored division)
    m128 = Int128(m)
    x_neg = x < Int128(0)
    m_neg = m128 < Int128(0)

    xu = x_neg ? reinterpret(UInt128, -x) : reinterpret(UInt128, x)
    mu = m_neg ? reinterpret(UInt128, -m128) : reinterpret(UInt128, m128)

    _, rem_u = udivrem128(xu, mu)
    rem = reinterpret(Int128, rem_u)

    if x_neg
        rem = -rem
    end

    # Adjust for Julia mod semantics: result has sign of divisor
    if rem != Int128(0) && (rem < Int128(0)) != m_neg
        rem += m128
    end

    return rem
end

@inline function unchecked_div(x::Int128, m::Int128)
    # Julia div: truncated toward zero
    x_neg = x < Int128(0)
    m_neg = m < Int128(0)

    xu = x_neg ? reinterpret(UInt128, -x) : reinterpret(UInt128, x)
    mu = m_neg ? reinterpret(UInt128, -m) : reinterpret(UInt128, m)

    q, _ = udivrem128(xu, mu)
    result = reinterpret(Int128, q)

    return (x_neg != m_neg) ? -result : result
end
