include("int128_stuff.jl")

@inline function sub_mod(x::Signed, y::Signed, m::Signed)
    return unchecked_mod(x - y, m)
end

@inline function sub_mod(x::Unsigned, y::Unsigned, m::Unsigned)
    if y > x
        return (m - y) + x
    else
        return x - y
    end
end

@inline function add_mod(x::Signed, y::Signed, m::Signed)
    result = x + y
    if result < 0
        return result + m
    elseif result >= m
        return result - m
    else
        return result
    end
    return
end

@inline function add_mod(x::Unsigned, y::Unsigned, m::Unsigned)
    result = x + y
    return result > m ? result - m : result
end

function mywiden(x)
    throw(MethodError(mywiden, (typeof(x),)))
end

macro generate_widen()
    int_types = [Int8, Int16, Int32, Int64, Int128, Int256, Int512]
    uint_types = [UInt8, UInt16, UInt32, UInt64, UInt128, UInt256, UInt512]

    widen_methods = quote end
    for i in 1:length(int_types) - 1
        push!(widen_methods.args, :(
            Base.@eval mywiden(x::$(int_types[i])) = $(int_types[i+1])(x)
        ))
        push!(widen_methods.args, :(
            Base.@eval mywiden(x::$(uint_types[i])) = $(uint_types[i+1])(x)
        ))
    end

    return widen_methods
end

@generate_widen()

function mywidemul(x, y)
    return mywiden(x) * mywiden(y)
end

@inline function mul_mod(x::T, y::T, m::T) where T<:Unsigned
    return T(unchecked_mod(mywidemul(x, y), m))
end

function crt(vec, pregen)
    x = eltype(pregen)(vec[1])
    # @cuprintln(x)
    for i in axes(pregen, 2)
        a = mul_mod(x, pregen[2, i], pregen[3, i])
        b = mul_mod(eltype(pregen)(vec[i + 1]), pregen[1, i], pregen[3, i])
        x = add_mod(a, b, pregen[3, i])
        # @cuprintln(x)
    end

    return x
end

function get_fft_size(vec::Union{Vector, CuVector}, pow)
    finalLength = (length(vec) - 1) * pow + 1
    return Base._nextpow2(finalLength)
end

function get_fft_size(veclength::Int, pow)
    finalLength = (veclength - 1) * pow + 1
    return Base._nextpow2(finalLength)
end

function pregen_crt(primeArray::Vector{T}) where T<:Integer
    # There really shouldn't be any overflow behavior, but 
    # I'm doing it in BigInt just to be safe. This is all for
    # the pregeneration step anyways.
    primeArray = BigInt.(primeArray)

    result = zeros(BigInt, 3, length(primeArray) - 1)

    currmod = primeArray[1]
    for i in 2:length(primeArray)
        m1, m2 = extended_gcd_iterative(currmod, primeArray[i])
        result[1, i - 1] = m1 * currmod
        currmod *= primeArray[i]
        result[2, i - 1] = m2 * primeArray[i]
        result[3, i - 1] = currmod
        if result[1, i - 1] < 0
            result[1, i - 1] += currmod
        end
        if result[2, i - 1] < 0
            result[2, i - 1] += currmod
        end
    end

    @assert all([i > 0 for i in result]) display(result)
    return T.(result)
end

function extended_gcd_iterative(a::T, b::T) where T<:Signed
    x0, x1 = T(1), T(0)
    y0, y1 = T(0), T(1)
    while b != 0
        q, r = divrem(a, b)
        a, b = b, r
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    end
    @assert a == 1 "$a and $b aren't coprime"
    return x0, y0
end

function power_mod(n::T, p::Int, m::T) where T<:Integer
    result = eltype(n)(1)
    base = unchecked_mod(n, m)

    while p > 0
        if p & 1 == 1
            result = mul_mod(result, base, m)
        end
        base = mul_mod(base, base, m)
        p = p >> 1
    end

    return result
end

function mod_inverse(n::Integer, p::Integer)
    n = BigInt(n)
    p = BigInt(p)
    n = mod(n, p)

    t, new_t = 0, 1
    r, new_r = p, n

    while new_r != 0
        quotient = r ÷ new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    end

    return t < 0 ? typeof(n)(t + p) : typeof(n)(t)
end

function nth_principal_root_of_unity(n::Integer, p::Integer)
    @assert mod(p - 1, n) == 0 "n must divide p-1"

    order = (p - 1) ÷ n

    function is_primitive_root(g, p, order)
        for i in 1:(n-1)
            if powermod(g, i * order, p) == 1
                return false
            end
        end
        return true
    end
    
    g = 2
    while !is_primitive_root(g, p, order)
        g += 1
    end

    root_of_unity = powermod(g, order, p)
    return typeof(p)(root_of_unity)
end

function npruarray_generator(primearray::Array{<:Integer}, n)
    return map(p -> nth_principal_root_of_unity(n, p), primearray)
end

function parallel_bit_reverse_copy(p)
    @assert ispow2(length(p)) "p must be an array with length of a power of 2"
    len = length(p)
    result = CUDA.zeros(eltype(p), len)
    log2n = Int(log2(len))

    function kern(p, dest, len, log2n)
        idx1 = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
        idx2 = idx1 + Int(len / 2)
    
        rev1 = bit_reverse(idx1, log2n)
        rev2 = bit_reverse(idx2, log2n)
    
        dest[idx1 + 1] = p[rev1 + 1]
        dest[idx2 + 1] = p[rev2 + 1]
        return nothing
    end

    kernel = @cuda launch = false kern(p, result, len, log2n)
    config = launch_configuration(kernel.fun)
    threads = min(len ÷ 2, prevpow(2, config.threads))
    blocks = cld(len ÷ 2, threads)

    kernel(p, result, len, log2n; threads = threads, blocks = blocks)
    
    return result
end

function bit_reverse(x::Integer, log2n::Integer)
    temp = 0
    for i in 0:log2n-1
        temp <<= 1
        temp |= (x & 1)
        x >>= 1
    end
    return temp
end

function generate_butterfly_permutations(n::Int)
    @assert ispow2(n) "n must be a power of 2"
    perm = parallel_bit_reverse_copy(CuArray([i for i in 1:n]))
    if n >= 134217728
        return Array(perm)
    else 
        return perm
    end
end

function find_ntt_primes(n)
    prime_list = []
    k = 1
    while length(prime_list) < 10
        candidate = k * n + 1
        if isprime(candidate)
            push!(prime_list, candidate)
        end
        k += 1
    end

    return prime_list
end

function get_types(primeArray::Vector{<:Unsigned})
    max = BigInt(maximum(primeArray))
    nttType = get_uint_type(Base._nextpow2(Int(ceil(log2(max + 1)))))

    totalprod = prod(BigInt.(primeArray))
    crtType = get_uint_type(Base._nextpow2(Int(ceil(log2(totalprod + 1)))))
    resultType = get_uint_type(Base._nextpow2(Int(ceil(log2(totalprod + 1)))))
    
    return nttType, crtType, resultType
end

# function get_types(primeArray::Vector{<:Signed})
#     max = BigInt(maximum(primeArray))
#     nttType = get_int_type(Base._nextpow2(1 + Int(ceil(log2(max^2 + 1)))))

#     totalprod = prod(BigInt.(primeArray))
#     crtType = get_int_type(Base._nextpow2(1 + Int(ceil(log2(totalprod^2 + 1)))))
#     resultType = get_int_type(Base._nextpow2(1 + Int(ceil(log2(totalprod + 1)))))

#     return nttType, crtType, resultType
# end
