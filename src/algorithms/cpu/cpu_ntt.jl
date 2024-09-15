module CPUNTT

include("../ntt_utils.jl")

export CPUNTTPregen, CPUINTTPregen, pregen_ntt, pregen_intt, cpu_ntt!, cpu_intt!

mutable struct CPUNTTPregen{}
    primeArray::Vector{<:Unsigned}
    npruArray::Vector{<:Unsigned}
    len::Int
    log2len::Int
    butterfly::Vector
end

mutable struct CPUINTTPregen
    nttpregen::CPUNTTPregen
    lenInverseArray::Vector{<:Unsigned}
end

function pregen_ntt(primeArray::Vector{<:Signed}, len)
    return pregen_ntt(unsigned(eltype(primeArray)).(primeArray), len)
end

function pregen_ntt(primeArray::Vector{<:Unsigned}, len)
    if !ispow2(len)
        throw(ArgumentError("len must be a power of 2."))
    end

    type = eltype(primeArray)

    primeArray = type.(primeArray)

    butterfly = Array(generate_butterfly_permutations(len))

    lenInverseArray = type.(map(p -> mod_inverse(len, p), primeArray))
    # println("lenInverseArray: $(Int.(lenInverseArray))")
    log2len = Int(log2(len))

    npruArray = npruarray_generator(primeArray, len)
    # println("npruArray: $(Int.(npruArray))")

    @assert eltype(primeArray) <: Unsigned
    @assert eltype(npruArray) <: Unsigned

    nttpregen = CPUNTTPregen(primeArray, npruArray, len, log2len, butterfly)

    npruInverseArray = type.(mod_inverse.(npruArray, primeArray))

    @assert eltype(npruInverseArray) <: Unsigned

    temp = CPUNTTPregen(primeArray, npruInverseArray, len, log2len, butterfly)
    inttpregen = CPUINTTPregen(temp, lenInverseArray)

    return nttpregen, inttpregen
end

function pregen_intt(nttpregen::CPUNTTPregen)
    lenInverseArray = map(p -> mod(nttpregen.len, p), nttpregen.primeArray)
    return CPUINTTPregen(nttpregen, lenInverseArray)
end

function cpu_ntt!(stackedvec::Array{<:Unsigned}, pregen::CPUNTTPregen)
    if size(stackedvec, 1) != pregen.len
        throw(ArgumentError("Vector doesn't have same length as pregen object. Vector length: $(size(stackedvec, 1)), pregen.len: $(pregen.len)"))
    end
    stackedvec .= stackedvec[pregen.butterfly, :]

    lenover2 = pregen.len >> 1
    for i in 1:pregen.log2len
        m = 1 << i
        m2 = m >> 1
        magic = 1 << (pregen.log2len - i)
        theta_m = power_mod.(pregen.npruArray, pregen.len ÷ m, pregen.primeArray)

        for idx in 0:lenover2 - 1
            k = m * mod(idx, magic) + idx ÷ magic
            # k = Int(2 * m2 * (idx % magic) + floor(idx/magic))
            # println("k + 1: $(k + 1), k + m2 + 1: $(k + m2 + 1)")
            for p in eachindex(pregen.primeArray)
                theta = power_mod(theta_m[p], idx ÷ magic, pregen.primeArray[p])
                t = theta * stackedvec[k + m2 + 1, p]
                u = stackedvec[k + 1, p]

                if t > u
                    a = pregen.primeArray[p] - mod(t - u, pregen.primeArray[p])
                else
                    a = mod(u - t, pregen.primeArray[p])
                end

                stackedvec[k + 1, p] = mod(u + t, pregen.primeArray[p])
                stackedvec[k + m2 + 1, p] = mod(a, pregen.primeArray[p])
            end
        end
    end
end

function cpu_intt!(stackedvec::Array{T}, pregen::CPUINTTPregen) where T<:Unsigned
    if size(stackedvec, 1) != pregen.nttpregen.len
        throw(ArgumentError("Vector doesn't have same length as pregen object."))
    end

    cpu_ntt!(stackedvec, pregen.nttpregen)
    for i in 1:length(pregen.nttpregen.primeArray)
        stackedvec[:, i] .*= pregen.lenInverseArray[i]
        stackedvec[:, i] .= map(x -> mod(x, pregen.nttpregen.primeArray[i]), stackedvec[:, i])
    end
    
end
end
