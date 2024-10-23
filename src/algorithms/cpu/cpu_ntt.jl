module CPUNTT

include("../../utils/ntt_utils.jl")

export CPUNTTPregen, CPUINTTPregen, pregen_ntt, pregen_intt, cpu_ntt!, cpu_intt!

mutable struct CPUNTTPregen{T<:Integer}
    primeArray::Vector{T}
    npruArray::Vector{T}
    thetaArray::Array{T}
    len::Int
    log2len::Int
    butterfly::Vector{Int}
end

mutable struct CPUINTTPregen{T<:Integer}
    nttpregen::CPUNTTPregen{T}
    lenInverseArray::Vector{T}
end

function pregen_ntt(primeArray::Vector{<:Integer}, len)
    if !ispow2(len)
        throw(ArgumentError("len must be a power of 2."))
    end

    butterfly = Array(generate_butterfly_permutations(len))

    lenInverseArray = map(p -> mod_inverse(len, p), primeArray)
    @assert all([i > 0 for i in lenInverseArray])
    log2len = Int(log2(len))

    npruArray = npruarray_generator(primeArray, len)
    @assert all([i > 0 for i in npruArray])
    thetaArray = generate_theta_m(primeArray, len, log2len, npruArray)
    @assert all([i > 0 for i in thetaArray])
    nttpregen = CPUNTTPregen{eltype(primeArray)}(primeArray, npruArray, thetaArray, len, log2len, butterfly)

    npruInverseArray = eltype(primeArray).(mod_inverse.(npruArray, primeArray))
    @assert all([i > 0 for i in npruInverseArray])

    inverseThetaArray = generate_theta_m(primeArray, len, log2len, npruInverseArray)
    @assert all([i > 0 for i in npruInverseArray])
    temp = CPUNTTPregen{eltype(primeArray)}(primeArray, npruInverseArray, inverseThetaArray, len, log2len, butterfly)
    inttpregen = pregen_intt(temp)

    return nttpregen, inttpregen
end

function pregen_intt(nttpregen::CPUNTTPregen{T}) where T<:Integer
    lenInverseArray = T.(map(p -> mod_inverse(nttpregen.len, p), nttpregen.primeArray))
    return CPUINTTPregen{T}(nttpregen, lenInverseArray)
end

function generate_theta_m(primeArray, len, log2len, npruArray)
    result = zeros(eltype(primeArray), length(primeArray), log2len)
    for i in 1:log2len
        m = 1 << i
        result[:, i] = power_mod.(npruArray, len ÷ m, primeArray)
    end

    return result
end

function cpu_ntt!(stackedvec::Array{T}, pregen::CPUNTTPregen{T}) where T<:Integer
    if size(stackedvec, 1) != pregen.len
        throw(ArgumentError("Vector doesn't have same length as pregen object. Vector length: $(size(stackedvec, 1)), pregen.len: $(pregen.len)"))
    end
    stackedvec .= stackedvec[pregen.butterfly, :]

    lenover2 = pregen.len >> 1
    for i in 1:pregen.log2len
        m = 1 << i
        m2 = m >> 1
        magicbits = pregen.log2len - i
        magicmask = (1 << magicbits) - 1
        
        # theta_m = power_mod.(pregen.npruArray, pregen.len ÷ m, pregen.primeArray)
        theta_m = view(pregen.thetaArray, :, i)
        for idx in 0:lenover2 - 1
            k = m * (idx & magicmask) + (idx >> magicbits)

            for p in eachindex(pregen.primeArray)
                # theta = power_mod(theta_m[p], idx ÷ magic, pregen.primeArray[p])
                theta = power_mod(theta_m[p], idx >> magicbits, pregen.primeArray[p])
                t = theta * stackedvec[k + m2 + 1, p]
                u = stackedvec[k + 1, p]

                stackedvec[k + 1, p] = unchecked_mod(u + t, pregen.primeArray[p])
                stackedvec[k + m2 + 1, p] = sub_mod(u, t, pregen.primeArray[p])
            end
        end
    end
end

function cpu_intt!(stackedvec::Array{T}, pregen::CPUINTTPregen{T}) where T<:Integer
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
