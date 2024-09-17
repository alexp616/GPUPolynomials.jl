include("cpu_ntt.jl")

module CPUNTTPow

include("../ntt_utils.jl")
using ..CPUNTT

export CPUPowPregen, get_fft_size, pregen_cpu_pow, cpu_ntt_pow

mutable struct CPUPowPregen
    primeArray::Vector{<:Integer}
    nttpregen::CPUNTTPregen
    inttpregen::CPUINTTPregen
    crtpregen::Array{<:Integer}
    resultType::DataType
end

function get_fft_size(vec, pow)
    finalLength = (length(vec) - 1) * pow + 1
    return Base._nextpow2(finalLength)
end

function get_types(primeArray::Vector{<:Unsigned})
    max = BigInt(maximum(primeArray))
    nttType = get_uint_type(Base._nextpow2(Int(ceil(log2(max^2 + 1)))))

    totalprod = prod(BigInt.(primeArray))
    crtType = get_uint_type(Base._nextpow2(Int(ceil(log2(totalprod^2 + 1)))))
    resultType = get_uint_type(Base._nextpow2(Int(ceil(log2(totalprod + 1)))))
    
    return nttType, resultType, crtType
end

function get_types(primeArray::Vector{<:Signed})
    max = BigInt(maximum(primeArray))
    nttType = get_int_type(Base._nextpow2(1 + Int(ceil(log2(max^2 + 1)))))

    totalprod = prod(BigInt.(primeArray))
    crtType = get_int_type(Base._nextpow2(1 + Int(ceil(log2(totalprod^2 + 1)))))
    resultType = get_int_type(Base._nextpow2(1 + Int(ceil(log2(totalprod + 1)))))

    return nttType, crtType, resultType
end

function pregen_cpu_pow(primeArray::Vector{T}, fftSize) where T<:Integer
    pregentime = @timed begin
        nttType, crtType, resultType = get_types(primeArray)
        nttpregen, inttpregen = pregen_ntt(nttType.(primeArray), fftSize)
        crtpregen = pregen_crt(crtType.(primeArray))
    end
    println("Entire pregeneration took $(pregentime.time) s")

    return CPUPowPregen(primeArray, nttpregen, inttpregen, crtpregen, resultType)
end

function cpu_ntt_pow(vec::Vector{<:Integer}, pow::Int; pregen::Union{CPUPowPregen, Nothing} = nothing, docrt = true)
    finalLength = (length(vec) - 1) * pow + 1

    if eltype(vec) != eltype(pregen.primeArray)
        println("Casting vec to primeArray type...")
        vec = eltype(pregen.primeArray).(vec)
    end

    if pregen === nothing
        throw(ArgumentError("Default pregeneration has not been implemented yet."))
    end

    multimodvec = repeat(vcat(vec, zeros(eltype(vec), pregen.nttpregen.len - length(vec))), 1, length(pregen.primeArray))

    cpu_ntt!(multimodvec, pregen.nttpregen)

    broadcast_pow!(multimodvec, pregen.primeArray, pow)
    cpu_intt!(multimodvec, pregen.inttpregen)

    if length(pregen.primeArray) == 1
        return pregen.resultType.(multimodvec)[1:finalLength]
    elseif docrt
        return build_result(multimodvec, pregen.crtpregen, finalLength, pregen.resultType)
    else
        return eltype(pregen.crtpregen).(multimodvec[1:finalLength])
    end
end

function build_result(multimodvec, crtpregen, finalLength, resultType)::Vector
    result = zeros(eltype(crtpregen), finalLength)
    zerovec = zeros(eltype(multimodvec), size(multimodvec, 2))
    for row in 1:finalLength
        if multimodvec[row, :] == zerovec
            result[row] = 0
        else
            result[row] = crt(view(multimodvec, row, :), crtPregen)
        end
    end

    return resultType.(result)
end

function broadcast_pow!(multimodvec, primeArray, pow)
    @inbounds begin
        for p in axes(multimodvec, 2)
            prime = primeArray[p]
            for i in axes(multimodvec, 1)
                multimodvec[i, p] = power_mod(multimodvec[i, p], pow, prime)
            end
        end
    end

    return nothing
end

end