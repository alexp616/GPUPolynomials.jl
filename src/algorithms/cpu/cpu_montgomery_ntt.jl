module CPUMontgomeryNTT

include("../../utils/montgomery_reduction.jl")

mutable struct CPUMontNTTPregen
    montPrimeArray::Vector{MontReducer}
    npruArray::Vector{<:Unsigned}
    len::Int
    log2len::Int
    butterfly::Vector{Int}
end

mutable struct CPUMontINTTPregen
    nttpregen::CPUMontNTTPregen
    lenInverseArray::Vector{<:Unsigned}
end

function pregen_cpu_mont_ntt(primeArray::Vector{Integer}, len)
    if !ispow2(len)
        throw(ArgumentError("len must be a power of 2."))
    end

    

    butterfly = Array(generate_butterfly_permutations(len))
end

end