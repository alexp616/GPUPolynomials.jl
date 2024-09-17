module GPUNTT

include("../ntt_utils.jl")


mutable struct GPUNTTPregen{T<:Integer}
    primeArray::Vector{T}
    npruArray::Vector{T}
    len::Int
    log2len::Int
    butterfly::Vector{Int}
end

end