struct NTTMulPlan{T<:Integer}
    forwardPlan::NTTPlan{T}
    invPlan::INTTPlan{T}

    function NTTMulPlan(len::Int, prime::T) where T<:Integer
        npru = primitive_nth_root_of_unity(len, prime)
        forwardPlan, invPlan = plan_ntt(len, prime, npru)

        return new{T}(forwardPlan, invPlan)
    end
end

Base.eltype(::Type{NTTMulPlan{T}}) where T = T

function ntt_mul!(svec1::CuArray{T}, svec2::CuArray{T}, plan::NTTMulPlan{T}) where T<:Integer
    @assert length(svec1) == length(svec2)

    reducer = plan.forwardPlan.reducer

    ntt!(svec1, plan.forwardPlan, true)
    ntt!(svec2, plan.forwardPlan, true)

    componentwise_mul!(svec1, svec2, reducer)
    
    intt!(svec1, plan.invPlan, true)

    return 
end

function componentwise_mul!(vec1::CuVector{T}, vec2::CuVector{T}, reducer::CudaNTTs.Reducer{T}) where T<:Integer
    kernel = @cuda launch=false componentwise_mul_kernel!(vec1, vec2, reducer)
    config = launch_configuration(kernel.fun)
    threads = min(length(vec1), Base._prevpow2(config.threads))
    blocks = div(length(vec1), threads)

    kernel(vec1, vec2, reducer; threads = threads, blocks = blocks)

    return 
end

@inbounds function componentwise_mul_kernel!(vec1::CuDeviceVector{T}, vec2::CuDeviceVector{T}, reducer::CudaNTTs.Reducer{T}) where T<:Integer
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    vec1[idx] = CudaNTTs.mul_mod(vec1[idx], vec2[idx], reducer)

    return
end