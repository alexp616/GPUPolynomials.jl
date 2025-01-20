abstract type OperationPlan end

struct EmptyPlan <: OperationPlan

end

struct GPUPowPlan <: OperationPlan
    key::Int
    len::Int
    nttPowPlans::Vector{NTTPowPlan}
    crtPlan::CuArray
    # memorysafe::Bool
end

