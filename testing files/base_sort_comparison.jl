using CUDA
using Statistics

function run_benchmarks()
    len = 10000
    for _ in 1:4
        cputimes = []
        gputimes = []
        for i in 1:10
            A = cu(rand(1:len, len); unified=true)
            B = unsafe_wrap(Array, A)
            sfsf = CUDA.@timed sort!(B)
            push!(cputimes, sfsf.time)
            @assert issorted(Array(A))

            C = CuArray(rand(1:len, len))
            dfdf = CUDA.@timed sort!(C)
            push!(gputimes, dfdf.time)
        end

        println("average time to sort $len elements on the CPU with unified memory: ")
        println("$(mean(cputimes)) s")

        println("average time to sort $len elements on the GPU")
        println("$(mean(gputimes)) s")
        len *= 10
    end
end

run_benchmarks()