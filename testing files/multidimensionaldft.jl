using Test
include("..\\univariate algorithms\\CPUUnivariateAlgorithms.jl")

# CUDA.jl's only data structure is the CuArray, so everything is implemented
# with arrays

# 1 + x
polynomial1 = [
    1 1
    2 0
    3 0
]

# 1 + y
polynomial2 = [
    1 0 2
    1 0 4
]

function print_matrix(matrix)
    for i in 1:size(matrix, 1)
        for j in 1:size(matrix, 2)
            print(matrix[i, j], "\t")
        end
        println()
    end
end

"""
    getHighestDegrees(polynomial)

Return the highest x-degree and highest y-degree in the polynomial
represented by the matrix

Assumes user isn't putting in an empty array
"""
function getHighestDegrees(polynomial)
    greatestx = 0
    greatesty = 0
    rowzeros = zeros(Int, size(polynomial, 2))
    colzeros = zeros(Int, size(polynomial, 1))

    for i in size(polynomial, 2):-1:1
        if polynomial[:, i] != colzeros
            greatestx = i
            break
        end
    end

    for j in size(polynomial, 1):-1:1
        if polynomial[j, :] != rowzeros
            greatesty = j
            break
        end
    end

    return (greatestx - 1, greatesty - 1)
end


function padMatrix(matrix, r, c)
    m = ComplexF32.(matrix)
    function addZeroRows(matrix, n)
        num_cols = size(matrix, 2)
        zerosr = zeros(ComplexF32, n, num_cols)
        return vcat(matrix, zerosr)
    end

    function addZeroCols(matrix, m)
        num_rows = size(matrix, 1)
        zerosc = zeros(ComplexF32, num_rows, m)
        return hcat(matrix, zerosc)
    end

    println("type of m: $(typeof(m))")
    return addZeroCols(addZeroRows(m, r), c)
end

function DFT2d(input::AbstractArray, rows, log2rows, cols, log2cols, inverted = 1)
    # TODO for every vector v in dim1, replace with DFT1d(v)
    if inverted == 1
        for i in 1:size(input, 2)  # Iterate over columns
            input[:, i] .= CPUDFT(input[:, i], rows, log2rows)
        end

        for i in 1:size(input, 1)  # Iterate over rows
            input[i, :] .= CPUDFT(input[i, :], cols, log2cols)
        end
    end

    if inverted == -1
        for i in 1:size(input, 2)  # Iterate over columns
            input[:, i] .= CPUIDFT(input[:, i], rows, log2rows)
        end

        for i in 1:size(input, 1)  # Iterate over rows
            input[i, :] .= CPUIDFT(input[i, :], cols, log2cols)
        end
    end

    return input
end

function IDFT2d(input::AbstractArray, rows, log2rows, cols, log2cols)
    return DFT2d(input, rows, log2rows, cols, log2cols, -1)
end

function multiply2d(polynomial1, polynomial2)
    # Figure out dimensions of resulting product, pad to 2^n power

    # resultSize stores (col, row)
    resultSize = getHighestDegrees(polynomial1) .+ getHighestDegrees(polynomial2) .+ 1

    rows = Int.(2^ceil(log2(resultSize[2])))
    cols = Int.(2^ceil(log2(resultSize[1])))

    # Scale polynomial1 and polynomial2 dimensions to 2^n power
    copyp1 = padMatrix(polynomial1, rows - size(polynomial1, 1), cols - size(polynomial1, 2))
    copyp2 = padMatrix(polynomial2, rows - size(polynomial2, 1), cols - size(polynomial2, 2))

    log2rows = UInt32(log2(rows));
    log2cols = UInt32(log2(cols));

    # DFT polynomial1 and polynomial2
    y1 = DFT2d(copyp1, rows, log2rows, cols, log2cols)
    y2 = DFT2d(copyp2, rows, log2rows, cols, log2cols)

    # Perform element-wise multiplication -> result
    result = IDFT2d(y1 .* y2, rows, log2rows, cols, log2cols)
    # IDFT result, broadcast dividing by dimensions
    return Int.(round.(real.(result[1:resultSize[2], 1:resultSize[1]])))
end

print_matrix(multiply2d(polynomial1, polynomial2))



testpoly = [
    1 0 0 1 0
    1 0 1 0 0
    1 0 0 0 0
]

@test getHighestDegrees(testpoly) == (3, 2)

