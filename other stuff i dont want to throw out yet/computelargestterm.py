import time

def weak_compositions(n, k):
    def generate_compositions(n, k, start=0):
        if k == 1:
            yield [n]
        else:
            for i in range(n + 1):
                for comp in generate_compositions(n - i, k - 1, i):
                    yield [i] + comp
    
    return list(generate_compositions(n, k))

def compute_largest_term(resultTuple, inputCompositions):
    result = 0
    count = 1
    for composition in inputCompositions:
        if all(x >= y for x, y in zip(resultTuple, composition)):
            result += compute_largest_term_subproblem([x - y for x, y in zip(resultTuple, composition)], inputCompositions)
        print(f"{count} of {len(inputCompositions)} done")
        count += 1
    
    return result

def compute_largest_term_subproblem(resultTuple, inputCompositions):
    if all(element == 0 for element in resultTuple):
        return 1

    result = 0
    for composition in inputCompositions:
        if all(x >= y for x, y in zip(resultTuple, composition)):
            result += compute_largest_term_subproblem([x - y for x, y in zip(resultTuple, composition)], inputCompositions)

    return result
# Example usage:
n = 16
k = 4
compositions = weak_compositions(n, k)

start_time = time.time()
sus = compute_largest_term([20, 20, 20, 20], compositions)
print(f"largest term: {sus}")
end_time = time.time()
elapsed_time = start_time - end_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
