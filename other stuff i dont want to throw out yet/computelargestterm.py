import time

def weak_compositions(n, k):
    def generate_compositions(n, k, start=0):
        if k == 1:
            yield (n,)
        else:
            for i in range(n + 1):
                for comp in generate_compositions(n - i, k - 1, i):
                    yield (i,) + comp
    
    return list(generate_compositions(n, k))

def clt_helper(resultTuple, inputCompositions):
    from collections import defaultdict

    dp = defaultdict(int)
    dp[(0,) * len(resultTuple)] = 1

    # Not idiot-proof
    while(True):
        for state in list(dp.keys()):
            for composition in inputCompositions:
                new_state = tuple(x + y for x, y in zip(state, composition))
                if all(x >= y for x, y in zip(resultTuple, new_state)):
                    dp[new_state] += dp[state]
            del dp[state]
        if resultTuple in dp:
            return dp[resultTuple]

def compute_largest_term_firststep(numVars, prime):
    scale = (prime - 1) ** (prime - 1)    
    return scale * clt_helper((prime - 1,) * numVars, weak_compositions(numVars, numVars))

def compute_largest_term(numVars, prime):
    scale = (prime - 1) ** prime
    secondStepHomogDegree = numVars * (prime - 1)
    y = (prime - 1) * prime
    largestCoefficient = scale * clt_helper((y,) * numVars, weak_compositions(secondStepHomogDegree, numVars))
    return largestCoefficient

numVars = [4, 5, 6]
primes = [5, 7, 11, 13]

for n in numVars:
    for p in primes:
        t1 = time.time()
        bound1 = compute_largest_term_firststep(n, p)
        bound2 = compute_largest_term(n, p)
        t2 = time.time()
        print(f"n = {n}, p = {p}: ")
        print(f"\tbound1 = {bound1}, bound2 = {bound2}")
        print(f"\ttime taken: {(t2 - t1)} seconds")


# t1 = time.time()
# sus = compute_largest_term(4, 5)
# t2 = time.time()

# print('largest coefficient: ', sus)
# print(f'time taken: {t2 - t1} seconds')