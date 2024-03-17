import math
import cmath
import time

# Slow way to multiply polynomials
def slowMultiply(p1, p2) -> list:
    temp = [0] * (len(p1) + len(p2) - 1)
    for i in range(len(p1)):
        for j in range(len(p2)):
            temp[i + j] += p1[i] * p2[j]
    return temp

# DFT evaluates polynomial of degree n-1 at all n roots of unity. For purpose of testing,
# scaling the polynomials to degrees of power 2, because FFT wants polynomials with power 2
# to do recursion easily
def slowDFT(p1, n, inverted=1) -> list:
    # initializing y in Va=y
    temp = [0] * n

    # Calculating y
    for j in range(0, n):
        c = complex(0, 0)
        for k in range(0, n):
            c += p1[k] * cmath.rect(1, inverted * 2 * j * k * cmath.pi/n)
        temp[j] = c

    return temp

def slowIDFT(p, n) -> list:
    return [x/n for x in slowDFT(p, n, inverted=-1)]

# Fast Discrete Fourier Transform, uses recursion to do same thing as SlowDFT but faster
def fastDFT(p, inverted = 1):
    n = len(p)

    if n == 1:
        return p

    # Generating sub-polynomials, for p(x) = p1(x^2) + xp2(x^2)
    p1 = p[::2]
    p2 = p[1::2]

    y1 = fastDFT(p1, inverted)
    y2 = fastDFT(p2, inverted)

    result = [0] * n

    for j in range(n // 2):
        theta = cmath.exp(2 * j * inverted * cmath.pi * 1j / n)
        result[j] = y1[j] + theta * y2[j]
        result[j + n // 2] = y1[j] - theta * y2[j]

    return result


def fastIDFT(p):
    return [x / len(p) for x in fastDFT(p, inverted=-1)]


def fastMultiply(p1, p2) -> list:
    n = 2**math.ceil(math.log(len(p1)+len(p2)-1, 2))
    finalLength = len(p1) + len(p2) - 1
    p1 += [0] * (n - len(p1))
    p2 += [0] * (n - len(p2))

    vec1 = fastDFT(p1)
    vec2 = fastDFT(p2)

    vec3 = [vec1[i] * vec2[i] for i in range(n)]
    ans = fastIDFT(vec3)
    
    return [round(ans[i].real) for i in range(finalLength)]

arr = [i for i in range(1,10000)]
polynomial1 = arr
polynomial2 = arr

st = time.time()
print(slowMultiply(polynomial1, polynomial2)[9997])
print('Runtime for slow algorithm:',time.time() - st)
st = time.time()
print(fastMultiply(polynomial1, polynomial2)[9997])
print('Runtime for FFT:',time.time() - st)

# print(slowMultiply([1,2,3,4], [1,2,3,4]))
# print(fastMultiply([1,2,3,4], [1,2,3,4]))


# print(slowDFT([1,2,0,0], 4))
# print(fastDFT([1,2,0,0], 4))

# temp1 = slowDFT(slowDFT([1,2,3,4,0,0,0,0], 8), 8, -1)
# temp2 = fastIDFT(fastDFT([1,2,3,4,0,0,0,0]))
# temp1 = [[(round(temp1[i].real, 2), round(temp1[i].imag, 2)) for i in range(len(temp1))]]
# temp2 = [[(round(temp2[i].real, 2), round(temp2[i].imag, 2)) for i in range(len(temp2))]]

# temp1 = slowIDFT(slowDFT([1,2,0,0], 4), 4)
# temp2 = fastIDFT(slowDFT([1,2,0,0], 4), 4)
# temp1 = [[(round(temp1[i].real, 2), round(temp1[i].imag, 2)) for i in range(len(temp1))]]
# temp2 = [[(round(temp2[i].real, 2), round(temp2[i].imag, 2)) for i in range(len(temp2))]]

# print(temp1)
# print(temp2)