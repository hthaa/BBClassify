import scipy.special
import statistics
import numpy
from scipy.stats import binom

## Livingston and Lewis' effective test length.
# mean = the mean of the observed-score distribution.
# var = the variance of the observed-score distribution.
# reliability = the reliability of the scores.
# min = the minimum possible value.
# max = the maximum possible value.
def etl(mean, var, reliability, min = 0, max = 1):
    print(((mean - min) * (max - mean) - (reliability * var)) / (var * (1 - reliability)))

## Lord's k.
# mean = the mean of the observed-score distribution.
# var = the variance of the observed-score distribution.
# reliability = the reliability of the scores.
# length = the test-length.
def k(mean, var, reliability, length):
    vare = var * (1 - reliability)
    num = length * ((length - 1) * (var - vare) - length * var + mean * (length - mean))
    den = 2 * (mean * (length - mean) - (var - vare))
    return(num / den)

## Density function for the four-parameter beta distribution.
# x = specific point along the four-parameter beta distribution.
# a = alpha shape parameter.
# b = beta shape paramter.
# l = lower-bound location parameter.
# u = upper-bound location parameter.
def dbeta4p(x, a, b, l, u):
    return((1 / scipy.special.beta(a, b)) * (((x - l)**(a - 1) * (u - x)**(b - 1)) / (u - l)**(a + b - 1)))

## Function for fitting a four-parameter beta distribution to a vector of values-
# x = vector of values.
def beta4fit(x):
    m1 = statistics.mean(x)
    s2 = statistics.variance(x)
    x3 = list(x)
    x4 = list(x)
    for i in range(len(x)):
        x3[i] = ((x3[i] - m1)**3) / (s2**0.5)**3
        x4[i] = ((x4[i] - m1)**4) / (s2**0.5)**4
    g3 = (1 / len(x3)) * sum(x3)
    g4 = (1 / len(x4)) * sum(x4)
    r = 6 * (g4 - g3**2 - 1) / (6 + 3 * g3**2 - 2 * g4)
    if g3 < 0:
        a = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        b = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
    else:
        b = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        a = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
    l = m1 - ((a * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
    u = m1 + ((b * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
    return([a, b, l, u])

## Function for fitting a two-parameter beta distribution to a vector of values.
# x = vector of values.
# l = lower-bound location parameter.
# u = upper-bound location parameter.
def beta2fit(x, l, u):
    m1 = statistics.mean(x)
    s2 = statistics.variance(x)
    a = ((l - m1) * (l * (m1 - u) - m1**2 + m1 * u - s2)) / (s2 * (l - u))
    b = ((m1 - u) * (l * (u - m1) + m1**2 - m1 * u + s2)) / (s2 * (u - l))
    return([a, b, l, u])

## Density function for Lord's two-term approximation of the compound binomial distribution.
# p = probability of success.
# N = total number of trials.
# n = specific number of successes.
# k = Lord's k.
def dcbinom(p, N, n, k):
    a = binom.pmf(n, N, p)
    b = binom.pmf(n, N - 2, p)
    c = binom.pmf(n - 1, N - 2, p)
    d = binom.pmf(n - 2, N - 2, p)
    e = k * p * (1 - p)
    return(a - e * (b - 2*c + d))

## Function for calculating the descending factorial each value of a vector.
# x = vector of values.
# r = the power x is to be raised to.
def dfac(x, r):
    x1 = list(x)
    for i in range(len(x)):
        if r <= 1:
            x1[i] = x1[i]**r
        else:
            for j in range(1, r + 1):
                if j > 1:
                    x1[i] = x1[i] * (x[i] - j + 1)
    return(x1)

## Function for calculating the first four descending factorial raw moments.
# x = vector of values.
# n = the effective or actual test length.
# k = Lord's k.
def dfm(x, n, k):
    m = [0, 0, 0, 0]
    for i in range(0, 4):
        if i == 0:
            m[i] = statistics.mean(x) / n
        else:
            M = i + 1
            a = (dfac([n], 2)[0] + k * dfac([M], 2)[0])
            b = statistics.mean(dfac(x, M)) / dfac([n - 2], M - 2)[0]
            c = k * dfac([M], 2)[0] * m[i]
            m[i] = (b / a) + c
    return(m)


#print(18**0)
#print(dfac([18], 0))
#print(dfac([10, 13, 14, 15], 2))
print(dfm([1, 2, 3, 4, 5, 6, 7, 8, 9], 20, 1))

#print(dfm([10, 13, 14, 15], 20, 0))
#print(dfac([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 2))
#print(dfac([5], 2))


# 1/(dfac(N, 2) + k * dfac(i + 1 , 2)) * 
# ((mean(dfac(x, i)) / 
# dfac(N - 2, i - 2)) + 
# k * dfac(i, 2) * m[i - 1])