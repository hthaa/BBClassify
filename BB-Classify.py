import scipy.special
from scipy.integrate import quad
from scipy.stats import binom
import statistics as stats
import math
import pandas as pd
import numpy as np
import csv

## Livingston and Lewis' effective test length.
# mean = the mean of the observed-score distribution.
# var = the variance of the observed-score distribution.
# reliability = the reliability of the scores.
# min = the minimum possible value.
# max = the maximum possible value.
def etl(mean, var, reliability, min = 0, max = 1):
    return ((mean - min) * (max - mean) - (reliability * var)) / (var * (1 - reliability))

## Lord's k.
# mean = the mean of the observed-score distribution.
# var = the variance of the observed-score distribution.
# reliability = the reliability of the scores.
# length = the test-length.
def k(mean, var, reliability, length):
    vare = var * (1 - reliability)
    num = length * ((length - 1) * (var - vare) - length * var + mean * (length - mean))
    den = 2 * (mean * (length - mean) - (var - vare))
    return num / den

## Density function for the four-parameter beta distribution.
# x = specific point along the four-parameter beta distribution.
# a = alpha shape parameter.
# b = beta shape paramter.
# l = lower-bound location parameter.
# u = upper-bound location parameter.
def dbeta4p(x, a, b, l, u):
    if x < l or x > u:
        return 0
    else:
        return (1 / scipy.special.beta(a, b)) * (((x - l)**(a - 1) * (u - x)**(b - 1)) / (u - l)**(a + b - 1))

## Function for fitting a four-parameter beta distribution to a vector of values-
# x = vector of values.
# moments = an optional list of the first four raw moments
def beta4fit(x, moments = []):
    if len(moments) != 4:
        m1 = stats.mean(x)
        s2 = stats.variance(x)
        x3 = list(x)
        x4 = list(x)
        for i in range(len(x)):
            x3[i] = ((x3[i] - m1)**3) / (s2**0.5)**3
            x4[i] = ((x4[i] - m1)**4) / (s2**0.5)**4
        g3 = (1 / len(x3)) * sum(x3)
        g4 = (1 / len(x4)) * sum(x4)
    else:
        m1 = moments[0]
        s2 = moments[1] - moments[0]**2
        g3 = (moments[2] - 3 * moments[0] * moments[1] + 2 * moments[0]**3) / ((s2**0.5)**3)
        g4 = (moments[3] - 4 * moments[0] * moments[2] + 6 * moments[0]**2 * moments[0] - 3 * moments[0]**3) / ((s2**0.5)**4)
    r = 6 * (g4 - g3**2 - 1) / (6 + 3 * g3**2 - 2 * g4)
    if g3 < 0:
        a = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        b = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
    else:
        b = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        a = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
    l = m1 - ((a * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
    u = m1 + ((b * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
    return [a, b, l, u]

## Function for fitting a two-parameter beta distribution to a vector of values.
# x = vector of values.
# l = lower-bound location parameter.
# u = upper-bound location parameter.
def beta2fit(x, l, u):
    m1 = stats.mean(x)
    s2 = stats.variance(x)
    a = ((l - m1) * (l * (m1 - u) - m1**2 + m1 * u - s2)) / (s2 * (l - u))
    b = ((m1 - u) * (l * (u - m1) + m1**2 - m1 * u + s2)) / (s2 * (u - l))
    return [a, b, l, u]

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
    return a - e * (b - 2*c + d)

## Integrate across univariate BB distribution.
# a = alpha shape parameter.
# b = beta shape parameter.
# l = lower-bound location parameter.
# u = upper-bound location parameter.
# N = upper-bound of binomial distribution.
# n = specific binomial outcome.
# lower = lower-bound of integration.
# upper = upper bound of intergration.
# method = specify Livingston and Lewis ("LL") or Hanson and Brennan approach.
def bbintegrate1(a, b, l, u, N, n, k, lower, upper, method = "ll"):
    if method != "ll":
        def f(x, a, b, l, u, N, n, k):
            return dbeta4p(x, a, b, l, u) * dcbinom(x, N, n, k)
        return quad(f, lower, upper, args = (a, b, l, u, N, n, k))
    else:
        def f(x, a, b, l, u, N, n):
            return dbeta4p(x, a, b, l, u) * binom.pmf(n, N, x)
        return quad(f, lower, upper, args = (a, b, l, u, N, n))

## Integrate across bivariate BB distribution.
# a = alpha shape parameter.
# b = beta shape parameter.
# l = lower-bound location parameter.
# u = upper-bound location parameter.
# N = upper-bound of binomial distribution.
# n1 = specific binomial outcome on first binomial trial.
# n2 = specific binomial outcome on second binomial trial.
# lower = lower-bound of integration.
# upper = upper bound of intergration.
# method = specify Livingston and Lewis ("LL") or Hanson and Brennan approach.
def bbintegrate2(a, b, l, u, N, n1, n2, k, lower, upper, method = "ll"):
    if method != "ll":
        def f(x, a, b, l, u, N, n1, n2, k):
            return dbeta4p(x, a, b, l, u) * dcbinom(x, N, n1, k) * dcbinom(x, N, n2, k)
        return quad(f, lower, upper, args = (a, b, l, u, N, n1, n2, k))
    else:
        def f(x, a, b, l, u, N, n1, n2):
            return dbeta4p(x, a, b, l, u) * binom.pmf(n1, N, x) * binom.pmf(n2, N, x)
        return quad(f, lower, upper, args = (a, b, l, u, N, n1, n2))

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
    return x1

## Function for calculating the first four raw moments of the true-score distribution.
# x = vector of values.
# n = the effective or actual test length.
# k = Lord's k.
def tsm(x, n, k):
    m = [0, 0, 0, 0]
    for i in range(0, 4):
        if i == 0:
            m[i] = stats.mean(x) / n
        else:
            M = i + 1
            a = (dfac([n], 2)[0] + k * dfac([M], 2)[0])
            b = stats.mean(dfac(x, M)) / dfac([n - 2], M - 2)[0]
            c = k * dfac([M], 2)[0] * m[i]
            m[i] = (b / a) + c
    return m

## Estimate the true-score 2 or 4 parameter beta distribution parameters.
# x = vector of values
# n = actual or effective test length.
# k = Lord's k.
# model = whether 2 or 4 parameters are to be fit.
# l = if model = 2, specified lower-bound of 2-parameter distribution.
# u = if model = 2, specified upper-bound of 2-parameter distribution.
def betaparameters(x, n, k, model = 4, l = 0, u = 1):
    m = tsm(x, n, k)
    s2 = m[1] - m[0]**2
    g3 = (m[2] - 3 * m[0] * m[1] + 2 * m[0]**3) / (math.sqrt(s2)**3)
    g4 = (m[3] - 4 * m[0] * m[2] + 6 * m[0]**2 * m[1] - 3 * m[0]**4) / (math.sqrt(s2)**4)
    if model == 4:
        r = 6 * (g4 - g3**2 - 1) / (6 + 3 * g3**2 - 2 * g4)
        if g3 < 0:
            a = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
            b = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        else:
            b = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
            a = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        l = m[0] - ((a * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
        u = m[0] + ((b * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
    if model == 2:
        a = ((l - m[0]) * (l * (m[0] - u) - m[0]**2 + m[0] * u - s2)) / (s2 * (l - u))
        b = ((m[0] - u) * (l * (u - m[0]) + m[0]**2 - m[0] * u + s2)) / (s2 * (u - l))
    return {"alpha":  a, "beta": b, "l": l, "u": u}

## Function for estimating accuracy and consistency from beta-binomial models.
# x = vector of values representing test-scores, or a list of model parameters.
# reliability = the reliability coefficient of the test-scores.
# min = the minimum possible score to attain on the test (only necessary for 
#       the Livingston and Lewis approach).
# max = for the Livingston and Lewis approach, the maximum possible score to 
#       attain on the test. For the Hanson and Brennan approach, the actual
#       test length (number of items).
# model = how many parameters of the true-score distribution that is to be
#       estimated (4 or 2). Default is 4.
# l = the lower-bound location parameter for the two-parameter distribution.
# u = the lower-bound location parameter for the two-parameter distribution.
# failsafe = whether the function should automatically revert to a two-
#       parameter solution if the four-parameter fitting procedure produces
#       impermissible location-parameter estimates.
# method = whether the Livingston and Lewis or the Hanson and Brennan approach
#       is to be employed. Default is "ll" (Livingston and Lewis). Any other
#       value passed means the Hanson and Brennan approach.
def cac(x, reliability, min, max, cut, model = 4, l = 0, u = 1, failsafe = False, method = "ll"):
    output = {}
    cut = [min] + cut + [max]
    tcut = list(cut)
    for i in range(len(cut)):
        tcut[i] = (tcut[i] - min) / (max - min)
    if isinstance(x, dict):
        pars = x
        if method == "ll":
            N = pars["etl"]
        else:
            N = pars["atl"]
    else:
        if method == "ll":
            Nnotrounded = etl(stats.mean(x), stats.variance(x), reliability, min, max)
            N = round(Nnotrounded)
            pars = betaparameters(x, N, 0, model)
            if (failsafe == True and model == 4) and (l < 0 or u > 1):
                pars = betaparameters(x, N, 0, 2, l, u)
            pars["etl"] = Nnotrounded
            pars["etl rounded"] = N
            pars["lords_k"] = 0
            for i in range(len(cut)):
                cut[i] = tcut[i] * N
        else:
            N = max 
            K = k(stats.mean(x), stats.variance(x), reliability, N)
            pars = betaparameters(x, N, K, model, l, u)
            if (failsafe == True and model == 4) and (l < 0 or u > 1):
                pars = betaparameters(x, max, N, 2, l, u)
            pars["lords_k"] = K
    confmat = np.zeros((N + 1, len(cut) - 1))
    for i in range(len(cut) - 1):
        for j in range(N + 1):
            confmat[j, i] = bbintegrate1(pars["alpha"], pars["beta"], pars["l"], pars["u"], N, j, pars["lords_k"], tcut[i], tcut[i + 1], method)[0]
    confusionmatrix = np.zeros((len(cut) - 1, len(cut) - 1))
    for i in range(len(cut) - 1):
        for j in range(len(cut) - 1):
            if i != len(cut) - 2:
                confusionmatrix[i, j] = sum(confmat[cut[i]:cut[i + 1], j])
            else:
                confusionmatrix[i, j] = sum(confmat[cut[i]:, j])
    consmat = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            consmat[i, j] = bbintegrate2(pars["alpha"], pars["beta"], pars["l"], pars["u"], N, i, j, pars["lords_k"], 0, 1, method)[0]
    output["confusionMatrix"] = confusionmatrix
    consistencymatrix = np.zeros((len(cut) - 1, len(cut) - 1))
    for i in range(len(cut) - 1):
        for j in range(len(cut) - 1):
            if i == 0 and j == 0:
                consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], 0:cut[j + 1]]))
            if i == 0 and (j != 0 and j != len(cut) - 2):
                consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], cut[j]:cut[j + 1]]))
            if i == 0  and j == len(cut) - 2:
                consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], cut[j]:cut[j + 1] + 1]))
            if (i != 0 and i != len(cut) - 2) and j == 0:
                consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], 0:cut[j + 1]]))
            if (i != 0 and i != len(cut) - 2) and (j != 0 and j != len(cut) - 2):
                consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], cut[j]:cut[j + 1]]))
            if (i != 0 and i != len(cut) - 2) and j == len(cut) - 2:
                consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], cut[j]:cut[j + 1] + 1]))
            if i == len(cut) - 2 and j == 0:
                consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, 0:cut[j + 1]]))
            if i == len(cut) - 2 and (j != 0 and j != len(cut) - 2):
                consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, cut[j]:cut[j + 1]]))
            if i == len(cut) - 2 and j == len(cut) - 2:
                consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, cut[j]:cut[j + 1] + 1]))
    output["consistencyMatrix"] = consistencymatrix
    return output


#rawdata = [11, 6, 7, 14, 11, 13, 13, 13, 13, 18, 9, 10, 11, 13, 10, 11, 6, 11, 12, 14, 10, 11, 15, 10, 11, 10, 8, 12, 15, 11, 15, 7, 14, 5, 8, 13, 11, 15, 12, 10, 12, 8, 12, 11, 12, 12, 16, 13, 14, 7, 11, 12, 14, 10, 13, 13, 11, 8, 7, 15, 10, 16, 11, 9, 14, 8, 9, 11, 10, 10, 9, 13, 10, 10, 7, 10, 11, 6, 17, 10, 12, 8, 10, 14, 9, 15, 12, 5, 10, 9, 12, 12, 7, 7, 12, 8, 17, 13, 7, 7, 11, 10, 16, 14, 9, 4, 12, 5, 16, 12, 17, 10, 10, 9, 17, 14, 13, 8, 15, 12, 10, 13, 7, 11, 13, 11, 7, 16, 9, 17, 9, 9, 11, 4, 11, 10, 14, 6, 9, 15, 10, 4, 13, 16, 9, 13, 10, 14, 10, 11, 10, 13, 13, 16, 14, 9, 14, 13, 11, 11, 10, 11, 14, 7, 13, 10, 9, 14, 14, 11, 7, 10, 10, 15, 15, 10, 10, 10, 11, 9, 12, 11, 14, 11, 11, 8, 12, 7, 3, 14, 12, 14, 6, 17, 6, 12, 11, 7, 13, 13, 15, 10, 8, 12, 12, 12, 10, 12, 12, 12, 10, 9, 9, 14, 9, 15, 5, 12, 9, 10, 16, 4, 19, 10, 11, 13, 15, 12, 9, 13, 14, 15, 9, 7, 7, 12, 12, 10, 14, 12, 10, 8, 7, 16, 14, 14, 11, 12, 12, 14, 7, 13, 10, 11, 8, 13, 10, 13, 12, 16, 13, 13, 14, 11, 13, 11, 11, 9, 12, 10, 14, 8, 9, 12, 5, 11, 10, 13, 15, 11, 9, 11, 10, 13, 12, 17, 12, 11, 15, 13, 11, 7, 10, 19, 10, 13, 8, 10, 9, 10, 4, 7, 11, 12, 11, 14, 9, 11, 7, 10, 15, 11, 13, 15, 13, 10, 12, 9, 11, 15, 14, 14, 12, 15, 7, 12, 15, 12, 11, 8, 9, 12, 13, 19, 8, 12, 16, 11, 14, 11, 14, 12, 9, 13, 11, 12, 11, 12, 10, 11, 8, 16, 16, 15, 7, 13, 10, 10, 15, 11, 11, 12, 13, 15, 9, 9, 10, 5, 7, 12, 14, 7, 13, 9, 13, 13, 12, 17, 10, 13, 10, 12, 15, 12, 13, 10, 9, 7, 12, 14, 7, 13, 11, 7, 15, 13, 16, 15, 8, 9, 12, 8, 15, 7, 14, 9, 8, 12, 12, 11, 9, 12, 8, 15, 11, 13, 11, 15, 10, 10, 5, 8, 11, 8, 12, 7, 10, 9, 7, 11, 14, 12, 17, 10, 6, 9, 12, 10, 13, 8, 12, 11, 14, 12, 11, 9, 9, 14, 12, 12, 11, 12, 12, 10, 11, 9, 11, 10, 9, 12, 6, 16, 17, 12, 12, 12, 10, 12, 11, 7, 8, 10, 10, 13, 11, 13, 9, 9, 8, 12, 15, 14, 14, 12, 13, 13, 10, 13, 16, 12, 5, 15, 12, 16, 6, 8, 15, 5, 13, 8, 14, 13, 13, 10, 12, 12, 10, 10, 13, 11, 11, 11, 8, 15, 10, 10, 6, 9, 15, 12, 11, 15, 13, 9, 16, 7, 17, 12, 10, 4, 16, 12, 14, 10, 15, 12, 13, 11, 8, 10, 7, 16, 12, 9, 11, 14, 15, 5, 9, 14, 5, 15, 13, 15, 5, 9, 13, 13, 7, 15, 13, 12, 12, 12, 7, 7, 11, 15, 6, 10, 8, 12, 12, 15, 18, 10, 7, 14, 7, 8, 15, 14, 13, 13, 13, 16, 14, 7, 13, 10, 9, 9, 6, 15, 12, 13, 7, 11, 10, 12, 10, 10, 14, 12, 13, 9, 14, 6, 8, 8, 6, 10, 7, 6, 15, 5, 9, 5, 12, 7, 8, 10, 15, 11, 11, 7, 10, 8, 8, 12, 16, 11, 14, 13, 13, 13, 11, 17, 17, 8, 9, 14, 13, 9, 11, 7, 14, 14, 17, 10, 8, 13, 17, 14, 15, 14, 10, 15, 13, 5, 15, 14, 9, 8, 7, 10, 7, 12, 13, 10, 10, 7, 12, 10, 9, 11, 11, 13, 8, 12, 10, 6, 10, 11, 11, 10, 11, 15, 15, 12, 15, 12, 10, 12, 11, 6, 10, 9, 10, 10, 10, 9, 11, 13, 12, 12, 15, 7, 13, 13, 13, 17, 6, 7, 16, 13, 12, 12, 13, 17, 14, 9, 13, 12, 11, 11, 14, 17, 11, 13, 12, 13, 7, 6, 14, 13, 7, 13, 11, 10, 15, 16, 12, 9, 16, 8, 14, 7, 11, 7, 14, 13, 12, 14, 16, 11, 14, 16, 10, 14, 9, 17, 10, 15, 18, 6, 15, 9, 12, 11, 11, 8, 11, 10, 10, 13, 15, 9, 15, 16, 10, 13, 7, 13, 12, 11, 9, 11, 9, 11, 9, 14, 12, 11, 13, 14, 8, 11, 13, 3, 11, 11, 11, 11, 4, 13, 13, 16, 11, 10, 13, 16, 10, 5, 9, 11, 10, 9, 13, 10, 7, 12, 17, 9, 6, 7, 8, 10, 11, 13, 14, 10, 12, 12, 11, 13, 15, 16, 10, 10, 7, 11, 14, 15, 18, 9, 14, 11, 12, 11, 11, 17, 10, 12, 14, 11, 15, 9, 13, 14, 14, 6, 9, 7, 14, 11, 6, 12, 13, 10, 5, 9, 11, 10, 14, 12, 8, 12, 9, 12, 6, 10, 11, 14, 9, 15, 7, 18, 14, 13, 10, 10, 12, 12, 16, 9, 14, 12, 10, 10, 12, 14, 8, 11, 5, 8, 12, 9, 13, 10, 12, 8, 9, 13, 10, 7, 15, 10, 13, 12, 11, 12, 8, 13, 15, 6, 9, 8, 8, 15, 16, 10, 4, 7, 8, 9, 5, 11, 8, 11, 10, 11, 15, 12, 15, 13, 12, 9, 15, 10, 14, 11, 10, 6, 14, 8, 10, 6, 16, 7, 15, 15, 12, 12, 11, 11, 15, 14, 11, 15, 8, 10, 11, 9, 15, 8, 9, 12, 14, 16, 13, 11, 15, 17, 12, 8, 6, 12, 6, 16, 13, 6, 4, 15, 13, 5, 13, 14, 10, 8]

#pars = {"alpha": 5, "beta": 3, "l": 0, "u": 1, "etl": 20, "lords_k": 0}
pars = {"alpha": 5, "beta": 3, "l": 0, "u": 1, "atl": 20, "lords_k": 0}

pars = cac(pars, 0.4571138, cut = [5, 10, 15], min = 0, max = 20, model = 4, l = 0, u = 1, failsafe = False, method = "hb")
print(pars)