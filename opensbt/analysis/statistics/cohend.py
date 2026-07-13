from math import sqrt
import numpy as np
from numpy import mean
from numpy import var
from scipy.stats import wilcoxon

def cohend(d1, d2):
    """"Function to calculate Cohen's d for independent samples

    :param d1: First dataset.
    :type d1: list
    :param d2: Second dataset.
    :type d2:list
    :return: Class and value.
    :rtype: tuple[Literal['large', 'medium', 'small', 'negligible', ''], Any]
    """

    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    d = (u1 - u2) / s
    d = abs(d)

    result = ''
    if d < 0.2:
        result = 'negligible'
    if 0.2 <= d < 0.5:
        result = 'small'
    if 0.5 <= d < 0.8:
        result = 'medium'
    if d >= 0.8:
        result = 'large'

    return result, d