import itertools as it

from bisect import bisect_left
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as ss
import logging as log
from pandas import Categorical


def VD_A(treatment: List[float], control: List[float]):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    :param treatment: a numeric list
    :param control: another numeric list
    :returns the value estimate and the magnitude
    """
    m = len(treatment)
    n = len(control)

    if m != n:
        raise ValueError("Data d and f must have the same length")

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude


def VD_A_DF(data, val_col: str = None, group_col: str = None, sort=True):
    """
    :param data: pandas DataFrame object
        An array, any object exposing the array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.
    :param val_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains values.
    :param group_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains group names.
    :param sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not. Recommended
        unless you sort your data manually.
    :return: stats : pandas DataFrame of effect sizes
    Stats summary ::
    'A' : Name of first measurement
    'B' : Name of second measurement
    'estimate' : effect sizes
    'magnitude' : magnitude
    """

    x = data.copy()
    if sort:
        x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)

    groups = x[group_col].unique()

    # Pairwise combinations
    g1, g2 = np.array(list(it.combinations(np.arange(groups.size), 2))).T

    # Compute effect size for each combination
    ef = np.array([VD_A(list(x[val_col][x[group_col] == groups[i]].values),
                        list(x[val_col][x[group_col] == groups[j]].values)) for i, j in zip(g1, g2)])

    return pd.DataFrame({
        'A': np.unique(data[group_col])[g1],
        'B': np.unique(data[group_col])[g2],
        'estimate': ef[:, 0],
        'magnitude': ef[:, 1]
    })


if __name__ == '__main__':
    # Examples

    # # negligible
    # F = [0.8236111111111111, 0.7966666666666666, 0.923611111111111, 0.8197222222222222, 0.7108333333333333]
    # G = [0.8052777777777779, 0.8172222222222221, 0.8322222222222223, 0.783611111111111, 0.8141666666666666]
    # log.info(VD_A(G, F))

    # # small
    # A = [0.478515625, 0.4638671875, 0.4638671875, 0.4697265625, 0.4638671875, 0.474609375, 0.4814453125, 0.4814453125,
    #      0.4697265625, 0.4814453125, 0.474609375, 0.4833984375, 0.484375, 0.44921875, 0.474609375, 0.484375,
    #      0.4814453125, 0.4638671875, 0.484375, 0.478515625, 0.478515625, 0.45703125, 0.484375, 0.419921875,
    #      0.4833984375, 0.478515625, 0.4697265625, 0.484375, 0.478515625, 0.4638671875]
    # B = [0.4814453125, 0.478515625, 0.44921875, 0.4814453125, 0.4638671875, 0.478515625, 0.474609375, 0.4638671875,
    #      0.474609375, 0.44921875, 0.474609375, 0.478515625, 0.478515625, 0.474609375, 0.4697265625, 0.474609375,
    #      0.45703125, 0.4697265625, 0.478515625, 0.4697265625, 0.4697265625, 0.484375, 0.45703125, 0.474609375,
    #      0.474609375, 0.4638671875, 0.45703125, 0.474609375, 0.4638671875, 0.4306640625]

    # log.info(VD_A(A, B))

    # # medium
    # C = [0.9108333333333334, 0.8755555555555556, 0.900277777777778, 0.9274999999999999, 0.8777777777777779]
    # E = [0.8663888888888888, 0.8802777777777777, 0.7816666666666667, 0.8377777777777776, 0.9305555555555556]
    # log.info(VD_A(C, E))

    # # Large
    # D = [0.7202777777777778, 0.77, 0.8544444444444445, 0.7947222222222222, 0.7577777777777778]
    # log.info(VD_A(C, D))

    # Metrics Paper (10 runs)
    NSGAII = [0.18184417518118298, 0.2245256586915804, 0.17647426490463292, 0.16230448654974916, 0.18738405941240882, 0.3101233019449206, 0.3054707753488759, 0.27873281142439615, 0.25083069533060276, 0.1612208601808664]
    RS = [0.0753816102665673, 0.06437859231871611, 0.0666155505693268, 0.07059016975382326, 0.07065245326751085, 0.06454400375044805, 0.07402294266750688, 0.07003813819778802, 0.0687958141591721, 0.07122835191659796]
    log.info(VD_A(RS, NSGAII))