from __future__ import division 
from math import pow, sqrt

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(P,Q):
    L1 = line(P[0], P[1])
    L2 = line(Q[0], Q[1])

    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return [x,y]
    else:
        return []

def dist(p1,p2):
    return sqrt(pow(p1[0]- p2[0],2) +  pow(p1[1]- p2[1],2))

def distPair(a,b):
    res = []
    for i in range(len(a)):
        d = dist(a[i],b[i])
        res.append(d)
    return res