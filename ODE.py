import numpy as np
"""
Copied from Exercise 4.
Used for testing deterministic simulation during development.
"""


def odefun(x, k):
    """Differential function f. 

    :param x: System state. An array with length 2.
    :param k: Rate parameters. An array with length 2.
    :returns: dx/dt an array with length 2
    """
    f1 = x[0] - k[0]*x[0]*x[1]
    f2 = k[0]*x[0]*x[1]-k[1]*x[1]
    return [f1, f2]


def dimerisation_kinetics_odefun(P, k=np.array([5e5, 0.2])):
    cell = 10**(-15)
    p = P[0]
    p2 = P[1]
    k1 = k[0]
    k2 = k[1]

    # d[P_2]/dt
    f1 = cell * p * p * k1
    # d[P_2]/dt
    f2 = cell * p2 * k2
    return [f1, f2]


def dimerisation_kinetics_odefun_deterministic(P, k=np.array([5e5, 0.2])):
    k1 = k[0]  # c1 * Na * V / 2
    k2 = k[1]  # c2

    p = P[0]
    p2 = P[1]

    # d[P]/dt
    f1 = 2*k2*p2 - 2*k1*p**2
    # d[P_2]/dt
    f2 = k1*p**2 - k2*p2
    return [f1, f2]


def dimerisation_kinetics_odefun_stochastic(P, k=np.array([5e5, 0.2])):
    c1 = 1.66*10**(-3)
    c2 = k[1]
    p = P[0]
    p2 = P[1]

    f1 = c1 * p * (p-1) / 2
    f2 = c2 * p2
    return [f1, f2]


def michaelis_odefun(X, k):
    k1 = k[0]
    k2 = k[1]
    k3 = k[2]

    S = X[0]
    E = X[1]
    SE = X[2]

    # d[S]/dt
    f1 = k2 * SE - k1 * S * E
    # d[E]/dt
    f2 = (k2 + k3)*SE - k1 * S * E
    # d[SE]/dt
    f3 = k1 * S * E - (k2 + k3) * SE
    # d[P]/dt
    f4 = k3 * SE
    return [f1, f2, f3, f4]


def auto_regulatory_odefun(X, k):
    g, P_2, r, P, gP_2 = X
    k1, k2, k3, k4, k5, k6, k7, k8 = k

    # d[g]/dt
    f1 = k2*gP_2 - k1 * g * P_2
    # d[P_2]/dt
    f2 = k5*P*P - k6 * P_2 - k1*g*P_2 + k2*gP_2
    # d[r]/dt
    f3 = k3*g - k7*r
    # d[P]/dt
    f4 = 2*k6*P_2 - 2*k5*P*P + k4*r-k8*P
    # d[gP_2]/dt
    f5 = k1*g*P_2 - k2*gP_2

    return [f1, f2, f3, f4, f5]


def lac_operon_odefun(X, k):
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11 = k
    i, rI, I, Lac, o, RNAP, A, Y, Z, r, ILac, Io, RNAPo = X

    # d[i]/dt
    f1 = 0
    # d[rI]/dt
    f2 = k1 * i
    # d[I]/dt
    f3 = k2*rI - k3*I*Lac + k4 * ILac - k5 * I * o + k6 * Io
    # d[Lac]/dt
    f4 = -k3 * i * Lac + k4 * ILac - k11 * Lac * Z
    # d[o]/dt
    f5 = -k5 * I * o + k6 * Io - k7 * o * RNAP + k8*RNAPo + k9 * RNAPo
    # d[RNAP]/dt
    f6 = -k7 * o * RNAP + k8 * RNAPo + k9 * RNAP
    # d[A]/dt
    f7 = k10*r
    # d[Y]/dt
    f8 = k10*r
    # d[Z]/dt
    f9 = k10*r
    # d[r]/dt
    f10 = k9*RNAPo
    # d[ILac]/dt
    f11 = k3 * I * Lac - k4 * ILac
    # d[Io]/dt
    f12 = k5 * I * o - k6*Io
    # d[RNAPo]/dt
    f13 = k7 * o * RNAP - k8*RNAPo - k9 * RNAPo

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]
