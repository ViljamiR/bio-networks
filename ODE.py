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

    f1 = cell * p * p * k1
    f2 = cell * p2 * k2
    return [f1, f2]


def dimerisation_kinetics_odefun_deterministic(P, k=np.array([5e5, 0.2])):
    #V = 10**(-15)
    #Na = 6.02*10**23
    #c1 = 1.66*10**(-3)
    #c2 = 0.2

    k1 = k[0]#c1 * Na * V / 2
    k2 = k[1]#c2

    p = P[0]
    p2 = P[1]

    f1 = 2*k2*p2 - 2*k1*p**2  
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

    f1 = k2 * SE - k1 * S * E
    f2 = (k2 + k3)*SE - k1 * S * E
    f3 = k1 * S * E - (k2 + k3) * SE
    f4 = k3 * SE
    return [f1, f2, f3, f4]
