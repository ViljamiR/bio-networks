
from ODE import lac_operon_odefun
import numpy as np
from DSM import deterministic_simulation
from gillespieSSA import gillespieSSA
from poisson_approx import poisson_approx
from CLE import CLE
from matplotlib import pyplot as plt
from utils import bin_linear, plot_result, simulate_many


def simulate_lac_operon():

    P_NAMES = ("i", "rI", "I", "Lac", "o", "RNAP", "A",
               "Y", "Z", "r", "ILac", "Io", "RNAPo")

    M, c, S, k = generate_lac_operon_instance()
    P_init = M
    k_guess = k

    T_max = 1000
    step_size = 4

    # simulate using Deterministic simulation (DSM)
    P_dsm, T_dsm = deterministic_simulation(
        lac_operon_odefun, P_init, T_max, step_size, k_guess)
    plot_result(T_dsm, P_dsm, title="Deterministic lac operon",
                legend=P_NAMES)

    # simulate using Gillespie
    simulate_many(S, M, lac_operon_hazards, c, T_max, P_NAMES,
                  "Gillespie", gillespieSSA, "Lac operon")

    # Linspace size
    Nt = 4000

    # simulate using the Poisson approximation method
    simulate_many(S, M, lac_operon_hazards, c, T_max,
                  P_NAMES, "Poisson", poisson_approx, "Lac operon", Nt)

    # simulate using the CLE method
    simulate_many(S, M, lac_operon_hazards, c, T_max,
                  P_NAMES, "CLE", CLE, "Lac operon", Nt)


def generate_lac_operon_instance():
    # Initial values.
    Na = 6.02e23
    V = 1e-29
    M = np.array([
        1,  # i
        0,  # rI
        50,  # I
        20,  # Lac
        1,  # o
        100,  # RNAP
        0,  # A
        0,  # Y
        0,  # Z
        0,  # r
        0,  # ILac
        0,  # Io
        0,  # RNAPo
    ])

    denominator = Na * V
    P_init = M / denominator

    c = np.array([
        0.02,           # c1
        0.1,            # c2
        0.005,          # c3
        0.1,             # c4
        1,             # c5
        0.01,             # c6
        0.1,             # c7
        0.01,             # c8
        0.003,             # c9
        0.1,             # c10
        1e-5             # c11
    ])

    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = c

    k = np.array([
        c1,             # k1
        c2,             # k2
        c3*Na*V,        # k3
        c4,             # k4
        c5*Na*V,        # k5
        c6,             # k6
        c7*Na*V,        # k7
        c8,             # k8
        c9,             # k9
        c10,            # k10
        c11*Na*V        # k11
    ])

    # Initializing pre and post-matrices

    pre = np.array([
        #i,   rI,  I,   Lac, o,   RNAP,   A,   Y,     Z,    r,   ILac, Io,  RNAPo
        [1,   0,   0,   0,   0,   0,      0,   0,
         0,    0,   0,    0,   0],          # f1
        [0,   1,   0,   0,   0,   0,      0,   0,
         0,    0,   0,    0,   0],          # f2
        [0,   0,   1,   1,   0,   0,      0,   0,
         0,    0,   0,    0,   0],          # f3
        [0,   0,   0,   0,   0,   0,      0,   0,
         0,    0,   1,    0,   0],          # f4
        [0,   0,   1,   0,   1,   0,      0,   0,
         0,    0,   0,    0,   0],          # f5
        [0,   0,   0,   0,   0,   0,      0,   0,
         0,    0,   0,    1,   0],          # f6
        [0,   0,   0,   0,   1,   1,      0,   0,
         0,    0,   0,    0,   0],          # f7
        [0,   0,   0,   0,   0,   0,      0,   0,
         0,    0,   0,    0,   1],          # f8
        [0,   0,   0,   0,   0,   0,      0,   0,
         0,    0,   0,    0,   1],          # f9
        [0,   0,   0,   0,   0,   0,      0,   0,
         0,    1,   0,    0,   0],          # f10
        [0,   0,   0,   1,   0,   0,      0,   0,
         1,    0,   0,    0,   0]           # f11
    ])

    post = np.array([
        #i,   rI,  I,   Lac, o,   RNAP,   A,   Y,     Z,    r,   ILac, Io,  RNAPo
        [1,   1,   0,   0,   0,   0,      0,   0,
         0,    0,   0,    0,   0],          # f1
        [0,   1,   1,   0,   0,   0,      0,   0,
         0,    0,   0,    0,   0],          # f2
        [0,   0,   0,   0,   0,   0,      0,   0,
         0,    0,   1,    0,   0],          # f3
        [0,   0,   1,   1,   0,   0,      0,   0,
         0,    0,   0,    0,   0],          # f4
        [0,   0,   0,   0,   0,   0,      0,   0,
         0,    0,   0,    1,   0],          # f5
        [0,   0,   1,   0,   1,   0,      0,   0,
         0,    0,   0,    0,   0],          # f6
        [0,   0,   0,   0,   0,   0,      0,   0,
         0,    0,   0,    0,   1],          # f7
        [0,   0,   0,   0,   1,   1,      0,   0,
         0,    0,   0,    0,   0],          # f8
        [0,   0,   0,   0,   1,   1,      0,   0,
         0,    1,   0,    0,   0],          # f9
        [0,   0,   0,   0,   0,   0,      1,   1,
         1,    1,   0,    0,   0],          # f10
        [0,   0,   0,   0,   0,   0,      0,   0,
         1,    0,   0,    0,   0]           # f11
    ])

    A = post - pre

    # Computing Stoichiometry matrix
    S = A.T
    return M, c, S, k


def lac_operon_hazards(x, c):
    """ Evaluates the hazard functions of the Lotka-Volterra system.

    :param x: Current system state. One-dimensional numpy array with length N.
    :param c: Vector of stochastic rate constants. One-dimensional numpy array with length V.
    :return: All reaction hazards as a one-dimensional numpy array of length V.
    """
    # Reactions: Repression, reverse repression, transcription,
    # translation, dimerisation, dissociation, mRNA degradation, protetin degradation

    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = c

    i, rI, I, Lac, o, RNAP, A, Y, Z, r, ILac, Io, RNAPo = x

    h = [
        c1*i,                    # f1
        c2*rI,                   # f2
        c3*I*Lac,                # f3
        c4*ILac,                 # f4
        c5*I*o,                  # f5
        c6*Io,                   # f6
        c7*o*RNAP,               # f7
        c8*RNAPo,                # f8
        c9*RNAPo,                # f9
        c10*r,                   # f10
        c11*Lac*Z                # f11
    ]

    return h
