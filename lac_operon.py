
from ODE import lac_operon_odefun
import numpy as np
from DSM import deterministic_simulation
from gillespieSSA import gillespieSSA
from poisson_approx import poisson_approx
from CLE import CLE
from matplotlib import pyplot as plt


def simulate_lac_operon():

    P_NAMES = ( "i", "rI", "I", "Lac", "o", "RNAP", "A", "Y", "Z", "r", "ILac", "Io", "RNAPo")
    P_NAMES_STOCH = ("Repression product $gP_2$", "gene g",
                     "mRNA r", "Protein P", "Protein dimer $P_2$")
    # simulate using Deterministic simulation (DSM)
    M, c, S, k = generate_lac_operon_instance()
    P_init = np.array([
      1,  # i
      0,  # rI
      50, # I
      20, # Lac
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
    k_guess = c
    #print(0.1*0 - k3*I*Lac + k4 * ILac - k5 * I * o + k6 * Io)
    #print(lac_operon_odefun(P_init, k_guess))
    T_max = 800
    step_size = 0.01

    # Values in order:
    # r,P_2, r, P, gP_2
    P_dsm, T_dsm = deterministic_simulation(
        lac_operon_odefun, P_init, T_max, step_size, k_guess)
    plot_result(T_dsm, P_dsm, title="Deterministic auto-regulation",
                 legend=P_NAMES)

    # For stochastic values are in order:
    # gP_2, g, r, P, P_2
    # simulate using Gillespie
    #T_g, X_g = gillespieSSA(S, M, lac_operon_hazards, c, t_max=T_max)
    #plot_result(T_g, X_g, title="Gillespie dimeritisation",
    #            legend=P_NAMES_STOCH)
    #plot_result(T_g, X_g[:, 2], title="Gillespie dimeritisation",
    #            legend=("RNA"))

    # simulate using the Poisson approximation method
    ##Nt = 400
    #T_p, X_p = poisson_approx(
    #    S, M, lac_operon_hazards, c, np.linspace(1, T_max, Nt))
    #plot_result(T_p, X_p, title="Poisson auto-regulation",
    ##            legend=P_NAMES_STOCH)
    #plot_result(T_p, X_p[:, 4], title="Poisson auto-regulation",
    #            legend=("P_2"))

    # # simulate using the CLE method
    #M, c, S = generate_auto_reg_instance()
    #print("M",M)
    #Nt = 4000  # choosing delta_t such that propensity * delta_t >> 1.

    #T_p, X_p = CLE(S, M, lac_operon_hazards, c, np.linspace(1, T_max, Nt))
    #plot_result(T_p, X_p, title="CLE auto-regulation", legend=P_NAMES_STOCH)


"""
Copied from Exercises to visualize data.
"""


def plot_result(T, X, title="", legend=("A (prey)", "B (Predator)")):
    """Visualize a Lotka-Volterra simulation result. 

    :param T: Time step vector
    :param X: State vector
    :return: Nothing.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(T, X)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.legend(legend, loc='upper right')
    plt.show()


def generate_lac_operon_instance():
    # Initial values.
    Na = 6.02*10**23
    V = 1e-15
    M = np.array([10, 10, 10, 10, 10])
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
    pre = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]).reshape(5, 8)
    post = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                     0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]).reshape(8, 5)
    A = np.array([1, -1, 0, 0, -1, -1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                  0, 0, -2, 1, 0, 0, 0, 2, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0]).reshape(8, 5)
    print(pre, '\n', post)
    # Computing Stoichiometry matrix
    print("This A", A)
    S = A.T
    return M, c, S, k


def lac_operon_hazards(x, c):
    """ Evaluates the hazard functions of the Lotka-Volterra system.

    :param x: Current system state. One-dimensional numpy array with length N.
    :param c: Vector of stochastic rate constants. One-dimensional numpy array with length V.
    :return: All reaction hazards as a one-dimensional numpy array of length V.
    """
    # Repression, reverse repression, transcription,
    # translation, dimerisation, dissociation, mRNA degradation, protetin degradation
    print("x in hazards",x)
    print("c in hazards",c)
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = c

    i, rI, I, Lac, o, RNAP, A, Y, Z, r, ILac, Io, RNAPo = x

    #h = [
    #    0,
    #    0,
    #    c1 * g*P_2,             # repression
    #    c2 * gP_2,              # reverse repression
    #    c3 * g,                 # transcription
    #    c4 * r,                 # translation
    #    c5 * (P*(P-1))/2,       # dimerisation
    #    c6 * P_2,               # dissociation
    #    c7 * r,                 # mRNA degradation
    #    c8 * P,                 # protein degradation
    #]

    return []
