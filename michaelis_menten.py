import numpy as np
from matplotlib import pyplot as plt
import math

from gillespie import gillespie
from gillespieSSA import gillespieSSA
from DSM import deterministic_simulation
from ODE import michaelis_odefun
from CLE import CLE
from poisson_approx import poisson_approx
from utils import bin_linear, plot_result, simulate_many


def simulate_michaelis():
    # setup constants
    M, S_init, c, S = generate_michaelis_instance()
    S_NAMES = ("S", "E", "SE", "P")

    # simulate using deterministic simulation
    T_max = 50
    step_size = 0.01
    k_guess = np.array([1*10**6, 1*10**(-4), 0.1])
    # S_dsm, T_dsm = deterministic_simulation(
    #     michaelis_odefun, S_init, T_max, step_size, k_guess)
    # plot_result(
    #     T_dsm, S_dsm, title="Deterministic Michaelis-Menten", legend=S_NAMES)

    # simulate using Gillespie
    # T_g, X_g = gillespieSSA(S, M, michaelis_hazards, c, t_max=T_max)
    # plot_result(T_g, X_g, title="Gillespie Michaelis-Menten", legend=S_NAMES)

    # simulate using the Poisson approximation method
    # Nt = 200
    # T_p, X_p = poisson_approx(S, M, michaelis_hazards,
    #                           c, np.linspace(1, T_max, Nt))
    # plot_result(T_p, X_p, title="Poisson Michaelis-Menten", legend=S_NAMES)

    # simulate using the CLE method
    # Nt = 2000
    # T_p, X_p = CLE(S, M, michaelis_hazards, c, np.linspace(1, T_max, Nt))
    # plot_result(T_p, X_p, title="CLE Michaelis-Menten", legend=S_NAMES)

    simulate_many(S, M, michaelis_hazards, c, T_max, S_NAMES,
                  "Gillespie", gillespieSSA, "Michaelis-Menten", bw=5)

    # Linspace size
    Nt = 200

    # simulate using the Poisson approximation method
    simulate_many(S, M, michaelis_hazards, c, T_max,
                  S_NAMES, "Poisson", poisson_approx, "Michaelis-Menten", Nt)

    # simulate using the CLE method
    simulate_many(S, M, michaelis_hazards, c, T_max, S_NAMES,
                  "CLE", CLE, "Michaelis-Menten", Nt)


def generate_michaelis_instance():
    # Initial values.
    Na = 6.02*10**23
    V = 10**-15
    k1 = 1*10**6
    k2 = 1*10**(-4)
    k3 = 0.1
    M = np.array([301, 120, 0, 0])
    P_init = M / (Na*V)
    c = np.array([k1 / (Na * V), k2, k3])

    # Initializing pre and post-matrices
    pre = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]).reshape(3, 4)
    post = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1]).reshape(3, 4)

    # Computing Stoichiometry matrix
    S = np.transpose(post-pre)
    return M, P_init, c, S


def michaelis_hazards(x, c):
    """ Evaluates the hazard functions of the Michaelis-Menten system.

    :param x: Current system state. One-dimensional numpy array with length N.
    :param c: Vector of stochastic rate constants. One-dimensional numpy array with length V.
    :return: All reaction hazards as a one-dimensional numpy array of length V.
    """
    c1 = c[0]
    c2 = c[1]
    c3 = c[2]

    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    h = [
        c1*x1*x2,  # S + E -> SE
        c2*x3,    # SE -> S + E
        c3*x3     # SE -> P + E
    ]

    return h
