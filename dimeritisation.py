import numpy as np
from matplotlib import pyplot as plt
import math

from gillespie import gillespie
from gillespieSSA import gillespieSSA
from DSM import deterministic_simulation
from ODE import dimerisation_kinetics_odefun_deterministic
from CLE import CLE
from poisson_approx import poisson_approx
from utils import bin_linear, plot_result, simulate_many


def simulate_dimerisation():

    P_NAMES = ("Protein P", "Protein $P_2$")
    # simulate using Deterministic simulation (DSM)

    P_init = np.array([5*10**(-7), 0])
    T_max = 12
    step_size = 0.01

    k_guess = np.array([5e5, 0.2])
    P_dsm, T_dsm = deterministic_simulation(
        dimerisation_kinetics_odefun_deterministic, P_init, T_max, step_size, k_guess)
    plot_result(T_dsm, P_dsm, title="Deterministic dimeritisation",
                legend=P_NAMES)

    # simulate using Gillespie
    M, c, S = generate_dimerisation_instance()
    #T_g, X_g = gillespieSSA(S, M, dimeritisation_hazards, c, t_max=12)
    #plot_result(T_g, X_g, title="Gillespie dimeritisation", legend=P_NAMES)
    #
    # simulate using the Poisson approximation method
    #M, c, S = generate_dimerisation_instance()
    #Nt = 1000
    # T_p, X_p = poisson_approx(
    #    S, M, dimeritisation_hazards, c, np.linspace(1, 12, Nt))
    #plot_result(T_p, X_p, title="Poisson dimeritisation", legend=P_NAMES)
    #
    # simulate using the CLE method
    #M, c, S = generate_dimerisation_instance()
    # Nt = 400  # choosing delta_t such that propensity * delta_t >> 1.
    #
    #T_p, X_p = CLE(S, M, dimeritisation_hazards, c, np.linspace(1, 100, Nt))
    #plot_result(T_p, X_p, title="CLE dimeritisation", legend=P_NAMES)
    simulate_many(S, M, dimeritisation_hazards, c, T_max, P_NAMES,
                  "Gillespie", gillespieSSA, "Dimeritisation", bw=1)

    # Linspace size
    Nt = 4000

    # simulate using the Poisson approximation method
    simulate_many(S, M, dimeritisation_hazards, c, T_max,
                  P_NAMES, "Poisson", poisson_approx, "Dimeritisation", Nt)

    # simulate using the CLE method
    simulate_many(S, M, dimeritisation_hazards, c, T_max,
                  P_NAMES, "CLE", CLE, "Dimeritisation", Nt)


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


def generate_dimerisation_instance():
    # Initial values.
    M = np.array([301, 0])
    c = np.array([1.66*10**(-3), 0.2])

    # Initializing pre and post-matrices
    pre = np.array([2, 0, 0, 1]).reshape(2, 2)
    post = np.array([0, 1, 2, 0]).reshape(2, 2)

    # Computing Stoichiometry matrix
    S = np.transpose(post-pre)
    return M, c, S


def dimeritisation_hazards(x, c):
    """ Evaluates the hazard functions of the Lotka-Volterra system.

    :param x: Current system state. One-dimensional numpy array with length N.
    :param c: Vector of stochastic rate constants. One-dimensional numpy array with length V.
    :return: All reaction hazards as a one-dimensional numpy array of length V.
    """
    c1 = c[0]
    c2 = c[1]
    x1 = x[0]
    x2 = x[1]

    h = [
        c1 * (x1*(x1-1)) / 2,
        c2 * x2
    ]

    return h
