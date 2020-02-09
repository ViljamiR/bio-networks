
from ODE import auto_regulatory_odefun
import numpy as np
from DSM import deterministic_simulation
from matplotlib import pyplot as plt


def simulate_auto_regulation():

    P_NAMES = ("gene g", "Protein dimer $P_2$",  "mRNA r",
               "Protein P", "Repression product  $gP_2$")
    # simulate using Deterministic simulation (DSM)

    P_init = np.array([10, 0, 0, 0, 0])
    T_max = 2
    step_size = 0.01

    k_guess = np.array([1, 10, 0.01, 10, 1, 1, 0.1, 0.01])
    P_dsm, T_dsm = deterministic_simulation(
        auto_regulatory_odefun, P_init, T_max, step_size, k_guess)
    print(len(T_dsm))
    plot_result(T_dsm, P_dsm, title="Deterministic auto-regulation",
                legend=P_NAMES)

    # simulate using Gillespie
   #M, c, S = generate_auto_reg_instance()
   #T_g, X_g = gillespieSSA(S, M, dimeritisation_hazards, c, t_max=12)
   #plot_result(T_g, X_g, title="Gillespie dimeritisation", legend=P_NAMES)

    # simulate using the Poisson approximation method
    # M, c, S = generate_auto_reg_instance()
    # Nt = 1000
    # T_p, X_p = poisson_approx(
    #     S, M, auto_regulatory_hazards, c, np.linspace(1, 12, Nt))
    # plot_result(T_p, X_p, title="Poisson auto-regulation", legend=P_NAMES)

    # # simulate using the CLE method
    # M, c, S = generate_auto_reg_instance()
    # Nt = 100  # choosing delta_t such that propensity * delta_t >> 1.

    # T_p, X_p = CLE(S, M, auto_regulatory_hazards, c, np.linspace(1, 12, Nt))
    # plot_result(T_p, X_p, title="CLE auto-regulation", legend=P_NAMES)


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


def generate_auto_reg_instance():
    # Initial values.
    M = np.array([10, 0, 0, 0, 0])
    c = np.array([1.66*10**(-3), 0.2])

    # Initializing pre and post-matrices
    pre = np.array([2, 0, 0, 1]).reshape(2, 2)
    post = np.array([0, 1, 2, 0]).reshape(2, 2)

    print(pre, '\n', post)
    # Computing Stoichiometry matrix
    S = np.transpose(post-pre)
    return M, c, S


def auto_regulatory_hazards(x, c):
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
