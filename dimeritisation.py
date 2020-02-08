import numpy as np
from matplotlib import pyplot as plt
import math

from gillespie import gillespie
from gillespieSSA import gillespieSSA
from DSM import deterministic_simulation
from ODE import dimerisation_kinetics_odefun_deterministic
from CLE import CLE
from poisson_approx import poisson_approx


def main():
    M, c, S = generate_LV_instance()
    #T1, X1 = gillespie(S,M,LV_hazards, c,t_max=30, max_reactions=int(1e5))
    #plot_result(T1,X1)

    #Nt = 500
    #T, X1 = CLE(S, M, LV_hazards, c, np.linspace(0, 30, Nt))
    #plot_result(T, X1)
    # Nt = 500
    # T, X1 = CLE(S, M, LV_hazards, c, np.linspace(0, 30, Nt))
    # plot_result(T, X1)

    Nt = 1000
    T, X1 = poisson_approx(S, M, LV_hazards, c, np.linspace(0, 30, Nt))
    plot_result(T, X1)

    #X_init = np.array([1, 1.5])
    #T_max = 12
    #step_size = 0.01
    #k_guess = np.array([1.5, 3.0])
    #X_dsm, T_dsm = deterministic_simulation(odefun, X_init, T_max, step_size, k_guess)
    #plot_result(T_dsm, X_dsm)

    simulate_dimerisation()

def simulate_dimerisation():

    P_NAMES = ("Protein P", "Protein $P_2$")
    # simulate using Deterministic simulation (DSM)
    
    P_init = np.array([5*10**(-7), 0])
    T_max = 12
    step_size = 0.01

    k_guess = np.array([5e5, 0.2])
    P_dsm, T_dsm = deterministic_simulation(dimerisation_kinetics_odefun_deterministic, P_init, T_max, step_size, k_guess)
    plot_result(T_dsm, P_dsm, title="Deterministic dimeritisation", legend=P_NAMES)
    

    # simulate using Gillespie
    M, c, S = generate_dimerisation_instance()
    T_g, X_g = gillespieSSA(S,M,dimeritisation_hazards, c,t_max=12)
    plot_result(T_g, X_g,title="Gillespie dimeritisation", legend=P_NAMES)

    # simulate using the Poisson approximation method
    M, c, S = generate_dimerisation_instance()
    Nt = 1000
    T_p, X_p = poisson_approx(S,M,dimeritisation_hazards, c,np.linspace(1,12,Nt))
    plot_result(T_p, X_p, title="Poisson dimeritisation", legend=P_NAMES)

     # simulate using the Poisson approximation method
    M, c, S = generate_dimerisation_instance()
    Nt = 100 # choosing delta_t such that propensity * delta_t >> 1.

    T_p, X_p = CLE(S,M,dimeritisation_hazards, c,np.linspace(1,12,Nt))
    plot_result(T_p, X_p, title="CLE dimeritisation", legend=P_NAMES)


"""
Copied from Exercises to visualize data.
"""
def plot_result(T, X, title="", legend=("A (prey)","B (Predator)")):
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

    print(pre,'\n', post)
    # Computing Stoichiometry matrix
    S = np.transpose(post-pre)
    return M, c, S

def generate_LV_instance():
    # Initial values.
    M = np.array([100, 150])
    c = np.array([1.0, 0.01, 0.6])
    timespan = 30

    # Initializing pre and post-matrices
    pre = np.array([1, 0, 1, 1, 0, 1]).reshape(3, 2)
    post = np.array([2, 0, 0, 2, 0, 0]).reshape(3, 2)

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


"""
Copied from Exercises.
"""
def LV_hazards(x, c):
    """ Evaluates the hazard functions of the Lotka-Volterra system.

    :param x: Current system state. One-dimensional numpy array with length N.
    :param c: Vector of stochastic rate constants. One-dimensional numpy array with length V.
    :return: All reaction hazards as a one-dimensional numpy array of length V.
    """
    h = [x[0]*c[0], x[0]*x[1]*c[1], x[1]*c[2]]
    return h


main()
