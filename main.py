import numpy as np
from matplotlib import pyplot as plt
import math

from gillespie import gillespie
from DSM import deterministic_simulation
from DSM import odefun
from DSM import dimerisation_kinetics_odefun_stochastic
from DSM import dimerisation_kinetics_odefun_deterministic
from CLE import CLE
#from poisson_approx import poisson_approx


def main():
    #M, c, S = generate_LV_instance()
    #T1, X1 = gillespie(S,M,LV_hazards, c,t_max=30, max_reactions=int(1e5))
    #plot_result(T1,X1)

    #Nt = 500
    #T, X1 = CLE(S, M, LV_hazards, c, np.linspace(0, 30, Nt))
    #plot_result(T, X1)

    #X_init = np.array([1, 1.5])
    #T_max = 12
    #step_size = 0.01
    #k_guess = np.array([1.5, 3.0])
    #X_dsm, T_dsm = deterministic_simulation(odefun, X_init, T_max, step_size, k_guess)
    #plot_result(T_dsm, X_dsm)

    P_init = np.array([5*10**(-7), 0])
    T_max = 12
    step_size = 0.01
    
    k_guess = np.array([5e5, 0.2])
    P_dsm, T_dsm = deterministic_simulation(dimerisation_kinetics_odefun_deterministic, P_init, T_max, step_size, k_guess)
    plot_result(T_dsm, P_dsm)

"""
Copied from Exercises to visualize data.
"""
def plot_result(T, X):
    """Visualize a Lotka-Volterra simulation result. 

    :param T: First return value of the 'gillespie' function.
    :param X: Second return value of the 'gillespie' function.
    :return: Nothing.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(T, X)
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.legend(('A (Prey)', 'B (Predator)'), loc='upper right')
    plt.show()


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
