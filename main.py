import numpy as np
from matplotlib import pyplot as plt
import math

from gillespie import gillespie
from gillespieSSA import gillespieSSA
from DSM import deterministic_simulation
from ODE import dimerisation_kinetics_odefun_deterministic
from CLE import CLE
from poisson_approx import poisson_approx
from dimeritisation import simulate_dimerisation
from michaelis_menten import simulate_michaelis
from auto_regulatory import simulate_auto_regulation
from lac_operon import simulate_lac_operon


def main():
    """
    Simulation for all systems:
    """
    # simulate_dimerisation()
    # simulate_michaelis()
    # simulate_auto_regulation()
    # simulate_lac_operon()


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
