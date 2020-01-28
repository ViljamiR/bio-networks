import numpy as np
from matplotlib import pyplot as plt
import math

from gillespie import gillespie
from CLE import CLE
from poisson_approx import poisson_approx
def main():
  print("Hello World!")

def plot_result(T, X):
    """Visualize a Lotka-Volterra simulation result. 
    
    :param T: First return value of the 'gillespie' function.
    :param X: Second return value of the 'gillespie' function.
    :return: Nothing.
    """
    plt.figure(figsize = (10,6))
    plt.plot(T, X)
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.legend(('A (Prey)', 'B (Predator)'), loc='upper right')
    plt.show()

def generate_LV_instance():
    #Initial values.
    M=np.array([100,150])
    c=np.array([1.0,0.01,0.6])
    timespan=30

    # Initializing pre and post-matrices
    pre = np.array([1,0,1,1,0,1]).reshape(3,2)
    post = np.array([2,0,0,2,0,0]).reshape(3,2)

    # Computing Stoichiometry matrix
    S = np.transpose(post-pre)
    return M,c,S

 def LV_hazards(x, c):
    """ Evaluates the hazard functions of the Lotka-Volterra system.
        
    :param x: Current system state. One-dimensional numpy array with length N.
    :param c: Vector of stochastic rate constants. One-dimensional numpy array with length V.
    :return: All reaction hazards as a one-dimensional numpy array of length V.
    """
    h = [x[0]*c[0],x[0]*x[1]*c[1],x[1]*c[2] ]
    return h

main()