import numpy as np
from matplotlib import pyplot as plt


def bin_linear(T, X, bin_width, T_max):

    space = np.linspace(0, T_max, T_max / bin_width)
    binned = []
    for bin_start in space:
        bin_end = bin_start + bin_width
        data_in_range = [x for i, x in enumerate(X) if (
            T[i] >= bin_start and T[i] <= bin_end)]
        binned.append(np.mean(data_in_range, axis=0))

    return space, np.array(binned)


def simulate_many(S, M, lac_operon_hazards, c, T_max, P_NAMES, sim_type, sim_function, Nt=0):
    N = 100
    averaged = []
    time = []
    for i in range(N):
        if sim_type == "Gillespie":
            T_g, X_g = sim_function(S, M, lac_operon_hazards, c, t_max=T_max)
            T, X = bin_linear(T_g, X_g, 30, T_max)
        elif sim_type == "Poisson":
            T, X = sim_function(S, M, lac_operon_hazards,
                                c, np.linspace(1, T_max, Nt))
        elif sim_type == "CLE":
            T, X = sim_function(S, M, lac_operon_hazards,
                                c, np.linspace(1, T_max, Nt))
        averaged.append(X)
        time = T

    av = np.mean(averaged, axis=0)
    plot_result(time, av, title="Averaged {0} lac operon".format(
        sim_type), legend=P_NAMES)


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
    plt.ylabel('Number of molecules')
    plt.legend(legend, loc='upper right')
    plt.show()
