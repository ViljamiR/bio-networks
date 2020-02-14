import numpy as np
import time
"""
Implements the Deterministic simulation
"""


def deterministic_simulation(ode, X_init, T_max, step_size, k_guess):
    X = X_init
    t = 0
    Xs = []
    Xs.append(X_init)

    Ts = [t]

    start_time = time.time()
    while t < T_max:
        o = np.array(ode(X, k_guess))
        X = X + step_size * o

        Xs.append(X)

        t = t + step_size
        Ts.append(t)
    print("Non-averaged time to execute the deterministic simulation was {0}".format(time.time() - start_time))
    return Xs, Ts
