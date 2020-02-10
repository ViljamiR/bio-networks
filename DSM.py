import numpy as np


def deterministic_simulation(ode, X_init, T_max, step_size, k_guess):
    X = X_init
    print(X_init)
    t = 0
    Xs = []
    Xs.append(X_init)

    Ts = [t]

    while t < T_max:
        o = np.array(ode(X, k_guess))
        X = X + step_size * o

        Xs.append(X)

        t = t + step_size
        Ts.append(t)
    return Xs, Ts
