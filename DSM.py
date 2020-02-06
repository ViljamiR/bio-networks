import numpy as np

def deterministic_simulation(ode,X_init, T_max, step_size,k_guess, method="Euler"):
  X = X_init
  t = 0
  Xs = []
  Xs.append(X_init)

  Ts = [t]

  while t < T_max:
    o = np.array(ode(X, k_guess))
    print(o)
    X = X + step_size * o

    Xs.append(X)

    t = t + step_size
    Ts.append(t)
  return Xs, Ts

"""
Copied from Exercise 4.
Used for testing deterministic simulation during development.
"""
def odefun(x, k):
    """Differential function f. 
    
    :param x: System state. An array with length 2.
    :param k: Rate parameters. An array with length 2.
    :returns: dx/dt an array with length 2
    """
    f1 = x[0] - k[0]*x[0]*x[1]
    f2 = k[0]*x[0]*x[1]-k[1]*x[1]
    return [f1, f2]
