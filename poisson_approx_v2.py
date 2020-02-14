import numpy as np

def poisson_approx(S, M, h, c, T_max):
    """ Simulates the tau-leap approximation (Poisson approximation)

    :param S: Stoichiometry matrix. Two-dimensional numpy array with size N x V, where
        N is the number of species and V is the number of reactions.
    :param M: Initial state vector. One-dimensional numpy array with length N.
    :param h: Function that evaluates all reaction hazards and returns
        them as a one-dimensional numpy array of length V.
    :param c: Vector of stochastic rate constants. One-dimensional numpy array with length V.
    :param T: Time span, a vector giving the time discretisation. One-dimensional numpy array.

    :return: 
        T - One-dimensional numpy array containing the sample times. Returns T unchanged.
        X - Two dimensional numpy array, where the rows contain the system state at each event time.

    """
    N = S.shape[0]
    V = S.shape[1]
    X = np.zeros((len(T), N))
    current_state = np.copy(M)
    X[0, :] = current_state

    t = 1
    # Main loop
    while t < T_max:

        # Compute current reaction hazards
        A = h(current_state, c)

        # Compute Delta t
        Delta_t = T[idx] - T[idx - 1]

        # Sample Poisson random numbers
        r_i = [np.random.poisson(i * Delta_t) for i in A]

        # Update the current state
        current_state = current_state + np.matmul(S, r_i)

        #  Truncate to zero
        current_state[current_state < 0] = 0

        # Record current state
        X[idx, :] = current_state

    return T, X
