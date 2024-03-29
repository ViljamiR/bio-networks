import numpy as np
# Copied from exercise 3 with modifications to fulfill leap conditions.


def CLE(S, M, h, c, T):
    """ Simulates the diffusion approximation

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
    current_state = np.asarray(M, dtype=np.float64)
    X[0, :] = current_state
    A = S.T
    # Main loop
    for idx in range(1, len(T)):

        # Compute current reaction hazards
        reaction_hazards = h(current_state, c)

        # Compute Delta t
        Delta_t = T[idx] - T[idx - 1]
        tau = Delta_t
        # Sample Delta W
        d_W = np.sqrt(tau) * np.random.randn(V)

        # Update the current state

        decay_rate = 0.9
        j = 0
        while (any(i <= 1 for i in np.array(reaction_hazards) * tau) and (j < 10)):
            tau = tau * decay_rate
            j += 1

        # print(reaction_hazards)
        temp = np.array(reaction_hazards) * tau + np.sqrt(reaction_hazards)*d_W
        dx = np.matmul(S, temp)
        current_state = current_state + dx

        # Round to integers and truncate to zero
        current_state[current_state < 0] = 0

        # Record current state rounded to an integer
        X[idx, :] = np.round(current_state)

    return T, X
