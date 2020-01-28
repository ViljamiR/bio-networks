import numpy as np
"""
Implements the Gillespie algorithm
Inspiration for implementation from course material
and exercises.
"""
def gillespie(S, M, h, c, t_max, max_reactions):

  N = S.shape[0]
  V = S.shape[1]

  T = np.zeros(max_reactions)
  X = np.zeros((max_reactions, N))

  current_state = M
  current_time = 0.0
  
  for idx in range(0, 10^5):
    T[idx] = current_time
    X[idx, :] = current_state

    # Both species are extinct if sum of state is 0
    if sum(current_state)==0:
      print("""
      Both species went extinct at time {0} and state was {1}
      """.format(current_time, current_state))

    # hazards for current state and hazard rates
    rates = h(current_state, c)

    # sample time
    lambd = sum(rates)
    t_next = np.random.exponential(scale=1/lambd)

    # updating current time
    current_time += t_next
    if current_time > t_max:
      break # stop if timeis up
    
    reaction_index = np.random.choice(a=V, replace=True, p=rates / lambd)

    current_state = current_state + S[:, reaction_index].flatten()

  T = T[0:idx]
  X = X[0:idx, :]

  return T, X

    