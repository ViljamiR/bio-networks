import numpy as np

def gillespieSSA(S, M, h, c, t_max):

  X = M
  t = 0
  Ts = [t]
  Xs = [X]
  V = S.shape[1]
  
  while t < t_max:

    A = h(X,c)
    A_sum = sum(A)
    
    tau = np.random.exponential(scale=1/A_sum)
    t = t + tau

    mu = np.random.choice(V, p=A /A_sum)

    X = X + S[:,mu]
    Xs.append(X)
    Ts.append(t)

  return np.array(Ts), np.array(Xs)
