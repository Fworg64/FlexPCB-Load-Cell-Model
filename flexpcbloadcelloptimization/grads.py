import numpy as np
def gradA(A,Q,Xk):
    """
    Enter with:
    A: [NxN] State Transiton Matrix
    Q: [NxN] Input Covariance Matrix
    xk: [NxT]  State Matrix: Columns k is the state vector at K

    Exit with:
    grad: the Gradient of the Objective function w.r.t. A
    """
    Xk = np.asarray(Xk)
    Xk = Xk.T
    T = Xk.shape[1]
    sumOut = np.zeros_like(A)
    for i in range(T-1):
        sumOut  += 2.0*np.outer(Xk[:, i+1], Xk[:, i]) - 2.0*A@np.outer(Xk[:, i], Xk[:, i])

    grad = .5*np.linalg.inv(Q)*sumOut
    return grad


def gradQ(A,Q,Xk):
    """
    Enter with:
    A: [NxN] State Transiton Matrix
    Q: [NxN] Input Covariance Matrix
    xk: [NxT]  State Matrix: Columns k is the state vector at K

    Exit with:
    grad: the Gradient of the Objective function w.r.t. Qinv
    """
    Xk = np.asarray(Xk)
    Xk = Xk.T
    T = Xk.shape[1]
    sumOut = np.zeros_like(A)
    for i in range(T):
        sumOut += np.outer(Xk[:, i+1], Xk[:, i+1]) - np.outer(Xk[:, i+1], Xk[:, i])@A.T - A@np.outer(Xk[:, i], Xk[:, i+1]) + A@np.outer(Xk[:, i], Xk[:, i])@A.T

    grad = 0.5*T*Q  - 0.5*sumOut.T
    return grad

def gradR(H,R,Xk,Yk):
    """
    Enter with:
    H: [MxN] State Transiton Matrix
    R: [MxM] Measurement Covariance Matrix
    Xk: [NxT]  State Matrix: Column k is the state vector at K
    Yk: [MxT]  Output Matrix: Column k is the output vector at K

    Exit with:
    grad: the Gradient of the Objective function w.r.t. Rinv
    """
    Xk = np.asarray(Xk)
    Xk = Xk.T
    T = Xk.shape[0]
    sumOut = np.zeros_like(R)
    for i in range(T):
        sumOut += np.outer(Yk[:, i], Yk[:, i]) - np.outer(Yk[:, i], Xk[:, i])@H.T - H@np.outer(Xk[:, i], Yk[:, i]) + H@np.outer(Xk[:, i], Xk[:, i])@H.T

    grad = 0.5*(T+1)*R  - 0.5*sumOut.T
    return grad