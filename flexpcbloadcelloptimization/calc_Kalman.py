import numpy as np
def calc_Kalman(A,H,Q,R,zk,xk,Pk):
    # Enter with:
    #   A: MxN State transition matrix -> for our purposes should be square
    #   H: LxN Output matrix - Hyper
    #   Q: MxM Input Covariance Matrix
    #   R:  LxL Measurement Covariance Matrix
    #   zk: 1xL Measurement Vector
    #   xk: 1xN State Vector
    #
    #   Exit With:
    #   yk_p: 1xL Filter Error at [k]
    #   xk_p: 1xN state estimate at [k]
    #   Pk_p: NxN Error covariance at [k]

    """
    assert A.shape[1] == xk.shape,"Incorrect Dimensions between A and xk"
    assert  A.shape[1]== Pk.shape[0],"Incorrect Dimensions between A and Pk"
    assert  A.shape[0] == Q.shape[0],"Incorrect Dimensions between A and Q"
    """
    N = xk.shape[0] #Define Length of state vector

    xk_m = A@xk                # xk time update
    Pk_m = A@Pk@A.T + Q  # Pk time update
    yk_m = zk - H@xk_m     # yk time update
    Sk = H@Pk_m@H.T + R  # Sk time update
    Sk_i = np.linalg.inv(Sk)  # Invert Sk
    Kk = Pk_m@H.T@Sk_i   # Calculate Kalman Gain for time update

    I = np.eye(N)                  # Establish identity of proper dimension
    xk_p = xk_m + Kk@yk_m # xk measurement update
    Pk_p = (I - Kk@H)@Pk_m # Pk measurement update
    yk_p = zk - H@xk_p         # yk measurement update

    return yk_p, xk_p, Pk_p
