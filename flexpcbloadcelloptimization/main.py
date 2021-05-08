
import matplotlib.pyplot as plt
import numpy as np
import time
import csv

from optimizations.gradient_descent import GradientDescent
from calc_Kalman import calc_Kalman
import grads as g

def get_ndof(M):   # Given the number of masses for the system
  return 2*M*M + 1 # return number of variable entries, ndof, in dynamics matrix

def get_ndim(M): # Given the number of masses for the system
  return 2*M + 1 # return number of dimensions, ndim, of system (system dynmics matrix is (n x n)

def unpack_v(v, M):
  """
  Transform v from (ndof, ) to A matrix (ndim x ndim)

  MATLAB Code:
  % Build A
  dof = 2*M*M + 1;
  ndim = 2*M + 1;

  syms a [1 dof]

  A = [0, 1, zeros(1, (M-1)*2 + 1)];
  A = [A;
       a(1:ndim)];

  for index = 1:(M-1)
       A = [A;
            zeros(1,index*2), 0, 1, zeros(1,(M-1-index)*2), 0;
            a((ndim + 1 + (index-1)*(ndim-1)):(ndim + index*(ndim-1))), 0];
  end
  A = [A; zeros(1,ndim)];
  """
  ndim = get_ndim(M)
  ndof = get_ndof(M)
  A = []
  A.append([0, 1] + [0]*((M-1)*2 + 1))
  A.append([v[index] for index in range(ndim)])
  for index in range(1,M):
    A.append([0]*(index*2) + [0, 1] + [0]*((M-1-index)*2) + [0])
    A.append([v[idx] for idx in range(ndim + (index-1)*(ndim-1),(ndim + index*(ndim-1)))] + [0])
  A.append([0] * ndim)
  A = np.asarray(A, dtype=float)
  return A

def repack_A(A,M):
  ndim = get_ndim(M)
  v = []
  v.extend(list(A[1,:]))
  for idx in range(1,M):
    v.extend(A[1+2*idx,:-1])
  v = np.asarray(v, dtype=float)
  return v

def generate_kalman_objective(params):
  """
  Given a dictionary of params, generate an objective function of a single vector for use by
  the optimization framework.

  params: dictionary of following terms
    'M'      : int, Number of masses
    'p1_idx' : int, Mass index of top plate \in [0, M-2]; 0 is the plate at the input, M is near the boundary
    'p2_idx' : int, Mass index of bottom plate \in [p1_idx+1, M-1]
    'Q'      : np array, (ndim x ndim) process covariance matrix
    'R'      : float, Measurement covariance (only one channel for now)
    'zk'     : time series list of measurements
    'x0'     : initial system state
    'P0'     : initial system covariance
    'F_in'   : measured applied force (last state ground truth)
  """
  M = params["M"]
  ndim = get_ndim(M)
  ndof = get_ndof(M)
  p1_idx = params["p1_idx"]
  p2_idx = params["p2_idx"]
  H = np.zeros((1, ndim))
  H[0, 2*p1_idx] = 1.0
  H[0, 2*p2_idx] = -1.0
  Q = params["Q"]
  R = params["R"]
  zk = params["zk"]
  x0 = params["x0"]
  P0 = params["P0"]
  F_in = params["F_in"]

  def objective_func(v):
    A = unpack_v(v, M)
    state = x0
    cov = P0
    meas_err_rec = [0]
    state_rec    = [x0]
    cov_rec      = [cov] 

    for index in range(zk.shape[1] - 1):
        meas_err, state, cov = calc_Kalman(A, H, Q, R, zk[:,index], state, cov)
        # record and propagate
        meas_err_rec.append(meas_err)
        state_rec.append(state)
        cov_rec.append(cov)

    Fest = np.asarray([state[-1] for state in state_rec])
    sum_state_err = np.linalg.norm(Fest - F_in, 2)
    sum_meas_err  = np.linalg.norm(meas_err, 2)

    return sum_state_err + sum_meas_err, state_rec
  return objective_func


def generate_kalman_objective_gradient(params):
    M = params["M"]
    ndim = get_ndim(M)
    ndof = get_ndof(M)
    Q = params["Q"]
    x0 = params["x0"]

    def objective_grad(v,Xk):
      A = unpack_v(v,M)

      obj_grad = g.gradA(A,Q,Xk)
      obj_grad = repack_A(obj_grad,M)
      return obj_grad
    return objective_grad



def do_Calculations():
  # Load data, set initial params, make obj/obj_grad

  #Initialize Dictionary Values
  M = 2; # Number of masses
  ndim = get_ndim(M)
  ndof = get_ndof(M)
  nOuts = 1
  nSamples = 100
  Q = np.zeros((ndim,ndim))
  R = np.zeros((nOuts,nOuts))
  zk = np.random.rand(nOuts, nSamples)
  v0 = np.arange(1,ndof+1)  # v0 is the initialization of our optimization variable v (to avoid conflict with state variable x)
  x0 = np.zeros(ndim)
  P0 = np.eye(ndim)
  F_in = np.zeros(nSamples)

  #  Load Dictionary values
  R[0,0] = 1
  Q = np.diag([0.1, 0.1, 0.1, 0.1, 1])
  # Load Dictionary
  params = {'M': M, 'p1_idx': 0,'p2_idx':1, 'Q': Q, 'R': R, 'zk': zk, 'v0': v0, 'x0':x0, 'P0': P0, 'F_in': F_in}

  obj          = generate_kalman_objective(params)
  obj_grad = generate_kalman_objective_gradient(params)

  stepsize = 0.1
  obj_tol = 1e-6

  gradient_descent_solver = GradientDescent(obj, obj_grad, v0, obj_tol)
  gradient_descent_solver.set_params(stepsize)

  x_star, obj_hist = gradient_descent_solver.run()

  return x_star, obj_hist

def main():
  x_star, obj_hist = do_Calculations()

  # Do Plotting
  print(x_star)
  
  plt.figure()
  plt.plot(obj_hist)
  plt.show()
  



if __name__ == "__main__":
  main()