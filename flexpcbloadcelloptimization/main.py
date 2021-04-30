
import matplotlib.pyplot as plt
import numpy as np

from optimizations.gradient_descent import GradientDescent

def get_ndof(M):   # Given the number of masses for the system
  return 2*M*M + 1 # return number of variable entries, ndof, in dynamics matrix

def get_ndim(M): # Given the number of masses for the system
  return 2*M + 1 # return number of dimensions, ndim, of system (system dynmics matrix is (n x n)

def unpack_x(x, M):
"""
Transform x from (ndof, ) to A matrix (ndim x ndim)

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
  A.append([0, 1] + [0][0]*((M-1)*2 + 1)
  A.append([x[index] for index in range(ndim)])
  for index in range(1,M):
    A.append([0]*(index*2) + [0, 1] + [0]*((M-1-index)*2)))
    A.append([x[idx] for idx in ndim + 1 + (index-1)*(ndim-1)):(ndim + index*(ndim-1)))] + [0])
  A.append([0] * ndim)

  return A

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
  H = np.zeros(1, M)
  H[1, 2*p1_idx] = 1.0
  H[1, 2*p2_idx] = -1.0
  Q = params["Q"]
  R = params["R"]
  zk = params["zk"]
  x0 = params["x0"]
  P0 = params["P0"]
  F_in = params["F_in"]

  def objective_func(x):
    A = unpack_x(x, M)
    state = x0
    cov = P0
    meas_err_rec = [0]
    state_rec    = [state]
    cov_rec      = [cov] 

    for index in range(len(zk)):
      meas_err, state, cov = calc_Kalman(A, H, Q, R, zk[index], state, cov)
      # record and propagate
      meas_err_rec.append(meas_err)
      state_rec.append(state)
      cov_rec.append(cov)

    sum_state_err = np.linalg.norm(state_rec[:,-1] - F_in, 2)
    sum_meas_err  = np.linalg.norm(meas_err, 2)

    return sum_state_err + sum_meas_err

  return objective_func


def generate_kalman_objective_gradient(params):
  pass


def do_Calculations():
  # Load data, set initial params, make obj/obj_grad
  M = 2; # Number of masses
  obj      = generate_kalman_objective(params)
  obj_grad = generate_kalman_objective_gradient(params)
  x0 = zeros

  gradient_descent_solver = GradientDescent(obj, obj_grad, x0, obj_tol)
  gradient_descent_solver.set_params(stepsize)

  x_star, obj_hist = gradient_descent_solver.run()

  return x_star, obj_hist

def main():
  x_star, obj_hist = do_Calculations()

  # Do Plotting
  print(x_star)
  
  plt.figure()

  plt.show()
  



if __name__ == "__main__":
  main()