
import matplotlib.pyplot as plt
import numpy as np
import time

import control
import control.matlab
import scipy
import csv
from pathlib import Path

import sys
import traceback
import argparse

from sensordataproc import sensor_interp
from optimizations.gradient_descent import GradientDescent
from optimizations.nag import NesterovAcceleratedGradient as NAG
from optimizations.lbfgs import LBFGS
from calc_Kalman import calc_Kalman
import grads as g


import pdb

method_keys = ["GD", "NAG", "LBF"]

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
  A = np.array(A)
  v.extend(A[1,:])
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
    system_state = x0.reshape((-1,1)) # make state a column vector
    cov = P0
    meas_err_rec = [0]
    state_rec    = [system_state]
    cov_rec      = [cov] 

    # Convert from cont to disc
    B = np.zeros((system_state.shape[0], 1))
    C = H
    D = np.zeros((1,1))
    system = control.StateSpace(A, B, C, D)
    dis_system = control.matlab.c2d(system, 8.225e-3)

    for index in range(zk.shape[1] - 1):
        meas_err, system_state, cov = calc_Kalman(dis_system.A, H, Q, R, zk[:,index], system_state, cov)
        # record and propagate
        meas_err_rec.append(meas_err)
        state_rec.append(system_state)
        cov_rec.append(cov)

    Fest = np.asarray([state[-1] for state in state_rec], dtype=float).T
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

    p1_idx = params["p1_idx"]
    p2_idx = params["p2_idx"]
    H = np.zeros((1, ndim))
    H[0, 2*p1_idx] = 1.0
    H[0, 2*p2_idx] = -1.0

    dt = 8.225e-3 # add to params

    def objective_grad(v,Xk):
      A = unpack_v(v,M)
      Xk = np.array(Xk)
      Xk = Xk.reshape((-1, ndim)) # Drop spurrious dimension to be nSamples X ndim
      # Convert from cont to disc
      B = np.zeros((x0.shape[0],1))
      C = H
      D = np.zeros((1,1))
      system = control.StateSpace(A, B, C, D)
      dis_system = control.matlab.c2d(system, dt)

      obj_grad = g.gradA(dis_system.A,Q,Xk) # gradient w.r.t. discrete system matrix: pv/pAd
      # From chain rule, pv/pAd * pAd/pA = pv/pA
      # note that Ad = exp(A*dt) => pAd/pA = dt * exp(A*dt)
      # and then pv/pA = pv/pAd * (dt * exp(A*dt))
      obj_grad = obj_grad * (dt * scipy.linalg.expm(A * dt))
      obj_grad = repack_A(obj_grad, M) # Repack as vector
      return obj_grad
    return objective_grad

def generate_params(data, M):
  #Initialize Dictionary Values
  # M: Number of masses
  # data: Dictionary of experiment data with "common_um": sensor data, "common_kn" ground truth
  ndim = get_ndim(M) # number of dimensions in system
  ndof = get_ndof(M) # number of free CT coefficients
  nOuts = 1 # number of measurement channels
  nSamples = 100 # how many samples to consider
  Q = np.zeros((ndim,ndim))
  R = np.zeros((nOuts,nOuts))
  # Load the data from a list in a dictionary
  # Measurements for system
  zk = np.asarray(data["common_um"], dtype=float).reshape((nOuts,nSamples)) 
 # v0 is the initialization of optimization variable v (to avoid conflict with state variable x)
  v0 = np.array([-1500.0,-200.0,1000.0, 1000.0, 500.0, 100.0, 100.0, -1500.0, -100.0]) 
  x0 = np.zeros(ndim) # dynamic system state variable
  P0 = np.eye(ndim)
  F_in = np.asarray(data["common_kn"], dtype=float) # np.zeros(nSamples) # True applied force

  p1_idx = 0
  p2_idx = 1

  #  Load Dictionary values
  R[0,0] = 1
  Q = 20.0 * np.eye(ndim)
  Q[:-1, :-1] *= 2.01 # set all diagonal, except last, to 2.01
  # Load Dictionary
  params = {'M': M, 'p1_idx': p1_idx,'p2_idx':p2_idx, 'Q': Q, 'R': R,
          'zk': zk, 'v0': v0, 'x0':x0, 'P0': P0, 'F_in': F_in}
  return params

def get_solver_params_from_specs(specs):
  solver_params = {"GD":(specs['step'],), "NAG":(specs['alpha'], specs['beta']), "LBF":(specs['mem'],)}
  return solver_params

def do_Calculations(data, specs):
  """
   Given the data dictionary, does the calculations for each optimization method

   data - dictionary with keys:
      "tbs" - list of time stamps for each sample
      "common_kn" - list of true applied force
      "common_um" - list of calculated displacement values from the capacitive readings

   specs - the dictionary of parameters parsed from the command line
  """
  params       = generate_params(data, 2)
  obj          = generate_kalman_objective(params)
  obj_grad     = generate_kalman_objective_gradient(params)

  obj_tol  = 1.0e-8

  x_stars = {}
  obj_hists = {}
  optimal_hists = {}
  times = {}
  solvers = {"GD":GradientDescent, "NAG":NAG, "LBF":LBFGS}
  solver_params = get_solver_params_from_specs(specs)
  for method in method_keys:
    if specs[method]:
      method_solver = solvers[method](obj, obj_grad, params['v0'], obj_tol)
      # pack params acording to method
      method_solver.set_params(*solver_params[method])
      print("Going to run {0}".format(method))
      try:
        ti = time.time()
        x_stars[method], obj_hists[method] = method_solver.run(specs['update'], specs["max_iter"])
        to = time.time()
        times[method] = to - ti
        _, these_states = obj(x_stars[method])
        state_array = np.array(these_states)
        optimal_hists[method] = [state[-1] for state in state_array] # Force estimate is last state
      except:
        print("\nError occured with {0}!\n".format(method))
        print(traceback.print_tb(sys.exc_info()[2]))
        print(sys.exc_info())


  # return dictionary of optimal values, objective history, and times for each method. 
  # Also return "optimal" Force estimate
  return x_stars, obj_hists, times, optimal_hists

  x_starGD = []
  x_starNAG = []
  x_starLBFGS = []
  obj_histGD = []
  obj_histNAG = []
  obj_histLBFGS = []

  if specs['GD']:
    gradient_descent_solver = GradientDescent(obj, obj_grad, params['v0'], obj_tol)
    gradient_descent_solver.set_params(specs['step'])
    # Run solvers
    print("Going to run Gradient Descent")
    ti = time.time()
    x_starGD, obj_histGD = gradient_descent_solver.run(specs['update'], specs["max_iter"])
    to = time.time()
    times['GD'] = to - ti

  if specs['NAG']:
    NAG_solver = NAG(obj, obj_grad, v0, obj_tol)
    if specs['beta'] is not None:
      NAG_solver.set_params(specs['alpha'], specs['beta'])
      print(f"Going to run NAG, alpha = {specs['alpha']}, beta = {specs['beta']}")
    else:
      NAG_solver.set_params(specs['alpha'],0)
      print(f"Going to run NAG, alpha = {specs['alpha']}, beta calculated")
    ti = time.time()
    x_starNAG, obj_histNAG = NAG_solver.run(specs['update'], specs["max_iter"])
    to = time.time()
    times['NAG'] = to - ti

  if specs['LBF']:
    LBF_solver = LBFGS(obj, obj_grad, v0, obj_tol)
    LBF_solver.set_params(specs['mem'])

    ti = time.time()
    x_starLBFGS, obj_histLBFGS = LBF_solver.run(specs['update'], specs["max_iter"])
    to = time.time()
    times['LBF'] = to - ti


  x_star = [x_starNAG,x_starGD,x_starLBFGS]
  obj_hist = [obj_histNAG,obj_histGD,obj_histLBFGS]

  # Return the list of optimal values, the list of history lists,
  #          the dictionary of times, and the objective function (to generate state history).
  return x_star, obj_hist, times, obj

def main():
  parser = argparse.ArgumentParser(description="Specify which methods to run")

  parser.add_argument("-g","--GD",help="Run Gradient Descent", action='store_true')
  parser.add_argument("-n", "--NAG", help="Run NAG", action='store_true')
  parser.add_argument("-l", "--LBF", help="Run LBFGS", action='store_true')
  parser.add_argument("-s", "--step", help="Specify step length for GD, default is 1.0e4", 
    type=float, default=1.0e4)
  parser.add_argument("-a", "--alpha", help="Specify alpha for NAG, default is 1", type=float, default=1)
  parser.add_argument("-b", "--beta", help="Specify beta for NAG, no input -> beta calculated",
    type=float, default=0.0)
  parser.add_argument("-m", "--mem", help="Specify memory for LBFGS, default is 2", type=int, default=2)
  parser.add_argument("-u","--update", default=25, type=int, 
          help='Show system performance every UPDATE iterations, 0 suppresses printing, default is 25')
  parser.add_argument("-k", "--max_iter", help="Specify max # of iterations, default is 100",
                      type=int, default=100)
  runDet = vars(parser.parse_args())

  print(runDet)
  big_plateA = 0.001527
  lil_plateA = 0.001164
  all_data = sensor_interp.read_experiment_data("all_exp.txt")
  params_chan0 = {"area": lil_plateA, "L": 4.4684e-7, "cfilt": 1.9822e-8, "er": 3.7159}
  all_data["common_um"] = \
    sensor_interp.calculate_distance_from_readings_and_params(all_data["common_chan0"], params_chan0)
  data = {key: all_data[key][100:200] for key in ["tbs", "common_kn", "common_um"]}
  data['common_kn'] = np.asarray(data['common_kn'], dtype=float)
  data['common_kn'] *= 1.0e-8
  # Make the force state and measurements zero mean.
  data['common_kn'] -= data['common_kn'][0]
  data['common_um'] -= data['common_um'][0]
  x_star, obj_hist, times, state_hist = do_Calculations(data, runDet)

  # Save the list of optimal values, the list of history lists, 
  # the dictionary of times, the optimal state history
  solver_params = get_solver_params_from_specs(runDet)

  for method in method_keys:
    if runDet[method]:
      try:
        save_experiment_data(method, solver_params[method], x_star[method], obj_hist[method],
                           times[method], state_hist[method], data['common_kn'].tolist())
      except:
        print("\nError saving {0}!\n".format(method))

  print("\n\nDone!, Total time was {0}".format(sum(times.values())))
  

def save_experiment_data(method_name, method_params_tuple, optimal_point, objective_history,
                         solve_time, state_history, ground_truth):
  Path("out").mkdir(parents=True, exist_ok=True)
  filename = "out/" + method_name + time.strftime("%Y%m%d-%H%M%S") + ".txt"
  with open(filename, 'w', newline='') as csvfile:
    my_writer = csv.writer(csvfile, delimiter=',')
    my_writer.writerow([method_name, method_params_tuple])
    my_writer.writerow(optimal_point)
    my_writer.writerow(["Solve Time: ", solve_time])
    my_writer.writerow(["Objective History"])
    for obj in objective_history:
      my_writer.writerow([obj])
    my_writer.writerow(["State History"])
    for obj in state_history:
      my_writer.writerow(obj)
    my_writer.writerow(["Ground Truth"])
    for obj in ground_truth:
      my_writer.writerow([obj])
  print("Saved {0} to {1}".format(method_name, filename))



if __name__ == "__main__":
  main()