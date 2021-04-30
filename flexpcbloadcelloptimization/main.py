
import matplotlib.pyplot as plt

from optimizations.gradient_descent import GradientDescent

def get_ndof(M):   # Given the number of masses for the system
  return 2*M*M + 1 # return number of variable entries, ndof, in dynamics matrix

def get_ndim(M): # Given the number of masses for the system
  return 2*M + 1 # return number of dimensions, ndim, of system (system dynmics matrix is (n x n)

def unpackX(x):
"""
Transform x from (ndof, ) to A matrix (ndim x ndim)
"""
  pass

def generate_kalman_objective(params):
  pass

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