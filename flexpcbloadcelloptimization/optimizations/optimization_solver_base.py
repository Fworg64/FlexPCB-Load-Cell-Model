"""
Base class for optimization routines

Initialize and then set additional parameters as defined by the specific optimizer.

Call run to get the state and measurement residual history, the optimal x value, and if
used, the gradient norm history

"""

class OptimizationSolver:
  def __init__(self, obj, obj_grad, v0, obj_tol, obj_grad_tol=None, obj_grad_norm=2):
    self.obj = obj                # A function handle to the objective as a function of (N,) dimension numpy array
    self.obj_grad = obj_grad      # A function handle to the gradient of the objective, same argument
    self.v0 = v0                  # The initial iterate
    self.obj_tol = obj_tol        # The desired value of the objective function
    self.obj_grad_tol = obj_grad_tol  # Optional, desired value of gradient norm
    self.obj_grad_norm = obj_grad_norm  # Optional, np.linalg.norm type to use

  def set_params(self, params):
    raise NotImplementedError() # Must be overidden by base class

  def run(self):
    raise NotImplementedError() # Must be overidden by base class

