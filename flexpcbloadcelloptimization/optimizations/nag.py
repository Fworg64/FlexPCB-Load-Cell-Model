"""
Nesterov's Accelerated Gradient
"""

from optimization_solver_base import Optimization Solver

class NesterovAcceleratedGradient(OptimizationSolver):
  def set_params(a_k, b_k):
    self.a_k = a_k
    self.b_k = b_k

  def run(self):
    assert hasattr(self, 'a_k'), 'a_k, b_k must be given to set_params first!"
    assert hasattr(self, 'b_k'), 'a_k, b_k must be given to set_params first!"

    #k = 0
    xk = self.x0
    gfk = obj_grad(x0)
    fk = obj(x0)

    fk_rec.append(fk)

    if (self.obj_grad_tol is not None):
      gfk_norm = np.linalg.norm(self.obj_grad_norm)
      gfk_norm_rec = [gfk_norm]

    while (fk > self.obj_tol):
      xk_1 = xk
      xk = yk - alpha*gfk

      bk = 1.0 - 3.0/(k+1.0)
      yk = xk + bk*(xk - xk_1)

      gfk = obj_grad(yk)
      fk  = obj(xk)

      fk_rec.append(fk)

      if (self.obj_grad_tol is not None):
        gfk_norm = np.linalg.norm(gfk, self.obj_grad_norm)
        gfk_norm_rek.append(gfk_norm)
        if (gfk_norm < self.obj_grad_tol):
          break;

    if (self.obj_grad_tol is not None):
      return (x_star, fk_rec, gfk_norm_rec)
    else:
      return (x_star, fk_rec)