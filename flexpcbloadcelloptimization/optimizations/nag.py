"""
Nesterov's Accelerated Gradient
"""

from optimization_solver_base import OptimizationSolver
import numpy as np

class NesterovAcceleratedGradient(OptimizationSolver):
  def set_params(self,a_k, b_k):
    self.a_k = a_k
    self.b_k = b_k

  def run(self):
    assert hasattr(self, 'a_k') # 'a_k, b_k must be given to set_params first!"
    assert hasattr(self, 'b_k') # 'a_k, b_k must be given to set_params first!"

    #k = 0
    vk = self.v0
    gfk = self.obj_grad(vk)
    fk = self.obj(vk)
    yk = vk

    fk_rec = [fk]

    if (self.obj_grad_tol is not None):
      gfk_norm = np.linalg.norm(self.obj_grad_norm)
      gfk_norm_rec = [gfk_norm]

    while (fk > self.obj_tol):
      vk_1 = vk
      vk = yk - self.a_k*gfk

      # bk = 1.0 - 3.0/(k+1.0)
      yk = vk + self.b_k*(vk - vk_1)

      gfk = self.obj_grad(yk)
      fk  = self.obj(vk)

      fk_rec.append(fk)

      if (self.obj_grad_tol is not None):
        gfk_norm = np.linalg.norm(gfk, self.obj_grad_norm)
        gfk_norm_rec.append(gfk_norm)
        if (gfk_norm < self.obj_grad_tol):
          break;

    if (self.obj_grad_tol is not None):
      return (vk, fk_rec, gfk_norm_rec)
    else:
      return (vk, fk_rec)