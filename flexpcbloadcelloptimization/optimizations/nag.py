"""
Nesterov's Accelerated Gradient
"""

from optimizations.optimization_solver_base import OptimizationSolver
import numpy as np

class NesterovAcceleratedGradient(OptimizationSolver):
  def set_params(self,a_k, b_k):
    self.a_k = a_k
    self.b_k = b_k

  def run(self, do_print=0, max_iter=np.Inf):
    assert hasattr(self, 'a_k') # 'a_k, b_k must be given to set_params first!"
    assert hasattr(self, 'b_k') # 'a_k, b_k must be given to set_params first!"
    bCalc = False
    if self.b_k == 0:
      bCalc = True
    k = 0
    vk = self.v0
    fk,state = self.obj(vk)
    gfk = self.obj_grad(vk,state)

    yk = vk
    pk = 0
    fk_rec = [fk]

    if (self.obj_grad_tol is not None):
      gfk_norm = np.linalg.norm(self.obj_grad_norm)
      gfk_norm_rec = [gfk_norm]

    while (fk > self.obj_tol and k < max_iter):
      vk_1 = vk
      vk = yk - self.a_k*gfk

      if bCalc:
        pk_1 = pk
        pk = np.roots([-1, (pk - 1), 1])
        pk = [val for val in pk if val <= 1 and val >= 0]
        pk = pk[0]
        b_k = pk*(pk_1**2)
        yk = vk + b_k * (vk - vk_1)
      else:
        yk = vk + self.b_k * (vk - vk_1)


      fk,state = self.obj(vk)
      gfk = self.obj_grad(yk,state)

      fk_rec.append(fk)

      k+=1

      if (do_print != 0 and k % do_print == 0):
        print("k = {0}, fk = {1}, \n vk = {2} \n gfk = {3}".format(k,fk, vk, gfk))

      if (self.obj_grad_tol is not None):
        gfk_norm = np.linalg.norm(gfk, self.obj_grad_norm)
        gfk_norm_rec.append(gfk_norm)
        if (gfk_norm < self.obj_grad_tol):
          break;

    if (self.obj_grad_tol is not None):
      return (vk, fk_rec, gfk_norm_rec)
    else:
      return (vk, fk_rec)