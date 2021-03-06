"""
Gradient Descent OptimizationSolver

  -=- Does Gradient Descent !
"""
import numpy as np
from .optimization_solver_base import OptimizationSolver

class GradientDescent(OptimizationSolver):
  def set_params(self, stepsize):
    self.stepsize = stepsize

  def run(self, do_print=0, max_iter=np.Inf):
    assert hasattr(self, 'stepsize'), "Stepsize must be given to set_params first!"

    k = 0
    vk = self.v0
    fk, state = self.obj(self.v0)
    gfk = self.obj_grad(self.v0,state)


    fk_rec = [fk]
    
    if (self.obj_grad_tol is not None):
      gfk_norm = np.linalg.norm(self.obj_grad_norm)
      gfk_norm_rec = [gfk_norm]

    while (fk > self.obj_tol and k < max_iter):
      vk = vk - self.stepsize * gfk;
      fk,state = self.obj(vk)
      gfk = self.obj_grad(vk,state)


      fk_rec.append(fk)
      
      if (self.obj_grad_tol is not None):
        gfk_norm = np.linalg.norm(gfk, self.obj_grad_norm)
        gfk_norm_rec.append(gfk_norm)
        if (gfk_norm < self.obj_grad_tol):
          break
      
      k = k+1

      if (do_print != 0 and k % do_print == 0):
        print("k = {0}, fk = {1}, \n vk = {2} \n gfk = {3}".format(k,fk, vk, gfk))

    if (self.obj_grad_tol is not None):
      return (vk, fk_rec, gfk_norm_rec)
    else:
      return (vk, fk_rec)
    
    
