"""
Gradient Descent OptimizationSolver

  -=- Does Gradient Descent !
"""
from .optimization_solver_base import OptimizationSolver

class GradientDescent(OptimizationSolver):
  def set_params(self, stepsize):
    self.stepsize = stepsize

  def run(self):
    assert hasattr(self, 'stepsize'), "Stepsize must be given to set_params first!"

    #k = 0
    xk = self.x0
    gfk = self.obj_grad(self.x0)
    fk = self.obj(self.x0)

    fk_rec = [fk]
    
    if (self.obj_grad_tol is not None):
      gfk_norm = np.linalg.norm(self.obj_grad_norm)
      gfk_norm_rec = [gfk_norm]

    while (fk > self.obj_tol):
      xk = xk - self.stepsize * gfk;
      gfk = self.obj_grad(xk)
      fk = self.obj(xk)

      fk_rec.append(fk)
      
      if (self.obj_grad_tol is not None):
        gfk_norm = np.linalg.norm(gfk, self.obj_grad_norm)
        gfk_norm_rec.append(gfk_norm)
        if (gfk_norm < self.obj_grad_tol):
          break

    if (self.obj_grad_tol is not None):
      return (xk, fk_rec, gfk_norm_rec)
    else:
      return (xk, fk_rec)
    
    
