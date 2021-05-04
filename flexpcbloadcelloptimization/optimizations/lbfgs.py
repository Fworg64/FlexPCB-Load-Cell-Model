"""
Limited Memory BFGS
"""
import numpy as np

from optimization_solver_base import OptimizationSolver

def Inverse_Hessian_Direction(Yk, Sk, recent_grad):
  q = np.array(recent_grad)
  Pk = np.zeros((len(Yk), 1)) #seq of scalar
  alphas = np.zeros((len(Yk), 1)) #seq of scalar

  for index in range(0, len(Yk)): #from most recent to least
      Pk[index] = (1.0/(np.transpose(Yk[index]).dot(Sk[index])))
      alphas[index] = Pk[index].dot(np.transpose(Sk[index])).dot(q)
      #print("yk = ", Yk[index])
      #print("alphak = ", alphas[index])
      q = np.subtract(q,(alphas[index])*Yk[index])
  gamma = (np.transpose(Sk[-1]).dot(Yk[-1]) /
           np.transpose(Yk[-1]).dot(Yk[-1]) )
  #print("q = ", q)
  #print("gamma = ", gamma)
  z = q.dot(gamma)
  #print("zcalc = ", z)
  for index in range(len(Yk) -1, 0, -1): #from old to new
      Beta = Pk[index].dot(np.transpose(Yk[index])).dot(z)
      epp = alphas[index] - Beta
      z = np.add(z,epp*Sk[index])
      #print("epp = ", epp)
      #print("Sk = ", Sk[index])
      #print("Znew = ", z)
  return z

class LBFGS(OptimizationSolver):
  def set_params(self, memory):
    self.memory = memory

  def run(self):
    assert hasattr(self, 'memory'), "Memory must be given to set_params first!"

    #k = 0
    xk = self.x0
    gfk = obj_grad(x0)
    fk = obj(x0)

    fk_rec = [fk]
    
    if (self.obj_grad_tol is not None):
      gfk_norm = np.linalg.norm(self.obj_grad_norm)
      gfk_norm_rec = [gfk_norm]

    while (fk > self.obj_tol):
      xk = xk - stepsize * gfk;
      gfk = obj_grad(xk)
      fk = obj(x0)

      fk_rec.append(fk)
      
      if (self.obj_grad_tol is not None):
        gfk_norm = np.linalg.norm(gfk, self.obj_grad_norm)
        gfk_norm_rec.append(gfk_norm)
        if (gfk_norm < self.obj_grad_tol):
          break

    if (self.obj_grad_tol is not None):
      return (x_star, fk_rec, gfk_norm_rec)
    else:
      return (x_star, fk_rec)
    
    


# Define L-BFGS



def lbfgs_rosen(ALPHA, MEM):
  #L-BFGS Method
  #STEP = .25
  #MEM = 15 #number of iterations to rememeber
  N = 100
  Xk = [np.ones(N)]
  Xk[0] = np.ones((N,1)).dot(-1.0)
  #0 get a starter iteration from gradient descent
  gradF = Rosenbrock_grad( Xk[0], ALPHA )
  Xk.insert(0,np.subtract(Xk[0], gradF.dot(.2)))
  #print(Xk[0], " xk0")
  #print(Xk[1], " xk1")
  #print(gradF.dot(.2))
  #print(np.subtract(Xk[0], gradF.dot(.2)))
  print("f= ", Rosenbrock_func(Xk[1], ALPHA))

  Sk = [np.array(np.subtract(Xk[0], Xk[1]))]
  temper = np.subtract(Rosenbrock_grad(Xk[0], ALPHA),Rosenbrock_grad(Xk[1], ALPHA))
  Yk = [np.array(temper)]

  last_grad = Rosenbrock_grad(Xk[0], ALPHA)
  #print("last_grad = ", last_grad)
  num_iter = 60
  Fvalrecord = np.zeros((num_iter,1))
  for iterations in range(0,num_iter):
    #compute search dir
    z = Inverse_Hessian_Direction(Yk, Sk, last_grad)
    #Xk.insert(0,np.subtract(Xk[0],STEP*z)) #line search here
    div = 200
    steps = np.zeros((div,1))
    potentialXkfun = np.zeros((div,1))
    for b_index in range(0,div):
        steps[b_index] = steps[b_index-1] + 5/div
        potentialXkfun[b_index] = Rosenbrock_func(
                            np.subtract(Xk[0],steps[b_index]*z), ALPHA)
    Xk.insert(0,np.subtract(Xk[0],
                            steps[np.argmin(potentialXkfun)]*z)) #line search here

    #print("xk",iterations,"  ", Xk[-1])
    #extend Yk, Sk
    Sk.insert(0,np.subtract(Xk[0], Xk[1]))
    Yk.insert(0,np.subtract(Rosenbrock_grad(Xk[0], ALPHA), Rosenbrock_grad(Xk[1], ALPHA)))
    last_grad = Rosenbrock_grad(Xk[0], ALPHA)
    #trim Yk, Sk for L of L-BFGS
    #print(len(Yk))
    if (len(Yk) > MEM):
      Yk = Yk[0:MEM]
      Sk = Sk[0:MEM]
    #print("Xk= ", Xk[-1])
    #print("f(Xk)= ",Rosenbrock_func(Xk[-1]))
    if Rosenbrock_func(Xk[0], ALPHA)[0] > 10**(-6):
      Fvalrecord[iterations] = Rosenbrock_func(Xk[0], ALPHA)
    else:
      Fvalrecord[iterations] = 10**(-6)

  return Fvalrecord
    

