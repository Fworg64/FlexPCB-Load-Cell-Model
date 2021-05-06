"""
This file containts the tests for the interface and functionality of the 
Gradient Descent method

"""



import unittest
import numpy as np
from scipy.optimize import check_grad

import pdb

from optimizations.gradient_descent import GradientDescent
from optimizations.optimization_solver_base import OptimizationSolver

def test_obj(x):
  A = np.asarray([[2, 1], [0, 1]])
  value = x.T @ A @ x
  return value

def test_obj_grad(x):
  A = np.asarray([[2, 1], [0, 1]])
  return x.T@(A + A.T)

class TestGradientDescent(unittest.TestCase):
  def test_isOptimizationSolver(self):
    self.assertTrue(issubclass(GradientDescent, OptimizationSolver))

  def test_can_instantiate(self):
    my_gd_optimizer = GradientDescent(test_obj, test_obj_grad, np.asarray([[1],[1]]), 1e-4)
    self.assertTrue(True)

  def test_test_obj_grad(self):
    x0 = np.asarray([[1], [1]])
    x0 = x0.flatten()
    err = check_grad(test_obj, test_obj_grad, x0)
    self.assertLess(err, 1e-4)
  
  def test_can_minimize_obj_tol(self):
    x0 = np.asarray([1, 1])
    my_gd_optimizer = GradientDescent(test_obj, test_obj_grad, x0, 1e-4)
    my_gd_optimizer.set_params(0.05)
    x_star, fk_rec = my_gd_optimizer.run()
    self.assertTrue(test_obj(x_star) < 1e-4)


if __name__ == '__main__':
  unittest.main()
