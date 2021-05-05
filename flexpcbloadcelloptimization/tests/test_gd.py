"""
This file containts the tests for the interface and functionality of the 
Gradient Descent method

"""



import unittest

from optimizations.gradient_descent import GradientDescent
from optimizations.optimization_solver_base import OptimizationSolver

class TestGradientDescent(unittest.TestCase):
  def test_isOptimizationSolver(self):
    my_gd_optimizer = GradientDescent(None, None, None, None)
    self.assertTrue(issubclass(GradientDescent, OptimizationSolver))


if __name__ == '__main__':
  unittest.main()
