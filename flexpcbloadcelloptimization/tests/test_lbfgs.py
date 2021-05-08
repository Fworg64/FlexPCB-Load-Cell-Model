"""
This file contains the tests for the interface and functionality of the 
L-BFGS method

"""


import unittest
import numpy as np
from scipy.optimize import check_grad

import pdb

from optimizations.lbfgs import LBFGS
from optimizations.optimization_solver_base import OptimizationSolver

class TestLBFGS(unittest.TestCase):
  def test_isOptimizationSolver(self):
    self.assertTrue(issubclass(LBFGS, OptimizationSolver))

if __name__== '__main__':
  unittest.main()