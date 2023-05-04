import unittest
import numpy as np

from blume.model.CTM_alg import CtmAlg
from blume.model.post_props import Prop


class TestPostProps(unittest.TestCase):
    def test_magnetization(self):
        """
        Test that the algorithm gives a known value for the magnetization.
        """
        m_known = 0.911319  # known magnetization for beta = 0.5.
        beta = 0.5
        alg = CtmAlg(beta=beta, chi=8)
        alg.exe(tol=1e-7)
        self.assertAlmostEqual(
            m_known,
            Prop.m(alg.C, alg.T, alg.T_fixed, alg.beta, alg.a, alg.b),
            places=6,
        )

    def test_correlation(self):
        """
        Test the value for the correlation length of the system.
        """
        # known that the correlation for chi=12 should be close to this value
        cor_known = 120
        beta_c = np.log(1 + np.sqrt(2)) / 2
        alg = CtmAlg(beta=beta_c, chi=12)
        alg.exe(tol=1e-9)
        self.assertAlmostEqual(
            Prop.xi(alg.C, alg.T, alg.T_fixed, alg.beta, alg.a, alg.b),
            cor_known,
            delta=20,
        )
