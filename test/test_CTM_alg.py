from src.CTM_alg import CtmAlg

import unittest
import numpy as np
from ncon import ncon


class TestCtmAlg(unittest.TestCase):
    def setUp(self):
        self.chi = 2
        self.d = 2
        self.alg = CtmAlg(beta=0.5, chi=self.chi, n_states=self.d)

        # 1 step of the algorithm:
        self.M = self.alg.new_M()
        self.U = self.alg.new_U(self.M)
        self.C = self.alg.new_C(self.U)
        self.T = self.alg.new_T(self.U)

    def test_shapes(self):
        """
        Test the shapes of all tensors used in the CtmAlg.
        """
        with self.subTest():
            self.assertEqual(
                self.M.shape,
                (self.chi, self.d, self.chi, self.d),
                "M is not the right shape.",
            )
        with self.subTest():
            self.assertEqual(
                self.U.shape, (self.chi, self.d, self.chi), "U is not the right shape."
            )
        with self.subTest():
            self.assertEqual(
                self.C.shape, (self.chi, self.chi), "C is not the right shape."
            )
        with self.subTest():
            self.assertEqual(
                self.T.shape, (self.chi, self.chi, self.d), "T is not the right shape."
            )

    def test_symmetry(self):
        """
        Test the symmetry of the new C and T tensors after one step of the
        algorithm.
        """
        with self.subTest(msg="Corner tensor symmetry test"):
            self.assertTrue(
                np.allclose(self.C, self.C.T, rtol=1e-8, atol=1e-8),
                f"The corner tensor is not symmetric, \nC = \n{self.C}, \nC.T = \n{self.C.T}",
            )
        with self.subTest(msg="Edge tensor symmetry test"):
            self.assertTrue(
                np.allclose(
                    self.T, np.transpose(self.T, axes=(0, 1, 2)), rtol=1e-8, atol=1e-8
                ),
                f"The edge tensor is not symmetric, \nT = \n{self.T}, \nT.T = \n{self.T.T}",
            )

    def test_unitarity(self):
        """
        Test that the renormalization matrix U is unitary
        """
        product = ncon([self.U, self.U], ([-1, 1, 2], [-2, 1, 2]))
        self.assertTrue(
            np.allclose(product, np.identity(self.chi), rtol=1e-6, atol=1e-6),
            f"The renormalization tensor U is not unitary,\n U*U.T = \n{product}",
        )


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
