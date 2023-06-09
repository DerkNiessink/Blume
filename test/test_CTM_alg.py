from blume.model.CTM_alg import CtmAlg
from blume.model.tensors import Tensors

import unittest
import numpy as np
from ncon import ncon


class TestIsingCtmAlg(unittest.TestCase):
    """
    Class for testing the Ctmalg from `CtmAlg.py`.
    """

    @classmethod
    def setUpClass(self):
        """
        Set up the algorithm which is used for: `test_shapes`, `test_symmetry`
        and `test_unitarity`.
        """

        self.chi = 8
        self.d = 2
        self.model = "ising"
        self.alg = CtmAlg(beta=0.5, chi=self.chi, model=self.model)
        self.alg.exe(max_steps=10)
        self.tensors = Tensors()

        # next steps of the algorithm:
        self.M = self.alg.new_M()
        self.U, _ = self.alg.new_U(self.M)
        self.untrunc_U, _ = self.alg.new_U(self.M, False)
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
                self.untrunc_U.shape,
                (self.d * self.chi, self.d, self.chi),
                "the untruncated U is not the right shape.",
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
        with self.subTest():
            self.assertTrue(
                np.allclose(self.C, self.C.T, rtol=1e-8, atol=1e-8),
                f"The corner tensor is not symmetric, \nC = \n{self.C}, \nC.T = \n{self.C.T}",
            )
        with self.subTest():
            self.assertTrue(
                np.allclose(
                    self.T, np.transpose(self.T, axes=(0, 1, 2)), rtol=1e-8, atol=1e-8
                ),
                f"The edge tensor is not symmetric, \nT = \n{self.T}, \nT.T = \n{self.T.T}",
            )

    def test_unitarity(self):
        """
        Test that the untruncated renormalization tensor U is unitary.
        """
        untrunc_product = ncon(
            [self.untrunc_U, self.untrunc_U], ([-1, 1, 2], [-2, 1, 2])
        )
        product = ncon([self.U, self.U], ([-1, 1, 2], [-2, 1, 2]))

        with self.subTest():
            self.assertTrue(
                np.allclose(
                    untrunc_product,
                    np.identity(self.chi * self.d),
                    rtol=1e-6,
                    atol=1e-6,
                ),
                f"The untruncated renormalization tensor U is not unitary,\n U*U.T = \n{untrunc_product}",
            )
        with self.subTest():
            self.assertTrue(
                np.allclose(product, np.identity(self.chi), rtol=1e-6, atol=1e-6),
                f"The truncated renormalization tensor U is not unitary,\n U*U.T = \n{product}",
            )

    def test_small_system(self):
        """
        Compare two contracted corners of the 5x5 system with one step of the algorithm.
        """

        # "Manual" calculation of two contracted corners.
        corner = ncon(
            [
                self.tensors.C_init(),
                self.tensors.T_init(),
                self.tensors.T_init(),
                self.tensors.a(),
            ],
            ([1, 2], [-3, 2, 3], [-1, 1, 4], [-2, -4, 3, 4]),
        )
        two_corners = ncon([corner, corner], ([-1, -2, 1, 2], [-3, -4, 1, 2]))

        alg = CtmAlg(beta=0.5, b_c=True, model=self.model)
        M = alg.new_M()
        untrunc_U, _ = alg.new_U(M, trunc=False)

        # Algorithm calculation of two contracted corners with two
        # untruncated renormalization tensors U in between.
        two_corners_alg = ncon(
            [M, untrunc_U, untrunc_U, M],
            ([-1, -2, 1, 2], [3, 1, 2], [3, 5, 4], [-3, -4, 5, 4]),
        )
        # The two should yield the same outcome if the untruncated U is unitary
        self.assertTrue(np.allclose(two_corners, two_corners_alg, rtol=1e-6, atol=1e-6))

    def _test_increasing_chi(self):
        """
        Test that chi increases to the given chi for a system with boundary
        conditions.
        """
        # Algorithm with boundary conditions
        alg_bc = CtmAlg(beta=0.5, chi=29, b_c=True, fixed=True, model=self.model)
        # Check that after one step chi=4
        alg_bc.exe(max_steps=1)
        with self.subTest():
            self.assertTrue(alg_bc.chi == 4)

        # Check that after that chi is and stays 16 after a few steps.
        alg_bc.exe(max_steps=10)
        with self.subTest():
            self.assertTrue(alg_bc.chi == 29)

        # Check that all tensors are computed: both the normal and fixed tensors.
        with self.subTest():
            self.assertTrue(
                alg_bc.C is not None
                and alg_bc.T is not None
                and alg_bc.T_fixed is not None
            )


class TestBlumeCtmAlg(TestIsingCtmAlg):
    @classmethod
    def setUpClass(self):
        """
        Set up the algorithm which is used for: `test_shapes`, `test_symmetry`
        and `test_unitarity`.
        """

        self.chi = 6
        self.d = 3
        self.model = "blume"
        self.alg = CtmAlg(beta=0.5, chi=self.chi, model=self.model)
        self.alg.exe(max_steps=10)
        self.tensors = Tensors(model="blume")

        # next steps of the algorithm:
        self.M = self.alg.new_M()
        self.U, _ = self.alg.new_U(self.M)
        self.untrunc_U, _ = self.alg.new_U(self.M, False)
        self.C = self.alg.new_C(self.U)
        self.T = self.alg.new_T(self.U)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
