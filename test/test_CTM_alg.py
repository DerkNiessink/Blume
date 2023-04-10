from src.CTM_alg import CtmAlg

import unittest


class TestCtmAlg(unittest.TestCase):
    def setUp(self):
        self.chi = 8
        self.d = 2
        self.alg = CtmAlg(beta=0.5, chi=self.chi, n_states=self.d)

    def test_shapes(self):
        """
        Test the shapes of all tensors used in the CtmAlg.
        """
        M = self.alg.new_M()
        self.assertEqual(
            M.shape,
            (self.chi, self.chi, self.d, self.d),
        )

        U = self.alg.new_U(M)
        self.assertEqual(
            U.shape,
            (self.chi, self.chi, self.d),
        )

        C = self.alg.new_C(U)
        self.assertEqual(C.shape, (self.chi, self.chi))
        T = self.alg.new_T(U)
        self.assertEqual(T.shape, (self.chi, self.chi, self.d))


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
