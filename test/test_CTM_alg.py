from src.CTM_alg import CtmAlg

import unittest


class TestCtmAlg(unittest.TestCase):
    def setUp(self):
        self.chi = 8
        self.d = 2
        self.alg = CtmAlg(beta=0.5, chi=self.chi, n_states=self.d)

    def test_eval_M(self):
        M = self.alg._eval_M()
        self.assertEqual(
            M.shape,
            (self.chi * self.d, self.chi * self.d),
            f"The shape of M {M.shape} should be equal to {self.chi * self.d, self.chi * self.d}",
        )

    def test_truncated_U(self):
        """Test if the turncated U matrix has shape (chi x chi x d)"""
        M = self.alg._eval_M()
        U_trunc = self.alg._truncated_U(M)
        self.assertEqual(
            U_trunc.shape,
            (self.chi, self.chi, self.d),
            f"The shape of Ut_trunc {U_trunc.shape} should be equal to {self.chi, self.chi, self.d}",
        )

    def test_eval_corner_edge(self):
        C = self.alg._eval_corner()
        self.assertEqual(C.shape, (self.chi, self.chi))
        T = self.alg._eval_edge()
        self.assertEqual(T.shape, (self.chi, self.chi, self.d))


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
