from src.tensors import Tensors

import unittest
import numpy as np


class TestTensors(unittest.TestCase):
    def setUp(self):
        self.tensors = Tensors()

    def test_delta(self):
        """
        Test that the delta tensor gives 1 if all indices are equal, else gives 0.
        """
        shapes = [(2, 2), (4, 4), (2, 2, 2), (3, 3, 3), (2, 2, 2, 2)]
        for shape in shapes:
            delta = Tensors.delta(shape)
            for index in np.ndindex(shape):
                with self.subTest():
                    # If all indices are the same.
                    if index.count(index[0]) == len(index):
                        self.assertEqual(delta[index], 1, f"delta({index}) should be 1")
                    else:
                        self.assertEqual(delta[index], 0, f"delta({index}) should be 0")

    def test_random(self):
        """
        Test the symmetry and normality of the random tensor.
        """
        for shape in [(3, 3), (2, 2), (5, 5), (4, 4, 3), (8, 8, 8), (5, 5, 8)]:
            c = Tensors.random(shape)
            with self.subTest():
                # Exchange the first two indices.
                axes = (1, 0, 2) if len(shape) == 3 else (1, 0)
                self.assertTrue(
                    np.allclose(c, np.transpose(c, axes), rtol=1e-05, atol=1e-08),
                    f"random tensor of shape {shape} is not symmetric",
                )
                self.assertTrue((0 <= c.all() <= 1), "Values are not normalized")

    def test_a(self):
        """
        Test that the a tensor equals the theoretical tensor.
        """
        theory_a = np.array(
            [
                [
                    [[2.53434211, 0.5], [0.5, 0.18393972]],
                    [[0.5, 0.18393972], [0.18393972, 0.5]],
                ],
                [
                    [[0.5, 0.18393972], [0.18393972, 0.5]],
                    [[0.18393972, 0.5], [0.5, 2.53434211]],
                ],
            ]
        )
        self.assertTrue(
            np.allclose(theory_a, self.tensors.a(), rtol=1e-8, atol=1e-8),
            "The computed lattice tensor `a` does not agree with the theoretical tensor.",
        )

    def test_b(self):
        """
        Test that the b tensor equals the theoretical tensor
        """
        theory_b = np.array(
            [
                [
                    [[2.52765822, 0.46493675], [0.46493675, 0]],
                    [[0.46493675, 0], [0, -0.46493675]],
                ],
                [
                    [[0.46493675, 0], [0, -0.46493675]],
                    [[0, -0.46493675], [-0.46493675, -2.52765822]],
                ],
            ]
        )
        self.assertTrue(
            np.allclose(theory_b, self.tensors.b(), rtol=1e-8, atol=1e-8),
            "The computed lattice tensor `b` does not agree with the theoretical tensor.",
        )

    def test_C_init(self):
        """
        Test the shape of C_init.
        """
        self.assertEqual(
            self.tensors.C_init().shape,
            (self.tensors.d, self.tensors.d),
            "C_init does not have the right shape.",
        )

    def test_T_init(self):
        """
        Test the shape of T_init.
        """
        self.assertEqual(
            self.tensors.T_init().shape,
            (self.tensors.d, self.tensors.d, self.tensors.d),
            "T_init does not have the right shape.",
        )


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
