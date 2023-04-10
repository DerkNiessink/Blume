from src.tensors import Tensors

import unittest
import numpy as np


class TestTensors(unittest.TestCase):
    def setUp(self):
        self.tensors = Tensors()

    def test_kronecker_tensor(self):

        for shape in [2, 4, 7, 9]:

            # There are shape^4 combinations of indices of a rank-4 tensor
            for i in range(shape**4):
                # Convert to the right base and add leading zeros
                index = np.base_repr(i, base=shape).rjust(4, "0")
                val = Tensors.kronecker_tensor(shape)[
                    int(index[0]), int(index[1]), int(index[2]), int(index[3])
                ]
                with self.subTest():

                    if index == index[0] * len(index):
                        self.assertEqual(val, 1, f"delta({index}) should be 1")
                    else:
                        self.assertEqual(val, 0, f"delta({index}) should be 0")

    def test_random_tensor(self):
        for shape in [(3, 3), (2, 2), (5, 5), (4, 4, 3), (8, 8, 8), (5, 5, 8)]:
            c = Tensors.random_tensor(shape)
            with self.subTest():
                # Exchange the first two indices.
                axes = (1, 0, 2) if len(shape) == 3 else (1, 0)
                self.assertTrue(
                    np.allclose(c, np.transpose(c, axes), rtol=1e-05, atol=1e-08),
                    f"random tensor of shape {shape} is not symmetric",
                )
                self.assertTrue((0 <= c.all() <= 1), "Values are not normalized")

    def test_a_tensor(self):
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
            np.allclose(theory_a, self.tensors.a_tensor(), rtol=1e-8, atol=1e-8),
            "The computed lattice tensor `a` does not agree with the theoretical tensor.",
        )

    def test_b_tensor(self):
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
            np.allclose(theory_b, self.tensors.b_tensor(), rtol=1e-8, atol=1e-8),
            "The computed lattice tensor `b` does not agree with the theoretical tensor.",
        )


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
