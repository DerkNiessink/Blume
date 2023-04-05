import unittest
from unittest.mock import Mock
import numpy as np
from tensors import Tensors


class TestTensors(unittest.TestCase):
    def setUp(self):
        self.shape = 5
        self.delta = Tensors.kronecker_tensor(Mock, self.shape)

    def test_shape(self):
        # There are shape^4 combinations of indices of a rank-4 tensor
        for i in range(self.shape**4):
            # Convert to the right base and add leading zeros
            index = np.base_repr(i, base=self.shape).rjust(4, "0")
            val = self.delta[int(index[0]), int(index[1]), int(index[2]), int(index[3])]

            if index == index[0] * len(index):
                self.assertEqual(val, 1, f"delta({index}) should be 1")
            else:
                self.assertEqual(val, 0, f"delta({index}) should be 0")


if __name__ == "__main__":
    unittest.main()
