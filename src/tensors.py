import numpy as np
from scipy.linalg import sqrtm
from ncon import ncon
from dataclasses import dataclass


@dataclass
class Tensors:
    """Class that holds tensors for conducting the corner transfer matrix
    renormalization group (CTMRG) method for evualuating the Blume-Capel model"""

    beta: float = 0.5
    Q = sqrtm(
        np.array(
            [
                [np.exp(beta), np.exp(-beta)],
                [np.exp(-beta), np.exp(beta)],
            ]
        )
    )

    @staticmethod
    def kronecker_tensor(n: int) -> np.array:
        """Returns a rank-4 kronecker delta matrix of shape n"""
        A = np.zeros((n, n, n, n))
        for i in range(n):
            A[i, i, i, i] = 1
        return A

    @staticmethod
    def random_tensor(shape: tuple) -> np.array:
        """Returns a random tensor of specific shape, which can be either rank 2
        or rank 3. The tensor is symmetric under the exchange of the first two
        indices and the values are normalized."""
        c = np.random.uniform(size=shape)
        axes = (1, 0) if len(shape) == 2 else (1, 0, 2)
        return np.maximum(c, np.transpose(c, axes))

    def a_tensor(self) -> np.array:
        """Returns the tensor representation of the partition function"""
        delta = self.kronecker_tensor(2)
        return ncon(
            [self.Q, self.Q, self.Q, self.Q, delta],
            ([-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4]),
        )

    def b_tensor(self) -> np.array:
        """Returns the tensor representation of the numerator of the magnetization
        (partition function = denominator)"""
        delta = self.kronecker_tensor(2)
        delta[1, :, :, :] *= -1.0
        return ncon(
            [self.Q, self.Q, self.Q, self.Q, delta],
            ([-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4]),
        )
