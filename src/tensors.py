import numpy as np
from scipy.linalg import sqrtm
from ncon import ncon
from dataclasses import dataclass


@dataclass
class Tensors:
    """Class that holds tensors for conducting the corner transfer matrix
    renormalization group (CTMRG) method for evualuating the Ising model"""

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
        """
        Returns a rank-4 kronecker delta matrix of shape n
        """
        A = np.zeros((n, n, n, n))
        for i in range(n):
            A[i, i, i, i] = 1
        return A

    @staticmethod
    def random_tensor(shape: tuple) -> np.array:
        """
        Returns a random tensor of specific shape, which can be either rank 2
        or rank 3. The tensor is symmetric under the exchange of the first two
        indices and the values are normalized.
        """
        c = np.random.uniform(size=shape)
        return Methods.symmetrize(c)

    def a_tensor(self) -> np.array:
        """
        Returns the tensor representation of a single lattice site in the
        partition function.
        """
        delta = self.kronecker_tensor(2)
        return ncon(
            [self.Q, self.Q, self.Q, self.Q, delta],
            ([-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4]),
        )

    def b_tensor(self) -> np.array:
        """
        Returns the tensor representation for a single lattice site in the
        numerator of the magnetization (partition function = denominator).
        """
        delta = self.kronecker_tensor(2)
        delta[1, :, :, :] *= -1.0
        return ncon(
            [self.Q, self.Q, self.Q, self.Q, delta],
            ([-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4]),
        )


@dataclass
class Methods:
    """This class contains methods for np.arrays, required for the CTM algorithm."""

    @staticmethod
    def symmetrize(M: np.array) -> np.array:
        """
        Symmetrize the array about the first two axes. Only works for 2 or 3
        dimensional arrays.
        """
        if len(M.shape) != 2 and len(M.shape) != 3:
            raise Exception("M has to a 2 or 3 dimensional array.")

        axes = (1, 0) if len(M.shape) == 2 else (1, 0, 2)
        return (M + np.transpose(M, axes)) / 2

    @staticmethod
    def normalize(M: np.array) -> np.array:
        """
        Divide all elements in the given array by its largest value.
        """
        return M / np.amax(M)
