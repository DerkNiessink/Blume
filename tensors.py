import numpy as np
from scipy.linalg import sqrtm
from ncon import ncon

from dataclasses import dataclass


@dataclass
class Tensors:
    """Class that holds tensors for conducting the corner transfer matrix
    renormalization group (CTMRG) method for evualuating the Blume-Capel model"""

    beta: float = 0.5

    def transfer_matrix(self) -> np.array:
        """Returns the transfer matrix"""
        return sqrtm(
            np.array(
                [
                    [np.exp(self.beta), np.exp(-self.beta)],
                    [np.exp(-self.beta), np.exp(self.beta)],
                ]
            )
        )

    def kronecker_matrix(self, n) -> np.array:
        """Returns a rank-4 kronecker delta matrix of shape n"""
        A = np.zeros((n, n, n, n))
        for i in range(n):
            A[i, i, i, i] = 1
        return A

    def lattice_tensor(self) -> np.array:
        Q = self.transfer_matrix()
        delta = self.kronecker_matrix(2)
        return ncon(
            [Q, Q, Q, Q, delta], ([-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4])
        )
