import numpy as np
from scipy.linalg import sqrtm
from ncon import ncon
from dataclasses import dataclass, field


@dataclass
class Tensors:
    """Class that holds tensors for conducting the corner transfer matrix
    renormalization group (CTMRG) method for evualuating the Ising model"""

    beta: float = 0.5
    d: int = 2
    Q: np.ndarray = field(init=False)

    def __post_init__(self):
        self.Q = sqrtm(
            np.array(
                [
                    [np.exp(self.beta), np.exp(-self.beta)],
                    [np.exp(-self.beta), np.exp(self.beta)],
                ]
            )
        )

    @staticmethod
    def delta(shape: tuple[int]) -> np.ndarray:
        """
        Returns a kronecker delta matrix of specific shape. The length of all
        dimensions of the shape has to be equal.
        """
        if shape.count(shape[0]) != len(shape):
            raise Exception("The length of all dimensions has to be equal.")

        A = np.zeros(shape)
        for index in np.ndindex(shape):
            # If all indices are the same, change value to 1.
            if index.count(index[0]) == len(index):
                A[index] = 1
        return A

    @staticmethod
    def random(shape: tuple) -> np.ndarray:
        """
        Returns a random tensor of specific shape, which can be either rank 2
        or rank 3. The tensor is symmetric under the exchange of the first two
        indices and the values are normalized.
        """
        c = np.random.uniform(size=shape)
        return Methods.symmetrize(c)

    def a(self) -> np.ndarray:
        """
        Returns the tensor representation of a single lattice site in the
        partition function.
        """
        delta = self.delta((self.d, self.d, self.d, self.d))
        return ncon(
            [self.Q, self.Q, self.Q, self.Q, delta],
            ([-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4]),
        )

    def b(self) -> np.ndarray:
        """
        Returns the tensor representation for a single lattice site in the
        numerator of the magnetization (partition function = denominator).
        """
        delta = self.delta((self.d, self.d, self.d, self.d))
        delta[1, :, :, :] *= -1.0
        return ncon(
            [self.Q, self.Q, self.Q, self.Q, delta],
            ([-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4]),
        )

    def C_init(self) -> np.ndarray:
        """
        Returns the initial corner tensor for a system with boundary conditions.
        """
        return ncon(
            [self.Q, self.delta((self.d, self.d)), self.Q], ([1, -1], [1, 2], [-2, 2])
        )

    def T_init(self) -> np.ndarray:
        """
        Returns the initial edge tensor for a system with boundary conditions.
        """
        return ncon(
            [self.Q, self.Q, self.Q, self.delta((self.d, self.d, self.d))],
            ([-1, 1], [-2, 2], [-3, 3], [1, 2, 3]),
        )


@dataclass
class Methods:
    """This class contains methods for np.arrays, required for the CTM algorithm."""

    @staticmethod
    def symmetrize(M: np.ndarray) -> np.ndarray:
        """
        Symmetrize the array about the first two axes. Only works for 2 or 3
        dimensional arrays.
        """
        if len(M.shape) != 2 and len(M.shape) != 3:
            raise Exception("M has to a 2 or 3 dimensional array.")

        axes = (1, 0) if len(M.shape) == 2 else (1, 0, 2)
        return (M + np.transpose(M, axes)) / 2

    @staticmethod
    def normalize(M: np.ndarray) -> np.ndarray:
        """
        Divide all elements in the given array by its largest value.
        """
        return M / np.amax(M)
