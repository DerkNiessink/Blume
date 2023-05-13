import numpy as np
from scipy.linalg import sqrtm
from ncon import ncon
from dataclasses import dataclass, field


@dataclass
class Tensors:
    """
    Class that holds tensors for conducting the corner transfer matrix
    renormalization group (CTMRG) method for evualuating the Ising model
    and Blume-Capel model.

    beta (float): Beta for the system, equivalent to 1 / temperature.
    model (str): "ising" or "blume".
    coupling (float): Crystal-field coupling parameter for the system.
    """

    beta: float = 0.5
    model: str = "ising"
    coupling: float = 1
    d: int = field(init=False)
    Q: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Compute the Q matrix for the given model.
        """

        if self.model == "ising":
            self.d = 2
            self.Q = sqrtm(
                np.array(
                    [
                        [np.exp(self.beta), np.exp(-self.beta)],
                        [np.exp(-self.beta), np.exp(self.beta)],
                    ]
                )
            )
        else:
            self.d = 3
            c = self.coupling
            self.Q = sqrtm(
                np.array(
                    [
                        [
                            np.exp((1 - c) * self.beta),
                            np.exp(-c * self.beta),
                            np.exp(-(1 + c) * self.beta),
                        ],
                        [1, 1, 1],
                        [
                            np.exp(-(1 + c) * self.beta),
                            np.exp(-c * self.beta),
                            np.exp((1 - c) * self.beta),
                        ],
                    ]
                )
            )

    @staticmethod
    def delta(shape: tuple[int, ...], adj=False) -> np.ndarray:
        """
        Returns a kronecker delta matrix of specific shape. The length of all
        dimensions of the shape has to be equal.

        `shape` (tuple): desired shape of the kronecker delta matrix.
        `adj` (bool): If true, an adjusted delta matrix is returned, which
        only yields 1 if all indices are 0.
        """
        if shape.count(shape[0]) != len(shape):
            raise Exception("The length of all dimensions has to be equal.")

        A = np.zeros(shape)
        for index in np.ndindex(shape):
            # If all indices are the same change value to 1, or if adj=True only
            # if all indices are 0.
            k = 0 if adj else index[0]
            if index.count(k) == len(index):
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

    def a(self, adj=False) -> np.ndarray:
        """
        Returns the tensor representation of a single lattice site in the
        partition function.

        `adj` (bool): If true the adjusted delta will be used for the lattice
        site, which fixes the corner spin to one direction.
        """
        delta = self.delta((self.d, self.d, self.d, self.d), adj)
        return np.array(
            ncon(
                [self.Q, self.Q, self.Q, self.Q, delta],
                ([-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4]),
            )
        )

    def b(self, adj=False) -> np.ndarray:
        """
        Returns the tensor representation for a single lattice site in the
        numerator of the magnetization (partition function = denominator).

        `adj` (bool): If true the adjusted delta will be used for the lattice
        site, which fixes the corner spin to one direction.
        """
        delta = self.delta((self.d, self.d, self.d, self.d), adj)
        if self.model == "ising":
            delta[1, :, :, :] *= -1.0
        else:
            delta[0, :, :, :] *= -1.0
            delta[1, :, :, :] *= 0

        return np.array(
            ncon(
                [self.Q, self.Q, self.Q, self.Q, delta],
                ([-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4]),
            )
        )

    def C_init(self, adj=False) -> np.ndarray:
        """
        Returns the initial corner tensor for a system with boundary conditions.

        `adj` (bool): If true the adjusted delta will be used for the lattice
        site, which fixes the corner spin to one direction.
        """
        return np.array(
            ncon(
                [self.Q, self.delta((self.d, self.d), adj), self.Q],
                ([1, -1], [1, 2], [-2, 2]),
            )
        )

    def T_init(self, adj=False) -> np.ndarray:
        """
        Returns the initial edge tensor for a system with boundary conditions.

        `adj` (bool): If true the adjusted delta will be used for the lattice
        site, which fixes the edge spin to one direction.
        """
        return np.array(
            ncon(
                [self.Q, self.Q, self.Q, self.delta((self.d, self.d, self.d), adj)],
                ([-1, 1], [-2, 2], [-3, 3], [1, 2, 3]),
            )
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
        return np.array(M + np.transpose(M, axes)) / 2

    @staticmethod
    def normalize(M: np.ndarray) -> np.ndarray:
        """
        Divide all elements in the given array by its largest value.
        """
        return np.array(M / np.amax(M))
