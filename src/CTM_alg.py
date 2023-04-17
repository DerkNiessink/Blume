import numpy as np
import scipy.sparse.linalg
import scipy.linalg
from ncon import ncon

try:
    from src.tensors import Tensors, Methods
except:
    from tensors import Tensors, Methods

norm = Methods.normalize
symm = Methods.symmetrize


class CtmAlg:
    """
    Class for the Corner Transfer Matrix (CTM) algorithm for evualuating the
    partition function and other physical properties of the Ising model.
    """

    def __init__(self, beta: float, boundary_conditions=False, chi=2, n_states=2):
        self.tensors = Tensors(beta)
        self.beta = beta
        self.b_c = boundary_conditions
        self.chi = chi
        self.d = n_states
        self.C, self.T = self.init_tensors()
        self.a = self.tensors.a()
        self.b = self.tensors.b()
        self.sv_sums = [0]
        self.magnetizations = []
        self.partition_functions = []

    def init_tensors(self) -> tuple[np.array, np.array]:
        """
        Returns a tuple of the corner and edge tensor. The tensors are random
        or initialized, depending on the boundary_conditions state
        """
        if self.b_c:
            return self.tensors.C_init(), self.tensors.T_init()
        else:
            return self.tensors.random((self.chi, self.chi)), self.tensors.random(
                (self.chi, self.chi, self.d)
            )

    def exe(self, tol=1e-3, max_steps=10000):
        """
        Execute the CTM algorithm for a number of steps `n_steps`. For each
        step, an `a` tensor is inserted, from which a new edge and corner tensor
        is evaluated. The new edge and corner tensors are normalized and
        symmetrized every step.

        `tol` (float): convergence criterion.
        `max_steps` (int): maximum number of steps before terminating the
        algorithm when convergered has not yet been reached.
        """
        for _ in range(max_steps):
            # Compute the new contraction `M` of the corner by inserting an `a` tensor.
            M = self.new_M()

            # Use `M` to compute the renormalization tensor
            U, s = self.new_U(M)

            # Normalize and symmetrize the new corner and edge tensors
            self.C = symm(norm(self.new_C(U)))
            self.T = symm(norm(self.new_T(U)))

            # Save singular values of C and magnetization
            self.sv_sums.append(np.sum(s))
            self.magnetizations.append(self.m())
            self.partition_functions.append(self.Z())

            if abs(self.sv_sums[-1] - self.sv_sums[-2]) < tol:
                break

    def new_C(self, U: np.array) -> np.array:
        """
        Insert an `a` tensor and evaluate a corner matrix `new_M` by contracting
        the new corner. Renormalized the new corner with the given `U` matrix.

        `U` (np.array): The renormalization tensor of shape (chi, d, chi).

        Returns an array of the new corner tensor of shape (chi, chi)
        """
        return ncon(
            [U, self.new_M(), U],
            ([-1, 2, 1], [1, 2, 3, 4], [-2, 4, 3]),
        )

    def new_M(self) -> np.array:
        """
        evaluate the `M`, i.e. the new contracted corner with the inserted `a`
        tensor.

        Returns am array of the contracted corner of shape (chi, d, chi, d).
        """
        return ncon(
            [self.C, self.T, self.T, self.a],
            ([1, 2], [-1, 1, 3], [2, -3, 4], [-2, 3, 4, -4]),
        )

    def new_T(self, U: np.array) -> np.array:
        """
        Insert an `a` tensor and evaluate a new edge tensor. Renormalize
        the edge tensor by contracting with the given truncated `U` tensor.

        `U` (np.array): The renormalization tensor of shape (chi, d, chi).
        """
        M = ncon([self.T, self.a], ([-1, -2, 1], [1, -3, -4, -5]))
        return ncon(
            [U, M, U],
            ([-1, 3, 1], [1, 2, 3, 4, -3], [-2, 4, 2]),
        )

    def new_U(self, M: np.array, trunc=True) -> tuple[np.array, np.array]:
        """
        Return a tuple of the truncated `U` tensor and `s` matrix, by conducting a
        singular value decomposition (svd) on the given corner tensor `M`. Using
        this factorization `M` can be written as M = U s V*, where the `U` matrix
        is used for renormalization and the `s` matrix contains the singular
        values in descending order.

        `M` (np.array): The new contracted corner tensor of shape (chi, d, chi, d).
        `trunc` (bool): If trunc is True, the `U` and `s` matrices are truncated,
        keeping only the singular values with the chi largest magnitude.

        Returns `s` and the renormalization tensor of shape (chi, d, chi) if
        `trunc` = True, else with shape (chi, d, 2*chi), which is obtained by reshaping
        `U` in a rank-3 tensor and transposing.
        """

        # Reshape M in a matrix
        M = np.reshape(M, (self.chi * self.d, self.chi * self.d))

        if trunc:
            # Get the chi largest singular values and corresponding singular vector matrix.
            k = self.chi
            U, s, _ = scipy.sparse.linalg.svds(M, k=k, which="LM")
        else:
            k = 2 * self.chi
            U, s, _ = scipy.linalg.svd(M)

        # Reshape U back in a three legged tensor and transpose
        return np.reshape(U, (k, self.d, self.chi)).T, s

    def Z(self) -> float:
        """
        Return the value for the partition function of the system.
        """
        return float(
            ncon(
                [
                    self.C,
                    self.T,
                    self.T,
                    self.C,
                    self.a,
                    self.C,
                    self.T,
                    self.T,
                    self.C,
                ],
                (
                    [1, 2],
                    [1, 4, 3],
                    [2, 8, 7],
                    [4, 5],
                    [3, 6, 7, 9],
                    [8, 12],
                    [5, 10, 6],
                    [11, 12, 9],
                    [10, 11],
                ),
            )
        )

    def m(self) -> float:
        """
        Return the value for the magnetization of the system.
        """
        return float(
            ncon(
                [
                    self.C,
                    self.T,
                    self.T,
                    self.C,
                    self.b,
                    self.C,
                    self.T,
                    self.T,
                    self.C,
                ],
                (
                    [1, 2],
                    [1, 4, 3],
                    [2, 8, 7],
                    [4, 5],
                    [3, 6, 7, 9],
                    [8, 12],
                    [5, 10, 6],
                    [11, 12, 9],
                    [10, 11],
                ),
            )
            / self.Z()
        )

    def f(self) -> float:
        """
        Return the free energy of the system.
        """
        corners = ncon(
            [self.C, self.C, self.C, self.C], ([1, 2], [1, 3], [3, 4], [4, 2])
        )
        denom = (
            ncon(
                [self.C, self.C, self.T, self.T, self.C, self.C],
                ([1, 2], [1, 3], [2, 5, 4], [3, 6, 4], [5, 7], [6, 7]),
            )
        ) ** 2
        return float(-(1 / self.beta) * np.log(self.Z() * corners / denom))
