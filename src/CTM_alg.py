from tensors import Tensors

import numpy as np
import scipy.linalg
from ncon import ncon
from tqdm import tqdm


class CtmAlg:
    """
    Class for conducting the Corner Transfer Matrix (CTM) algorithm for
    evualuating the partition function and other physical properties of the
    Ising model.
    """

    def __init__(self, beta: float, chi: int, n_states=2):
        self.tensors = Tensors(beta=beta)
        self.chi = chi
        self.d = n_states
        self.C = self.tensors.random_tensor((chi, chi))
        self.T = self.tensors.random_tensor((chi, chi, self.d))
        self.a = self.tensors.a_tensor()
        self.b = self.tensors.b_tensor()
        self.sv_sums = []

    def exe(self, n_steps: int):
        """
        Execute the CTM algorithm for a number of steps `n_steps`. For each
        step, an `a` tensor is inserted, from which a new edge and corner tensor
        is evaluated. The new edge and corner tensors are normalized and symmetrized
        every step.
        """
        for _ in tqdm(range(n_steps)):
            self.C = self._eval_corner()
            self.C = self._symmetrize(self._normalize(self.C))
            self.T = self._eval_edge()
            self.T = self._symmetrize(self._normalize(self.T))
            _, s, _ = np.linalg.svd(self.C)
            self.sv_sums.append(np.sum(s))

    def _eval_corner(self) -> np.array:
        """
        Insert an `a` tensor and evaluate a corner matrix `M_matrix` by contracting
        the new corner and reshaping it in a matrix. Conduct an eigenvalue
        decomposition and only keep the `chi` largest eigenvalues and
        corresponding eigenvectors. Renormalized the new corner with the new
        `U` matrix in which the columns are the eigenvectors.
        """
        M_matrix = self._eval_M()
        self.trunc_U = self._truncated_U(M_matrix)
        M = ncon(
            [self.C, self.T, self.T, self.a],
            ([1, 2], [-1, 1, 3], [2, -2, 4], [-3, -4, 3, 4]),
        )
        return ncon(
            [self.trunc_U, M, self.trunc_U.T],
            ([1, -1, 2], [1, 3, 2, 4], [4, 3, -2]),
        )

    def _eval_M(self) -> np.array:
        """
        evaluate the `M_matrix`, i.e. the new corner with the inserted `a`
        tensor, and reshape to a matrix of shape (chi*d x chi*d).
        """
        M = ncon(
            [self.C, self.T, self.T, self.a],
            ([1, 2], [-1, 1, 3], [2, -2, 4], [-3, -4, 3, 4]),
        )
        return np.reshape(M, (self.chi * self.d, self.chi * self.d))

    def _eval_edge(self) -> np.array:
        """
        Insert an `a` tensor and evaluate a new edge tensor. Renormalize
        the edge tensor by contracting with the truncated `U` tensor, obtained
        from the eigenvalue decomposition of the new corner tensor.
        """
        M = ncon([self.T, self.a], ([-1, -2, 1], [-3, -4, -5, 1]))
        T = ncon(
            [self.trunc_U, self.trunc_U.T, M],
            ([-3, 3, 4], [2, 1, -1], [3, 1, 4, -2, 2]),
        )
        return np.transpose(T, (0, 2, 1))

    def _truncated_U(self, M: np.array) -> np.array:
        """
        Return the truncated U matrix, by conducting an eigenvalue
        decomposition on the given corner matrix `M`. U is the matrix
        with the eigenvectors as collumns. This matrix is truncated
        by removing a number of lowest eigenvalues, which is equal to
        the number of bonds (chi).
        """
        (w, U) = scipy.linalg.eigh(M, eigvals_only=False)
        # Only keep the eigenvectors corresponding to the chi largest
        # eigenvalues
        U = U[:, -self.chi :]
        # Reshape U back in a three legged tensor        # Contract the inserted a matrix and the edge matrix T.
        return np.reshape(U, (self.chi, self.chi, self.d))

    def _symmetrize(self, M) -> np.array:
        """
        Symmetrize the given tensor M about the first two axes.
        """
        axes = (1, 0) if len(np.shape(M)) == 2 else (1, 0, 2)
        return np.maximum(M, np.transpose(M, axes))

    def _normalize(self, M) -> np.array:
        """
        Divide the given tensor M by its largest value.
        """
        return M / np.amax(M)

    def Z(self) -> float:
        """
        Return the value for the partition function of the system.
        """
        return ncon(
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

    def m(self) -> float:
        """
        Return the value for the magnetization of the system.
        """
        return (
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
