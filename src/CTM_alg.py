from src.tensors import Tensors
from src.tensors import Methods

norm = Methods.normalize
symm = Methods.symmetrize

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
            # Compute the new contraction `M` of the corner by inserting an `a` tensor.
            M = self.new_M()

            # Use `M` to compute the renormalization tensor
            U = self.new_U(M)

            # Normalize and symmetrize the new corner and edge tensors
            self.C = symm(norm(self.new_C(U)))
            self.T = symm(norm(self.new_T(U)))

            # Save singular values of C
            _, s, _ = np.linalg.svd(self.C)
            self.sv_sums.append(np.sum(s))

    def new_C(self, U: np.array) -> np.array:
        """
        Insert an `a` tensor and evaluate a corner matrix `M_matrix` by contracting
        the new corner and reshaping it in a matrix. Conduct an eigenvalue
        decomposition and only keep the `chi` largest eigenvalues and
        corresponding eigenvectors. Renormalized the new corner with the new
        `U` matrix in which the columns are the eigenvectors.
        """
        return ncon(
            [U, self.new_M(), U.T],
            ([1, -1, 2], [1, 3, 2, 4], [4, 3, -2]),
        )

    def new_M(self) -> np.array:
        """
        evaluate the `M_matrix`, i.e. the new corner with the inserted `a`
        tensor, and reshape to a matrix of shape (chi*d x chi*d).
        """
        return ncon(
            [self.C, self.T, self.T, self.a],
            ([1, 2], [-1, 1, 3], [2, -2, 4], [-3, -4, 3, 4]),
        )

    def new_T(self, U: np.array) -> np.array:
        """
        Insert an `a` tensor and evaluate a new edge tensor. Renormalize
        the edge tensor by contracting with the truncated `U` tensor, obtained
        from the eigenvalue decomposition of the new corner tensor.
        """
        M = ncon([self.T, self.a], ([-1, -2, 1], [-3, -4, -5, 1]))
        T = ncon(
            [U, U.T, M],
            ([-3, 3, 4], [2, 1, -1], [3, 1, 4, -2, 2]),
        )
        return np.transpose(T, (0, 2, 1))

    def new_U(self, M: np.array) -> np.array:
        """
        Return the truncated U matrix, by conducting an eigenvalue
        decomposition on the given corner matrix `M`. U is the matrix
        with the eigenvectors as collumns. This matrix is truncated
        by removing a number of lowest eigenvalues, which is equal to
        the number of bonds (chi).
        """
        M = np.reshape(M, (self.chi * self.d, self.chi * self.d))
        (w, U) = scipy.linalg.eigh(M, eigvals_only=False)
        # Only keep the eigenvectors corresponding to the chi largest
        # eigenvalues
        U = U[:, -self.chi :]
        # Reshape U back in a three legged tensor
        return np.reshape(U, (self.chi, self.chi, self.d))

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
