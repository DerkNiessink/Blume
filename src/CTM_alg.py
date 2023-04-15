import numpy as np
import scipy.sparse.linalg
import scipy.linalg
from ncon import ncon
from tqdm import tqdm

try:
    from src.tensors import Tensors, Methods
except:
    from tensors import Tensors, Methods

norm = Methods.normalize
symm = Methods.symmetrize


class CtmAlg:
    """
    Class for conducting the Corner Transfer Matrix (CTM) algorithm for
    evualuating the partition function and other physical properties of the
    Ising model.
    """

    def __init__(self, beta: float, boundary_conditions=False, chi=2, n_states=2):
        self.tensors = Tensors(beta)
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
        is evaluated. The new edge and corner tensors are normalized and symmetrized
        every step.
        """
        msg = "\nMax steps reached \u2717\n"
        for _ in tqdm(range(max_steps), desc="Percentage of max steps"):
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
                msg = "\nConvergence reached \u2713\n"
                break

        print(msg)

    def new_C(self, U: np.array) -> np.array:
        """
        Insert an `a` tensor and evaluate a corner matrix `M_matrix` by contracting
        the new corner and reshaping it in a matrix. Conduct an eigenvalue
        decomposition and only keep the `chi` largest eigenvalues and
        corresponding eigenvectors. Renormalized the new corner with the new
        `U` matrix in which the columns are the eigenvectors.
        """
        return ncon(
            [U, self.new_M(), U],
            ([-1, 2, 1], [1, 2, 3, 4], [-2, 4, 3]),
        )

    def new_M(self) -> np.array:
        """
        evaluate the `M_matrix`, i.e. the new corner with the inserted `a`
        tensor.
        """
        return ncon(
            [self.C, self.T, self.T, self.a],
            ([1, 2], [-1, 1, 3], [2, -3, 4], [-2, 3, 4, -4]),
        )

    def new_T(self, U: np.array) -> np.array:
        """
        Insert an `a` tensor and evaluate a new edge tensor. Renormalize
        the edge tensor by contracting with the truncated `U` tensor, obtained
        from the eigenvalue decomposition of the new corner tensor.
        """
        M = ncon([self.T, self.a], ([-1, -2, 1], [1, -3, -4, -5]))
        return ncon(
            [U, M, U],
            ([-1, 3, 1], [1, 2, 3, 4, -3], [-2, 4, 2]),
        )

    def new_U(self, M: np.array, trunc=True) -> tuple[np.array, np.array]:
        """
        Return a tuple of the truncated U and s matrix, by conducting an
        eigenvalue decomposition on the given corner matrix `M`. U is the
        matrix with the eigenvectors as columns and s the diagonal matrix
        with singular values. These matrices are truncated by removing a
        number of lowest eigenvalues, which is equal to the number of bonds (chi).
        """
        M = np.reshape(M, (self.chi * self.d, self.chi * self.d))
        # Get the chi largest singular values and corresponding singular vector matrix.
        if trunc:
            k = self.chi
            U, s, _ = scipy.sparse.linalg.svds(M, k=k, which="LM")
        else:
            k = 2 * self.chi
            U, s, _ = scipy.linalg.svd(M)
        # Truncate reshape U back in a three legged tensor
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
