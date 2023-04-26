import numpy as np
import scipy.sparse.linalg
import scipy.linalg
from ncon import ncon
import time

try:
    from model.tensors import Tensors, Methods
except:
    from blume.model.tensors import Tensors, Methods

norm = Methods.normalize
symm = Methods.symmetrize


class CtmAlg:
    """
    Class for the Corner Transfer Matrix (CTM) algorithm for evualuating the
    partition function and other physical properties of the Ising model.

    `beta` (float): Equivalent to the inverse of the temperature.
    `b_c` (bool): If true the edge and corner tensors are initialized with the
    boundary condition (bc) tensors, else random.
    `fixed` (bool): If true fix the spin on the corner to one spin. Only applies
    for a system with boundary conditions.
    `chi` (int): bond dimension of the edge and corner tensors.
    `C_init` (np.array) and `T_init` (np.array): Optional initial corner and
    edge tensor respectively of shapes (chi, chi) and (chi, chi, d). If boundary
    conditions is true, the tensors are overwritten by the initial bc tensors.
    If only one of the two is given, the other will be random initialized.
    `n_states` (int): number of possible spin (energy) states (= 2 for Ising
    model).
    """

    def __init__(
        self,
        beta: float,
        b_c=False,
        fixed=False,
        chi=2,
        C_init=None,
        T_init=None,
        n_states=2,
    ):
        self.tensors = Tensors(beta)
        self.beta = beta
        self.b_c = b_c
        self.chi = chi
        self.d = n_states
        self.C_init = C_init
        self.T_init = T_init
        self.C, self.T = self.init_tensors(fixed)
        self.a = self.tensors.a()
        self.b = self.tensors.b()
        self.sv_sums = [0]
        self.magnetizations = []
        self.partition_functions = []
        self.exe_time = None
        self.max_chi = chi

    def init_tensors(self, fixed: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple of the corner and edge tensor. The tensors are random
        or initialized, depending on the boundary_conditions state and the given
        `C_init` and `T_init`.
        """
        if self.b_c:
            self.chi = 2
            return self.tensors.C_init(fixed), self.tensors.T_init()

        C = (  # Initialize random, if no `C_init` is given.
            self.tensors.random((self.chi, self.chi))
            if self.C_init is None
            else self.C_init
        )
        T = (  # Initialize random, if no `T_init` is given.
            self.tensors.random((self.chi, self.chi, self.d))
            if self.T_init is None
            else self.T_init
        )
        return C, T

    def exe(self, tol=1e-3, max_steps=10000):
        """
        Execute the CTM algorithm for a number of steps `n_steps`. For each
        step, an `a` tensor is inserted, from which a new edge and corner tensor
        is evaluated. The new edge and corner tensors are normalized and
        symmetrized every step.

        `tol` (float): convergence criterion.
        `max_steps` (int): maximum number of steps before terminating the
        algorithm when convergence has not yet been reached.
        """
        self.L = max_steps + 4  # system size
        start = time.time()
        for _ in range(max_steps):
            # Compute the new contraction `M` of the corner by inserting an `a` tensor.
            M = self.new_M()

            # Keep increasing chi if there are boundary condition until the
            # given max_chi.
            trunc = False if self.b_c and self.chi < self.max_chi else True
            # Use `M` to compute the renormalization tensor
            U, s = self.new_U(M, trunc)

            # Normalize and symmetrize the new corner and edge tensors
            self.C = symm(norm(self.new_C(U)))
            self.T = symm(norm(self.new_T(U)))

            # Save sum of singular values
            self.sv_sums.append(np.sum(s))

            if abs(self.sv_sums[-1] - self.sv_sums[-2]) < tol:
                # Save the computational time and number of iterations
                self.n_iter = len(self.sv_sums)
                self.exe_time = time.time() - start
                break

    def new_C(self, U: np.ndarray) -> np.ndarray:
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

    def new_M(self) -> np.ndarray:
        """
        evaluate the `M`, i.e. the new contracted corner with the inserted `a`
        tensor.

        Returns a array of the contracted corner of shape (chi, d, chi, d).
        """
        return ncon(
            [self.C, self.T, self.T, self.a],
            ([1, 2], [-1, 1, 3], [2, -3, 4], [-2, 3, 4, -4]),
        )

    def new_T(self, U: np.ndarray) -> np.ndarray:
        """
        Insert an `a` tensor and evaluate a new edge tensor. Renormalize
        the edge tensor by contracting with the given truncated `U` tensor.

        `U` (np.array): The renormalization tensor of shape (chi, d, chi).

        Returns an array of the new edge tensor of shape (chi, chi, d).
        """
        M = ncon([self.T, self.a], ([-1, -2, 1], [1, -3, -4, -5]))
        return ncon(
            [U, M, U],
            ([-1, 3, 1], [1, 2, 3, 4, -3], [-2, 4, 2]),
        )

    def new_U(self, M: np.ndarray, trunc=True) -> tuple[np.ndarray, np.ndarray]:
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
        k = self.chi
        if trunc:
            # Get the chi largest singular values and corresponding singular vector matrix.
            U, s, _ = scipy.sparse.linalg.svds(M, k=self.chi, which="LM")
        else:
            # Also increase chi when using the untruncated U.
            self.chi *= 2
            U, s, _ = scipy.linalg.svd(M)

        # Reshape U back in a three legged tensor and transpose. Normalize the singular values.
        return np.reshape(U, (k, self.d, self.chi)).T, norm(s)
