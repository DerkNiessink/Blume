import numpy as np
import scipy.sparse.linalg
import scipy.linalg
from ncon import ncon
import time
from typing import Union

from .tensors import Tensors, Methods

norm = Methods.normalize
symm = Methods.symmetrize


class CtmAlg:
    """
    Class for the Corner Transfer Matrix (CTM) algorithm for evualuating the
    partition function and other physical properties of the Ising model.

    `beta` (float): Equivalent to the inverse of the temperature.
    `model` (str): "ising" or "blume"
    `coupling` (float): crystal-field coupling parameter for the Blume-Capel model.
    `h` (float): External magnetic field parameter.
    `b_c` (bool): If true the edge and corner tensors are initialized with the
    boundary condition (bc) tensors, else random.
    `fixed` (bool): If true fix the spin on the the edge to one spin. Only
    applies for a system with boundary conditions. Compute both the fixed and
    unfixed edge.
    `chi` (int): bond dimension of the edge and corner tensors.
    `C_init` (np.array) and `T_init` (np.array): Optional initial corner and
    edge tensor respectively of shapes (chi, chi) and (chi, chi, d). If boundary
    conditions is true, the tensors are overwritten by the initial bc tensors.
    If only one of the two is given, the other will be random initialized.
    """

    def __init__(
        self,
        beta: float,
        model="ising",
        coupling=1,
        h=0,
        b_c=False,
        fixed=False,
        chi=2,
        C_init=None,
        T_init=None,
    ):
        self.tensors = Tensors(beta, model, coupling, h)
        self.d = self.tensors.d
        self.beta = beta
        self.b_c = b_c
        self.chi = chi

        # Keep track of both the fixed and unfixed tensors.
        self.C, self.T, self.T_fixed = self.init_tensors(
            C_init=C_init, T_init=T_init, fixed=fixed
        )
        self.a = self.tensors.a()
        self.b = self.tensors.b()
        self.b_fixed = self.tensors.b(adj=True)
        self.a_fixed = self.tensors.a(adj=True)
        self.sv_sums = [0]
        self.exe_time = None
        self.max_chi = chi

    def init_tensors(
        self, fixed=False, C_init=None, T_init=None
    ) -> tuple[np.ndarray, np.ndarray, Union[None, np.ndarray]]:
        """
        Returns a tuple of the corner and edge tensor. The tensors are random
        or initialized, depending on the boundary_conditions state, the given
        `C_init` and `T_init` and the fixed_corner and/or fixed_edge state.

        `fixed` (bool): If true, fix the edge spin to one direction.
        `C_init` (np.ndarray | None): Optional initial corner tensor.
        `T_init` (np.ndarray | None): Optional initial edge tensor.
        """
        if self.b_c:
            self.chi = self.d
            T_fixed = self.tensors.T_init(fixed) if fixed else None
            return self.tensors.C_init(), self.tensors.T_init(), T_fixed

        C = (  # Initialize random, if no `C_init` is given.
            self.tensors.random((self.chi, self.chi)) if C_init is None else C_init
        )
        T = (  # Initialize random, if no `T_init` is given.
            self.tensors.random((self.chi, self.chi, self.d))
            if T_init is None
            else T_init
        )
        return C, T, None

    def exe(self, tol=1e-3, count=10, max_steps=10000):
        """
        Execute the CTM algorithm for a number of steps `n_steps`. For each
        step, an `a` tensor is inserted, from which a new edge and corner tensor
        is evaluated. The new edge and corner tensors are normalized and
        symmetrized every step.

        `tol` (float): convergence criterion.
        `count` (int): Consecutive times the tolerance has to be satified before
        terminating the algorithm.
        `max_steps` (int): maximum number of steps before terminating the
        algorithm when convergence has not yet been reached.
        """
        start = time.time()
        tol_counter = 0
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

            # Keep track of edge tensor with fixed spins if given.
            if self.T_fixed is not None:
                self.T_fixed = symm(norm(self.new_T(U, fixed=True)))

            # Save sum of singular values
            self.sv_sums.append(np.sum(s))

            tol_counter += 1 if abs(self.sv_sums[-1] - self.sv_sums[-2]) < tol else 0

            if tol_counter == count:
                break

        # Save the computational time and number of iterations
        self.n_iter = len(self.sv_sums)
        self.exe_time = time.time() - start

    def new_C(self, U: np.ndarray) -> np.ndarray:
        """
        Insert an `a` tensor and evaluate a corner matrix `new_M` by contracting
        the new corner. Renormalized the new corner with the given `U` matrix.

        `U` (np.ndarray): The renormalization tensor of shape (chi, d, chi).

        Returns an array of the new corner tensor of shape (chi, chi)
        """
        return np.array(
            ncon(
                [U, self.new_M(), U],
                ([-1, 2, 1], [1, 2, 3, 4], [-2, 4, 3]),
            )
        )

    def new_M(self) -> np.ndarray:
        """
        evaluate the `M`, i.e. the new contracted corner with the inserted `a`
        tensor.

        Returns a array of the contracted corner of shape (chi, d, chi, d).
        """
        return np.array(
            ncon(
                [self.C, self.T, self.T, self.a],
                ([1, 2], [-1, 1, 3], [2, -3, 4], [-2, 3, 4, -4]),
            )
        )

    def new_T(self, U: np.ndarray, fixed=False) -> np.ndarray:
        """
        Insert an `a` tensor and evaluate a new edge tensor. Renormalize
        the edge tensor by contracting with the given truncated `U` tensor.

        `U` (ndarray): The renormalization tensor of shape (chi, d, chi).
        `fixed` (ndarray): If true, evaluate the new edge tensor with an initial
        fixed spin.

        Returns an array of the new edge tensor of shape (chi, chi, d).
        """
        T = self.T_fixed if fixed else self.T
        M = ncon([T, self.a], ([-1, -2, 1], [1, -3, -4, -5]))
        return np.array(
            ncon(
                [U, M, U],
                ([-1, 3, 1], [1, 2, 3, 4, -3], [-2, 4, 2]),
            )
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
            # Truncate down to the desired chi value, if not yet reached.
            self.chi = self.max_chi
            U, s, _ = scipy.sparse.linalg.svds(M, k=self.chi, which="LM")
        else:
            # Also increase chi when using the untruncated U.
            self.chi *= self.d
            U, s, _ = scipy.linalg.svd(M)

        # Reshape U back in a three legged tensor and transpose. Normalize the singular values.
        return np.reshape(U, (k, self.d, self.chi)).T, norm(s)
