import numpy as np
from ncon import ncon
import scipy.linalg
import scipy.sparse.linalg


class Prop:
    """
    Class for post CTM algorithm calculations of thermodynamic properties,
    using the converged corner and edge tensor.
    """

    @staticmethod
    def Z(
        C: np.ndarray,
        T: np.ndarray,
        T_fixed: np.ndarray,
        beta: float,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Return the value for the partition function of the system.
        """
        return float(
            ncon(
                [C, T, T, C, a, C, T, T, C],
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

    @staticmethod
    def m(
        C: np.ndarray,
        T: np.ndarray,
        T_fixed: np.ndarray,
        beta: float,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Return the value for the magnetization of the system.
        """
        return abs(
            float(
                ncon(
                    [C, T, T, C, b, C, T, T, C],
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
                / Prop.Z(C, T, T_fixed, beta, a, b)
            )
        )

    @staticmethod
    def f(
        C: np.ndarray,
        T: np.ndarray,
        T_fixed: np.ndarray,
        beta: float,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Return the free energy of the system.
        """
        corners = ncon([C, C, C, C], ([1, 2], [1, 3], [3, 4], [4, 2]))
        denom = (
            ncon(
                [C, C, T, T, C, C],
                ([1, 2], [1, 3], [2, 5, 4], [3, 6, 4], [5, 7], [6, 7]),
            )
        ) ** 2
        return float(
            -(1 / beta) * np.log(Prop.Z(C, T, T_fixed, beta, a, b) * corners / denom)
        )

    @staticmethod
    def Es(
        C: np.ndarray,
        T: np.ndarray,
        T_fixed: np.ndarray,
        beta: float,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Return the energy per site of the system.
        """
        a_corner = ncon(
            [C, T, T, a],
            ([1, 2], [-1, 1, 3], [2, -3, 4], [-2, 3, 4, -4]),
        )
        denom = ncon(
            [a_corner, a_corner, C, T, T, C],
            ([1, 2, 3, 4], [1, 2, 5, 6], [3, 7], [7, 8, 4], [8, 9, 6], [9, 5]),
        )
        b_corner = ncon(
            [C, T, T, b],
            ([1, 2], [-1, 1, 3], [2, -3, 4], [-2, 3, 4, -4]),
        )
        num = ncon(
            [b_corner, b_corner, C, T, T, C],
            ([1, 2, 3, 4], [1, 2, 5, 6], [3, 7], [7, 8, 4], [8, 9, 6], [9, 5]),
        )

        return -float(num / denom) * 2

    @staticmethod
    def xi(
        C: np.ndarray,
        T: np.ndarray,
        T_fixed: np.ndarray,
        beta: float,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Return the correlation length of the system.
        """
        M = ncon([T, T], ([-1, -3, 3], [-2, -4, 3]))
        # Reshape to matrix
        M = M.reshape(T.shape[0] ** 2, T.shape[0] ** 2)
        w = scipy.linalg.eigh(M, eigvals_only=True)
        return 1 / np.log(abs(w[-1]) / abs(w[-2]))

    @staticmethod
    def Z_fixed(
        C: np.ndarray,
        T: np.ndarray,
        T_fixed: np.ndarray,
        beta: float,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Return the value for the partition function of a fixed system.
        """
        return float(
            ncon(
                [C, T_fixed, T, C, a, C, T, T, C],
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

    @staticmethod
    def xi_fixed(
        C: np.ndarray,
        T: np.ndarray,
        T_fixed: np.ndarray,
        beta: float,
        a: np.ndarray,
        b: np.ndarray,
    ):
        """
        Return the correlation length for a system with a fixed edge spin.
        """
        return Prop.Z_fixed(C, T, T_fixed, beta, a, b) / Prop.Z(
            C, T, T_fixed, beta, a, b
        )

    @staticmethod
    def m_fixed(
        C: np.ndarray,
        T: np.ndarray,
        T_fixed: np.ndarray,
        beta: float,
        a: np.ndarray,
        b: np.ndarray,
    ):
        """
        Return the magnetization for a system with a fixed edge spin.
        """
        return abs(
            float(
                ncon(
                    [C, T_fixed, T, C, b, C, T, T, C],
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
                / Prop.Z_fixed(C, T, T_fixed, beta, a, b)
            )
        )
