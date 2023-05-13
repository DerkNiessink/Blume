import numpy as np
from ncon import ncon
import scipy.linalg
import scipy.sparse.linalg
from typing import Callable


PropFunction = Callable[[dict], float]


def unpack(data: dict):
    return (
        data["C"],
        data["T"],
        data["T_fixed"],
        data["beta"],
        data["a"],
        data["a_fixed"],
        data["b"],
        data["b_fixed"],
    )


class Prop:
    """
    Class for post CTM algorithm calculations of thermodynamic properties,
    using the converged corner and edge tensor.
    """

    @staticmethod
    def Z(data: dict) -> float:
        """
        Return the value for the partition function of the system.
        """
        C, T, T_fixed, beta, a, a_fixed, b, b_fixed = unpack(data)
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
    def m(data: dict) -> float:
        """
        Return the value for the magnetization of the system.
        """
        C, T, T_fixed, beta, a, a_fixed, b, b_fixed = unpack(data)
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
                / Prop.Z(data)
            )
        )

    @staticmethod
    def f(data: dict) -> float:
        """
        Return the free energy of the system.
        """
        C, T, T_fixed, beta, a, a_fixed, b, b_fixed = unpack(data)
        corners = ncon([C, C, C, C], ([1, 2], [1, 3], [3, 4], [4, 2]))
        denom = (
            ncon(
                [C, C, T, T, C, C],
                ([1, 2], [1, 3], [2, 5, 4], [3, 6, 4], [5, 7], [6, 7]),
            )
        ) ** 2
        return float(-(1 / beta) * np.log(Prop.Z(data) * corners / denom))

    @staticmethod
    def Es(data: dict) -> float:
        """
        Return the energy per site of the system.
        """
        C, T, T_fixed, beta, a, a_fixed, b, b_fixed = unpack(data)
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
    def xi(data: dict) -> float:
        """
        Return the correlation length of the system.
        """
        C, T, T_fixed, beta, a, a_fixed, b, b_fixed = unpack(data)
        M = ncon([T, T], ([-1, -3, 3], [-2, -4, 3]))
        # Reshape to matrix
        M = M.reshape(T.shape[0] ** 2, T.shape[0] ** 2)
        w = scipy.linalg.eigh(M, eigvals_only=True)
        return float(1 / np.log(abs(w[-1]) / abs(w[-2])))

    @staticmethod
    def Z_fixed(data: dict) -> float:
        """
        Return the value for the partition function of a fixed system.
        """
        C, T, T_fixed, beta, a, a_fixed, b, b_fixed = unpack(data)
        return float(
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
        )

    @staticmethod
    def m_fixed(data: dict):
        """
        Return the magnetization for a system with a fixed edge spin.
        """
        C, T, T_fixed, beta, a, a_fixed, b, b_fixed = unpack(data)
        return np.sqrt(abs(float(Prop.Z_fixed(data) / Prop.Z(data))))
