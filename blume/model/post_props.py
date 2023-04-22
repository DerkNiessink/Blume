import numpy as np
from ncon import ncon


class Props:
    """
    Class for post CTM algorithm calculations of thermodynamic properties,
    using the converged corner and edge tensor.
    """

    @staticmethod
    def Z(C, T, beta, a, b) -> float:
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
    def m(C, T, beta, a, b) -> float:
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
                / Props.Z(C, T, beta, a, b)
            )
        )

    @staticmethod
    def f(C, T, beta, a, b) -> float:
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
        return float(-(1 / beta) * np.log(Props.Z(C, T, beta, a, b) * corners / denom))

    @staticmethod
    def Es(C, T, beta, a, b) -> float:
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
