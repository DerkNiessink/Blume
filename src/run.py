from CTM_alg import CtmAlg

import numpy as np
import json
from tqdm import tqdm
import os
from datetime import datetime


def sweep_T(
    chi: int, T_range: tuple, step: float, tol: float, max_steps: int, use_prev=True
) -> tuple:
    """Sweep beta for the given range and stepsize and execute the CTM algorithm.
    Use for every execution the corner and edge tensor of the previous step for
    less computational cost. Return data containing physical properties extracted
    from the converged system.

    `chi` (int): Bond dimension.
    `T_range` (tuple): Temperature range to sweep.
    `step` (float): Stepsize of the varying temperature.
    `tol` (float): Convergence criterion.
    `max_steps` (int): Maximum number of steps before terminating the
    algorithm when convergence has not yet been reached.
    `use_prev` (bool): If true, the converged corner and edge from the previous
    iteration is used as initial tensors, else the initial tensors are random.

    Returns a list containing physical properties in the following order:
    [chi, tolerance, computional time, temperatures, converged corner, converged edge].
    """
    data = []
    C_init, T_init = None, None

    for beta in tqdm(
        np.arange(1 / T_range[1], 1 / T_range[0], step)[::-1], desc=f"Chi = {chi}"
    ):
        alg = CtmAlg(beta, chi=chi, C_init=C_init, T_init=T_init)
        alg.exe(tol, max_steps)

        if use_prev:
            C_init, T_init = alg.C, alg.T

        # Save execution time, temperature and the converged corner and edge and
        # the a and b tensors.
        data.append((alg.exe_time, 1 / beta, alg.C, alg.T, alg.a, alg.b))

    return [chi, tol] + list(zip(*data))


class NumpyEncoder(json.JSONEncoder):
    """
    Class for encoding a np.ndarray to JSON
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save(data: list, dir: str):
    """
    Save the data containing physical properties for varying beta in the `data`
    folder.

    `data` (list): Should be of the following form:
    [chi, tolerance, execution times, converged corner, converged edge, a tensor,
     b tensor].
    """
    print(f"Saving data in for chi = {data[0]} in folder: '{dir}'")
    with open(f"data/{dir}/chi{data[0]}.json", "w") as fp:
        json.dump(
            {
                "chi": data[0],
                "tolerance": data[1],
                "execution times": data[2],
                "temperatures": data[3],
                "converged corners": data[4],
                "converged edges": data[5],
                "a tensors": data[6],
                "b tensors": data[7],
            },
            fp,
            cls=NumpyEncoder,
        )
    print("Done \n")


def new_folder():
    """
    Make a new folder in the data directory with the date as name, if it does
    not exist yet.

    Returns the folder name as string.
    """
    now = datetime.now().strftime("%d-%m %H:%M")
    if not os.path.isdir(f"data/{now}"):
        os.mkdir(f"data/{now}")
    return now


if __name__ == "__main__":
    dir = new_folder()
    for chi in [4, 8, 12]:
        data = sweep_T(
            chi,
            T_range=(2.25, 2.29),
            step=0.0001,
            tol=1e-9,
            max_steps=int(10e8),
            use_prev=True,
        )
        save(data, dir)
