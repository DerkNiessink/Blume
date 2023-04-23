try:
    from model.CTM_alg import CtmAlg
except:
    from blume.model.CTM_alg import CtmAlg

import numpy as np
import json
from tqdm import tqdm
import os
from datetime import datetime


def sweep_T(
    T_range: tuple | list,
    step=0.001,
    tol=1e-7,
    chi=2,
    max_steps=int(10e8),
    use_prev=False,
    b_c=False,
    bar=True,
) -> tuple:
    """
    Sweep temperature for the given range and stepsize and execute the CTM algorithm.
    Return data containing the algorithm paramaters and properties extracted during
    the algorithm.

    `T_range` (tuple | list): Temperature range or list to sweep.
    `step` (float): Stepsize of the varying temperature.
    `tol` (float): Convergence criterion.
    `chi` (int): Bond dimension.
    `max_steps` (int): Maximum number of steps before terminating the
    algorithm when convergence has not yet been reached.
    `use_prev` (bool): If true, the converged corner and edge from the previous
    iteration is used as initial tensors, else the initial tensors are random.
    `b_c` (bool): Set a fixed boundary conditions on the system if True.
    `bar` (bool): If true, a progress bar is displayed.

    Returns a list containing physical properties in the following order:
    [chi, tol, max_steps, b_c, computional time, temperatures, converged corner,
     converged edge, a tensor, b tensor].
    """
    data = []
    C_init, T_init = None, None
    desc = f"L = {max_steps}" if b_c else f"chi = {chi}"

    # Allow a list or range tuple for `T_range`.
    T_array = (
        T_range
        if isinstance(T_range, list)
        else np.arange(1 / T_range[1], 1 / T_range[0], step)[::-1]
    )

    for beta in tqdm(T_array, desc=desc, disable=not (bar)):
        alg = CtmAlg(beta, chi=chi, C_init=C_init, T_init=T_init, b_c=b_c)
        alg.exe(tol, max_steps)

        if use_prev:
            C_init, T_init = alg.C, alg.T

        # Save execution time, temperature and the converged corner and edge and
        # the a and b tensors.
        data.append((alg.exe_time, 1 / beta, alg.C, alg.T, alg.a, alg.b))

    return [chi, tol, max_steps, b_c] + list(zip(*data))


class NumpyEncoder(json.JSONEncoder):
    """
    Class for encoding a np.ndarray to JSON
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save(data: list, dir: str, msg=True):
    """
    Save the data containing physical properties for varying beta in the `data`
    folder.

    `data` (list): Should be of the following form:
    [chi, tolerance, max number of steps, boundary conditions, execution times,
    converged corner, converged edge, a tensor, b tensor].
    `dir` (str): Directory where the data is saved.
    `msg` (bool): If true, print a message at the start and end of saving.
    """
    if msg:
        print(f"Saving data in folder: '{dir}'")

    # Name the file L{max_steps} if b_c else chi{chi}
    fn = f"L{data[2]}" if data[3] else f"chi{data[0]}"

    with open(f"data/{dir}/{fn}.json", "w") as fp:
        json.dump(
            {
                "chi": data[0],
                "tolerance": data[1],
                "max number of steps": data[2],
                "boundary conditions": data[3],
                "execution times": data[4],
                "temperatures": data[5],
                "converged corners": data[6],
                "converged edges": data[7],
                "a tensors": data[8],
                "b tensors": data[9],
            },
            fp,
            cls=NumpyEncoder,
        )

    if msg:
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
    T_c = 2 / np.log(1 + np.sqrt(2))

    for chi in range(2, 20):
        data = sweep_T(
            chi=chi,
            T_range=(T_c, T_c + 0.01),
            step=0.01,
            tol=1e-9,
            max_steps=int(10e8),
            use_prev=False,
        )
        save(data, dir)

    """
    for L in [10000]:
        data = sweep_T(
            T_range=(1, 4),
            step=0.001,
            max_steps=L,
            b_c=True,
        )
        save(data, dir)
    """
