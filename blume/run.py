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
    which="chi",
    step=0.001,
    tol=1e-7,
    chi=2,
    max_steps=int(10e8),
    use_prev=False,
    b_c=False,
    fixed=False,
    bar=True,
) -> tuple:
    """
    Sweep temperature for the given range and stepsize and execute the CTM algorithm.
    Return data containing the algorithm paramaters and properties extracted during
    the algorithm.

    `T_range` (tuple | list): Temperature range or list to sweep.
    `which` (str): Which parameter to vary. Has to be one of the following:
    {"chi", "max_steps", "tol", "step"}
    `step` (float): Stepsize of the varying temperature.
    `tol` (float): Convergence criterion.
    `chi` (int): Bond dimension.
    `max_steps` (int): Maximum number of steps before terminating the algorithm when
    convergence has not yet been reached. For a system with boundary conditions
    this corresponds to system size - 4.
    `use_prev` (bool): If true, the converged corner and edge from the previous
    iteration is used as initial tensors, else the initial tensors are random.
    `b_c` (bool): Set a fixed boundary conditions on the system if True.
    `bar` (bool): If true, a progress bar is displayed.

    Returns a list containing physical properties in the following order:
    [which, chi, tol, max_steps, step, b_c, computional time, temperatures,
     converged corner, converged edge, a tensor, b tensor].
    """
    param_dict = {"chi": chi, "max_steps": max_steps, "tol": tol, "step": step}
    data = []
    C_init, T_init = None, None

    # Allow a list or range tuple for `T_range`.
    temps = (
        [1 / T for T in T_range]
        if isinstance(T_range, list)
        else np.arange(1 / T_range[1], 1 / T_range[0], step)[::-1]
    )

    for beta in tqdm(temps, desc=f"{which}={param_dict[which]}", disable=not (bar)):
        alg = CtmAlg(beta, chi=chi, C_init=C_init, T_init=T_init, b_c=b_c, fixed=fixed)
        alg.exe(tol, max_steps)

        if use_prev:
            C_init, T_init = alg.C, alg.T

        # Save execution time, temperature and the converged corner and edge and
        # the a and b tensors.
        data.append((alg.exe_time, 1 / beta, alg.C, alg.T, alg.a, alg.b))

    return [which, chi, tol, max_steps, step, b_c] + list(zip(*data))


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
    [which, chi, tol, max_steps, step, boundary conditions, execution times,
     converged corner, converged edge, a tensor, b tensor].
    `dir` (str): Directory where the data is saved.
    `msg` (bool): If true, print a message at the start and end of saving.
    """
    if msg:
        print(f"Saving data in folder: '{dir}'")

    data_dict = {
        "chi": data[1],
        "tol": data[2],
        "max_steps": data[3],
        "step": data[4],
        "boundary conditions": data[5],
        "execution times": data[6],
        "temperatures": data[7],
        "converged corners": data[8],
        "converged edges": data[9],
        "a tensors": data[10],
        "b tensors": data[11],
    }

    # Name the file `which` with the value.
    with open(f"data/{dir}/{data[0]}{data_dict[data[0]]}.json", "w") as fp:
        json.dump(data_dict, fp, cls=NumpyEncoder)

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
