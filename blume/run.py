from .model.CTM_alg import CtmAlg

import numpy as np
import json
from tqdm import tqdm
import os
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ModelParameters:
    """
    `model` (str): "ising" or "blume".
    `T_range` (tuple | list): Temperature range or list to sweep.
    `coupling` (float): Crystal-field coupling parameter, only applies to
    Blume-Capel model.
    `step` (float): Stepsize of the varying temperature.
    `tol` (float): Convergence criterion.
    `chi` (int): Bond dimension.
    `max_steps` (int): Maximum number of steps before terminating the algorithm when
    convergence has not yet been reached. For a system with boundary conditions
    this corresponds to system size - 4.
    `use_prev` (bool): If true, the converged corner and edge from the previous
    iteration is used as initial tensors, else the initial tensors are random.
    `b_c` (bool): Set a fixed boundary conditions on the system if True.
    `fixed`(bool): Compute an additional edge tensor with an initial fixed spin.
    `bar` (bool): If true, a progress bar and save messages are displayed.
    """

    model: str = "ising"
    T_range: list | tuple = (2, 2.3)
    coupling: float = 1
    step: float = 0.001
    tol: float = 1e-7
    count: int = 10
    chi: int = 2
    max_steps: int = int(10e8)
    use_prev: bool = False
    b_c: bool = False
    fixed: bool = False
    bar: bool = True


class Results:
    """
    Class for executing the CTM algorithm for a varying parameter and saving
    the data.
    """

    def __init__(self, varying_param=None, range=None):
        self.varying_param = varying_param
        self.range = range
        self.default_params = ModelParameters()

    def get(self, params: ModelParameters | None):
        """
        Execute the algorithm, varying the given parameters and for the given
        model parameters. Save the data in a directory with the datetime as name.

        `params` (ModelParameters): class instance of the dataclass containing the
        model parameters.
        """
        self.dir = new_folder()

        if params is None:
            params = self.default_params

        if self.varying_param:
            for val in self.range:
                setattr(params, self.varying_param, val)
                data = self.sweep_T(params)
                self.save(data, params.bar)
        else:
            data = self.sweep_T(params)
            self.save(data, params.bar)

    def sweep_T(self, params: ModelParameters):
        """
        Sweep temperature for the given range and stepsize and execute the CTM algorithm.
        Return data containing the algorithm parameters and properties extracted during
        the algorithm.

        `params` (ModelParameters): class instance of the dataclass containing the
        model parameters.

        Returns a dict containing the ModelParameters and algorithm data.
        """
        data = []
        C_init, T_init = (None, None)

        # Allow a list or range tuple for `T_range`.
        temps = (
            params.T_range
            if isinstance(params.T_range, list)
            else np.arange(params.T_range[0], params.T_range[1], params.step)
        )

        # Display which parameter value it is evaluating.
        desc = (
            f"{self.varying_param}={getattr(params, self.varying_param)}"
            if self.varying_param
            else None
        )
        for T in tqdm(temps, desc=desc, disable=not (params.bar)):  # type: ignore
            alg = CtmAlg(
                1 / T,
                model=params.model,
                coupling=params.coupling,
                chi=params.chi,
                C_init=C_init,
                T_init=T_init,
                b_c=params.b_c,
                fixed=params.fixed,
            )
            alg.exe(params.tol, params.count, params.max_steps)

            if params.use_prev:
                C_init, T_init = alg.C, alg.T

            # Save execution time, temperature and the converged corner and edge and
            # the a and b tensors.
            data.append(
                (
                    alg.n_iter,
                    T,
                    alg.C,
                    alg.T,
                    alg.T_fixed,
                    alg.a,
                    alg.a_fixed,
                    alg.b,
                    alg.b_fixed,
                )
            )

        # Return both the parameters and algorithm data in the same dict.
        return params.__dict__ | data_to_dict(data)

    def save(self, data: dict, msg: bool):
        """
        Save the data containing physical properties for varying beta in the `data`
        folder.

        `data` (dict): dictionary containing the data.
        `dir` (str): Directory where the data is saved.
        `msg` (bool): If true, print a message at the start and end of saving.
        """
        if msg:
            print(f"Saving data in folder: '{self.dir}'")

        # Name the file `varying_param` with the value.
        if self.varying_param:
            fn = self.varying_param + f"{data[self.varying_param]}"
        else:
            fn = "data"

        with open(f"data/{self.dir}/{fn}.json", "w") as fp:
            json.dump(data, fp, cls=NumpyEncoder)

        if msg:
            print("Done \n")


def data_to_dict(data: list) -> dict:
    """
    Convert the list with tuple of the data to a dict.
    """
    data = list(zip(*data))  # unpack
    return {
        "number of iterations": data[0],
        "temperatures": data[1],
        "converged corners": data[2],
        "converged edges": data[3],
        "converged fixed edges": data[4],
        "a tensors": data[5],
        "a_fixed tensors": data[6],
        "b tensors": data[7],
        "b_fixed tensors": data[8],
    }


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


class NumpyEncoder(json.JSONEncoder):
    """
    Class for encoding a np.ndarray to JSON
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
