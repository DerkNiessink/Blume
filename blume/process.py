import json
import matplotlib.pyplot as plt
import numpy as np
import typing

from .model.post_props import Prop, PropFunction


@typing.no_type_check
def plot_file(fn: str, range: tuple, prop: Prop | str, folder: str) -> plt.Line2D:
    """
    Plot a given variable against temperature.

    `param` (str): Parameter to plot.
    `val` (int): Parameter value to plot.
    `range` (tuple): Range of temperatures to plot.
    `prop` (Prop | str): Function for calculating the variable or a string of
    the name of the property.
    `folder` (str): Folder that contains the data of the specific chi.

    Returns the created line2D object.
    """
    data = read(folder, fn)

    # If string is given, the propery already exists in the data, else compute
    # the property with the function.
    y = data[prop] if type(prop) == str else compute(prop, data)

    temps = data["temperatures"]
    # Find the indices closest to the range values.
    lower_value = min(temps, key=lambda x: abs(x - range[1]))
    upper_value = min(temps, key=lambda x: abs(x - range[0]))
    lower_index = temps.index(lower_value)
    upper_index = temps.index(upper_value)

    (line,) = plt.plot(temps[upper_index:lower_index], y[upper_index:lower_index])
    return line


def read(folder: str, fn: str) -> dict:
    """
    Read the data in a specific folder for a specific parameter value.

    folder (str): name of the folder that contains the data.
    fn (str): file name of the json file.
    val (int): value of the parameter corresponding to the desired file to read.
    """
    with open(f"data/{folder}/{fn}.json", "r") as f:
        return dict(json.loads(f.read()))


def compute(
    prop: PropFunction,
    data: dict,
) -> list:
    """
    Compute the corresponding property for a given dictionary of data from the
    algorithm.

    `prop` (Prop): Desired thermodynamic property to compute from data.
    `data` (dict): Dictionary containing the algorithm data.

    Returns a list with the computed property for all temperatures in data.
    """
    data_zip = zip(
        data["temperatures"],
        np.asarray(data["converged corners"]),
        np.asarray(data["converged edges"]),
        np.asarray(data["converged fixed edges"]),
        np.asarray(data["a tensors"]),
        np.asarray(data["a_fixed tensors"]),
        np.asarray(data["b tensors"]),
        np.asarray(data["b_fixed tensors"]),
    )
    return [
        prop(
            {
                "beta": 1 / item[0],
                "C": item[1],
                "T": item[2],
                "T_fixed": item[3],
                "a": item[4],
                "a_fixed": item[5],
                "b": item[6],
                "b_fixed": item[7],
            }
        )
        for item in data_zip
    ]


def exact_m(range: tuple[float, float], step=0.0001) -> tuple[list, list]:
    """
    Give the exact solution for the magnetization on a given temperature range.

    range (tuple): The desired temperature range to plot.
    step (float): stepsize.
    """
    T_c = 2 / np.log(1 + np.sqrt(2))
    x, y = [], []
    for T in np.arange(range[0], range[1], step):
        x.append(T)
        y.append(0 if T > T_c else (1 - np.sinh(2 * 1 / T) ** (-4)) ** (1 / 8))
    return x, y
