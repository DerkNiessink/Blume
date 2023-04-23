import os
import json
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import sys
import types

try:
    from model.post_props import Props
except:
    from blume.model.post_props import Props


plt.style.use("science")
plt.rcParams["text.usetex"] = True


def plot_file(param: int, range: tuple, prop: types.FunctionType | str, folder: str):
    """
    Plot a given variable against temperature.

    `chi` (int): Bond dimension to plot.
    `range` (tuple): Range of temperatures to plot.
    `y` (function | str): Function for calculating the variable or a string of
    the name of the property.
    `folder` (str): Folder that contains the data of the specific chi.
    """
    data = read(folder, param)

    # If string is given, the propery already exists in the data, else compute
    # the property with the function.
    y = data[prop] if type(prop) == str else compute(prop, data)

    temps = data["temperatures"]
    # Find the indices closest to the range values.
    lower_value = min(temps, key=lambda x: abs(x - range[1]))
    upper_value = min(temps, key=lambda x: abs(x - range[0]))
    lower_index = temps.index(lower_value)
    upper_index = temps.index(upper_value)

    label = f"$L={param}$" if data["boundary conditions"] else f"$\chi={param}$"
    plt.plot(temps[upper_index:lower_index], y[upper_index:lower_index], label=label)


def read(folder: str, val: int) -> dict:
    """
    Read the data in a specific folder for a specific chi or L.

    folder (str): name of the folder that contains the data.
    val (int): chi or L value corresponding to the desired file to read.
    """
    try:
        f = open(f"data/{folder}/chi{val}.json", "r")
    except FileNotFoundError:
        f = open(f"data/{folder}/L{val}.json", "r")

    with f:
        return json.loads(f.read())


def compute(
    prop: Props,
    data: dict,
) -> list:
    """
    Compute the corresponding property for a given dictionary of data from the
    algorithm.

    prop (Props): Desired thermodynamic property to compute from `data`.
    data (dict): Dictionary containing the algorithm data.
    """
    temps, C_tensors, T_tensors, a_tensors, b_tensors = (
        data["temperatures"],
        np.asarray(data["converged corners"]),
        np.asarray(data["converged edges"]),
        np.asarray(data["a tensors"]),
        np.asarray(data["b tensors"]),
    )
    return [
        prop(C, T, 1 / temp, a, b)
        for temp, C, T, a, b in zip(temps, C_tensors, T_tensors, a_tensors, b_tensors)
    ]


def exact_m(range: tuple[float], step=0.0001) -> list:
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
