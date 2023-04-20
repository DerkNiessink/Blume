import os
import json
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use("science")
plt.rcParams["text.usetex"] = True


def plot_all_chi(
    chi_list: list,
    range: tuple,
    y: str,
    xlabel: str,
    ylabel: str,
    folder: str,
    ymin=0,
):
    """
    Plot a given variable against temperature.

    `chi_list` (list): Bond dimensions to plot.
    `range` (tuple): Range of temperatures to plot.
    `y` (str): Variable name
    `xlabel` and `ylabel` (str): axes labels.
    `ymin` (int): minimum y value to plot.
    """
    plt.figure(figsize=(7, 5))

    for chi in chi_list:
        filename = os.path.join("data", folder, "chi" + f"{chi}.json")
        with open(filename, "r") as f:
            data = json.loads(f.read())

        # Find the indices closest to the range values.
        lower_value = min(data["temperatures"], key=lambda x: abs(x - range[1]))
        upper_value = min(data["temperatures"], key=lambda x: abs(x - range[0]))
        lower_index = data["temperatures"].index(lower_value)
        upper_index = data["temperatures"].index(upper_value)

        chi = data["chi"]
        plt.plot(
            data["temperatures"][lower_index:upper_index],
            data[y][lower_index:upper_index],
            "x",
            label=f"$\chi={chi}$",
        )

        # Plot the exact solution of the magnetization as function of temperature
        if y == "magnetizations":
            T, m = exact_m(range)
            plt.plot(T, m, label="exact")

    plt.legend()
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.ylim(ymin=ymin)
    plt.savefig(f"data/{folder}/plots/{y}")


def exact_m(range: tuple[float], step=0.0001) -> list:
    """
    Give the exact solution for the magnetization on a given temperature range.

    range (tuple): The desired temperature range to plot.
    step (float): stepsize.
    """
    T_c = 2.26918531421
    x, y = [], []
    for T in np.arange(range[0], range[1], step):
        x.append(T)
        y.append(0 if T > T_c else (1 - np.sinh(2 * 1 / T) ** (-4)) ** (1 / 8))
    return x, y


if __name__ == "__main__":
    folder = "19-04 16:31"
    chi_list = [4]

    if not os.path.isdir(f"data/{folder}/plots"):
        os.mkdir(f"data/{folder}/plots")

    plot_all_chi(
        chi_list,
        range=(2, 2.3),
        y="magnetizations",
        xlabel=r"$T$",
        ylabel=r"m",
        folder=folder,
    )

    plot_all_chi(
        chi_list,
        range=(1, 4),
        y="free energies",
        xlabel=r"$T$",
        ylabel=r"$f$",
        ymin=-2.12,
        folder=folder,
    )

    plot_all_chi(
        chi_list,
        range=(2, 2.3),
        y="execution times",
        xlabel=r"$T$",
        ylabel=r"t",
        folder=folder,
    )
