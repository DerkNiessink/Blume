import os
import json
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")
plt.rcParams["text.usetex"] = True


def plot_all_chi(
    chi_list: list, range: tuple, y: str, xlabel: str, ylabel: str, fn: str, ymin=0
):
    plt.figure(figsize=(7, 5))

    for chi in chi_list:
        filename = os.path.join("data", "chi" + f"{chi}.json")
        with open(filename, "r") as f:
            try:
                data = json.loads(f.read())
            except:
                UnicodeDecodeError

        # Find the indices closest to the range values.
        lower_value = min(data["temperatures"], key=lambda x: abs(x - range[1]))
        upper_value = min(data["temperatures"], key=lambda x: abs(x - range[0]))
        lower_index = data["temperatures"].index(lower_value)
        upper_index = data["temperatures"].index(upper_value)

        chi = data["chi"]
        plt.plot(
            data["temperatures"][lower_index:upper_index],
            data[y][lower_index:upper_index],
            label=f"$\chi={chi}$",
        )
    plt.legend()
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.ylim(ymin=ymin)
    plt.savefig(fn)


plot_all_chi(
    [4, 8, 12, 24],
    range=(2.24, 2.3),
    y="magnetizations",
    xlabel=r"$T$",
    ylabel=r"m",
    fn="data/phase2",
)

plot_all_chi(
    [4, 8, 12, 24],
    range=(1, 4),
    y="free energies",
    xlabel=r"$T$",
    ylabel=r"$f$",
    fn="data/free2",
    ymin=-2.45,
)
