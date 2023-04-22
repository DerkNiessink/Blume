import matplotlib.pyplot as plt
import sys
import numpy as np
import os

from model.post_props import Props
from process import plot_file, exact_m


folder = sys.argv[1]
path_dir = f"data/{folder}/plots"
params = sys.argv[2:]

if not os.path.isdir(path_dir):
    os.mkdir(path_dir)

T_c = 2 / np.log(1 + np.sqrt(2))

"""
MAGNETIZATIONS
"""
plt.figure(figsize=(7, 5))
T_range = (2.25, 2.29)
# plt.axvline(T_c, color="k", linestyle="dashed", label=r"$T_c$")
for param in params:
    plot_file(param, range=T_range, prop=Props.m, folder=folder)
T, m = exact_m(T_range)
plt.plot(T, m, "k-", label="exact")
plt.legend()
plt.xlabel(r"$T$", fontsize=15)
plt.ylabel(r"m", fontsize=15)
plt.ylim(0)
plt.savefig(f"{path_dir}/magnetizations")

"""
FREE ENERGY
"""
plt.figure(figsize=(7, 5))
T_range = (1, 4)
plt.axvline(T_c, color="k", linestyle="dashed", label=r"$T_c$")
for param in params:
    plot_file(param, range=T_range, prop=Props.f, folder=folder)

plt.legend()
plt.xlabel(r"$T$", fontsize=15)
plt.ylabel(r"f", fontsize=15)
plt.savefig(f"{path_dir}/free energies")

""" 
PLOT EXECUTION TIMES 
"""
plt.figure(figsize=(7, 5))
T_range = (1, 4)
plt.axvline(T_c, color="k", linestyle="dashed", label=r"$T_c$")
for param in params:
    plot_file(param, range=T_range, prop="execution times", folder=folder)

plt.legend()
plt.xlabel(r"$T$", fontsize=15)
plt.ylabel(r"t", fontsize=15)
plt.savefig(f"{path_dir}/execution times")

""" 
ENERGY PER SITE
"""
plt.figure(figsize=(7, 5))
plt.axvline(T_c, color="k", linestyle="dashed", label=r"$T_c$")
T_range = (2, 2.3)
for param in params:
    plot_file(param, range=T_range, prop=Props.Es, folder=folder)
plt.legend()
plt.xlabel(r"$T$", fontsize=15)
plt.ylabel(r"$E_s$", fontsize=15)
plt.savefig(f"{path_dir}/energies per site")
