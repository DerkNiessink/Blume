import scienceplots
import matplotlib.pyplot as plt
import numpy as np

from process import read, compute
from model.post_props import Props

plt.style.use("science")
plt.rcParams["text.usetex"] = True

mags = []
chis = [chi for chi in range(2, 20)]
for chi in chis:
    data = read(folder="22-04 21:42", val=chi)
    m = compute(
        Props.m,
        data["temperatures"],
        np.asarray(data["converged corners"]),
        np.asarray(data["converged edges"]),
        np.asarray(data["a tensors"]),
        np.asarray(data["b tensors"]),
    )[0]
    mags.append(m)

plt.figure(figsize=(7, 5))
plt.plot(chis, mags, "x")
plt.xlabel("$\chi$", fontsize=15)
plt.ylabel("$m$", fontsize=15)
plt.savefig("data/22-04 21:42/critical")
