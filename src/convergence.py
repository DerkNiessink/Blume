from CTM_alg import CtmAlg
import scienceplots

import matplotlib.pyplot as plt

plt.style.use("science")
plt.rcParams["text.usetex"] = True

alg = CtmAlg(beta=0.5, chi=8)
alg.exe(tol=1e-7, max_steps=10000)

plt.figure(figsize=(7, 5))
plt.plot(alg.sv_sums[1:])
plt.ylabel(r"\[ \sum_{k} | s_k - s'_k | \]", fontsize=15)
plt.xlabel(r"$n$ steps", fontsize=15)
plt.savefig("data/sv_sums")

plt.figure(figsize=(7, 5))
plt.plot(alg.magnetizations)
plt.xlabel(r"$n$ steps", fontsize=15)
plt.ylabel(r"$m$", fontsize=15)
plt.savefig("data/m_conv")
