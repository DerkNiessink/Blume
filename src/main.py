from CTM_alg import CtmAlg

import matplotlib.pyplot as plt
import numpy as np


alg = CtmAlg(beta=0.5, chi=8)
alg.exe(n_steps=100)
print(f"Z = {alg.Z()}")
print(f"m = {alg.m()}")

plt.plot(np.abs(np.diff(alg.sv_sums)))
plt.ylabel("abs diff sum singular values of C")
plt.xlabel("n steps")
plt.show()
