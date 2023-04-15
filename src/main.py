from CTM_alg import CtmAlg

import matplotlib.pyplot as plt
import numpy as np


alg = CtmAlg(beta=0.1, chi=16)
alg.exe(max_steps=1000, tol=1e-4)
print(f"Z = {alg.Z()}")
print(f"m = {alg.m()}")

plt.plot((np.diff(alg.sv_sums)))
plt.ylabel("abs diff sum singular values of C")
plt.xlabel("n steps")
plt.show()

plt.plot(alg.partition_functions)
plt.ylabel("Z")
plt.xlabel("n steps")
plt.show()

plt.plot(alg.magnetizations)
plt.ylabel("m")
plt.xlabel("n steps")
plt.show()
