from CTM_alg import CtmAlg

import numpy as np
import json

for chi in [2, 4, 8, 16, 32]:
    data = []
    for beta in np.arange(0.4, 0.55, 0.001):
        alg = CtmAlg(beta, chi=chi)
        alg.exe(tol=1e-6, max_steps=10000)
        data.append((beta, alg.Z(), abs(alg.m())))

    data = list(zip(*data))
    with open(f"data/data_chi{chi}.json", "w") as fp:
        json.dump(
            {
                "betas": data[0],
                "partition functions": data[1],
                "magnetizations": data[2],
            },
            fp,
        )
