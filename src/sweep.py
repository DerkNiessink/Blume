from CTM_alg import CtmAlg

import numpy as np
import json
from tqdm import tqdm

for chi in [4, 8, 12, 24]:
    data = []
    for beta in tqdm(np.arange(0.33, 0.67, 0.0001), desc=f"Chi = {chi}"):
        alg = CtmAlg(beta, chi=chi)
        alg.exe(tol=1e-7, max_steps=100000)

        # Save temperature, partition function, magnetization and free energy
        data.append((1 / beta, alg.Z(), abs(alg.m()), alg.f()))

    data = [chi] + list(zip(*data))

    with open(f"data/chi{chi}.json", "w") as fp:
        json.dump(
            {
                "chi": data[0],
                "temperatures": data[1],
                "partition functions": data[2],
                "magnetizations": data[3],
                "free energies": data[4],
            },
            fp,
        )
