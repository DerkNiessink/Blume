import os
import json
import matplotlib.pyplot as plt
import scienceplots

plt.figure()
for filename in os.listdir("data"):
    filename = os.path.join("data", filename)
    with open(filename, "r") as f:
        try:
            data = json.loads(f.read())
        except:
            UnicodeDecodeError
    plt.plot(data["betas"], data["magnetizations"])

plt.savefig("data/phase")
