# Blume

Application for evaluating the Blume-Capel (BC) model using the Corner Transfer Renormalization Group (CTMRG) method. An extensive explanation and analysis of the BC model using this implementation is presented in [this thesis](graphs/Thesis-final-version.pdf).


## Installation

1. Clone the repository.
2. [Create a virtual environment and activate it](https://docs.python.org/3/library/venv.html).
3. Install the requirements: `pip install -r requirements.txt`


## Usage 

```python 
import blume

result = blume.run.Results(varying_param="coupling", range=[0, 0.5, 1, 1.5])
params = blume.run.ModelParameters(model = "blume", var_range=(0.1, 1.75), step=0.001, tol=1e-9, max_steps=int(10e9), use_prev=True, chi=16)
result.get(params)


  
```
