# MCUP
MCUP (Monte Carlo Uncertainity Propagation) is a Python library that estimates the uncertainty of least squares fit parameters with Monte Carlo.

## Status
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mcup.svg)](https://pypi.org/project/mcup/) [![PyPI version shields.io](https://img.shields.io/pypi/v/mcup.svg)](https://pypi.org/project/mcup/) [![master](https://github.com/detrin/MCUP/actions/workflows/package-main.yml/badge.svg)](https://github.com/detrin/MCUP/actions/workflows/package-main.yml) [![Documentation Status](https://readthedocs.org/projects/mcup/badge/?version=latest)](https://readthedocs.org/projects/mcup/?badge=latest) [![codecov](https://codecov.io/gh/detrin/MCUP/branch/master/graph/badge.svg?token=Dx6elQkztR)](https://codecov.io/gh/detrin/MCUP)


## Scope
The aim of this package is to estimate the error of regression parameters based on error intervals of the input data. 

PEE – Parameter Error Estimator, a bootstraping method which, takes input data for lsq (x, y, x_err, y_err), generates datapoints within given errors and calculate mean and std of parameters from lsq fit.


## Installing MCUP
#### PyPI

To install mlxtend, just execute  

```bash
python3 -m pip install mcup  
```

#### Dev Version

The MCUP version on PyPI may always be one step behind. You can install the latest development version from the GitHub repository by executing

```bash
python3 -m pip install git+https://github.com/detrin/MCUP.git#egg=mcup
```

## Example
```python
import numpy as np
from mcup import Measurement

x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y_data = np.array([0.1, 0.9, 2.2, 2.8, 3.9, 5.1])

y_sigma = np.array([0.0, 0.1, 0.1, 0.1, 0.1, 0.1])


def fun(x, c):
    return c[0] * x + c[1]

c_initial_guess = [0.0, 0.0]

measurement = Measurement(x=x_data, y=y_data, y_err=y_sigma)
measurement.set_function(fun, c_initial_guess)

params_mean, params_std = measurement.evaluate_params(iter_num=1000)
print(params_mean)
# [0.9901532  0.02477131]
print(params_std)
# [0.01881003 0.04965347]

params_mean, params_std = measurement.evaluate_params(atol=1e-4, rtol=1e-4)
print(params_mean)
# [0.98854127 0.02771339]
print(params_std)
# [0.0172098  0.04729087]
```

## Contributing
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.
