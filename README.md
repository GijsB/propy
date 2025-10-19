# propy
A package for dimensioning and optimizing (marine) propellers.


## Installation
This package is not (yet?) hosted on the Python Package-index (PyPi), but fortunately the package can still be installed
using `pip`:

```commandline
pip install git+https://github.com/GijsB/propy
```

After installation, the package can be used like any other python package.


## Usage

### Propeller types
The propy module contains parametric propeller models of the following types:
 - `WageningenBPropeller`: The famous Wageningen B-type propeller
 - ...


### Open-water model (1 Quadrant)
The most basic use-case is to use the open-water model of a propeller. Such a model is able to specify the thrust- and
torque-coefficient for different advance-ratio's. These coefficients are defined as follows:
 - Advance ratio: `j = speed / rotation_speed / diameter`, `0 <= j <= j_max`
 - Thrust coefficient: `kt(j) = thrust / rho / rotation_speed^2 / diameter^4`, `kt_min <= kt <= kt_max `
 - Torque coefficient: `kq(j) = torque / rho / rotation_speed^2 / diameter^5`, `kq_min <= kq <= kq_max`

The code below shows how the open water model kan be used to obtain the thrust and torque produced by the propeller.

```python
>>> from propy import WageningenBPropeller
>>>
>>> speed = 10  # 10 m/s speed
>>> rotation_speed = 20  # 20 Hz rotation speed
>>> rho = 1000  # 1000 kg/m3 water density
>>>
>>> prop = WageningenBPropeller()
>>>
>>> j = speed / rotation_speed / prop.diameter
>>> thrust = prop.kt(j) * rho * rotation_speed**2 * prop.diameter**4
>>> torque = prop.kq(j) * rho * rotation_speed**2 * prop.diameter**5
>>> efficiency = prop.eta(j)
>>>
>>> print(f'{j=:1.2}, {thrust=:1.2} N, {torque=:1.2} Nm, {efficiency=:1.2}')
j=0.5, thrust=6.9e+04 N, torque=9.5e+03 Nm, efficiency=0.58

```

The code below shows how an open-water chart can conveniently be generated
```python
import numpy as np
import matplotlib.pyplot as plt

from propy import WageningenBPropeller

prop = WageningenBPropeller(
    diameter=0.3,
    blades=2,
    area_ratio=0.9,
    pd_ratio=0.5
)

j = np.linspace(0, prop.j_max)

plt.figure()
plt.plot(j, prop.kt(j), label='kt')
plt.plot(j, prop.kq(j)*10, label='10*kq')
plt.plot(j, prop.eta(j), label='eta')

plt.xlabel('Advance ratio J')
plt.title(f'Open-water chart')
plt.grid()
plt.legend()
```

![Open water chart](doc/open_water_chart.png)



### 4-Quadrant model
The open-water model can only be used to calculate working points where the propeller is actually "propelling". When the
propeller is breaking of reversing, a 4-quadrant model must be used. The propeller characteristics for these working 
points can be very complex. Although some data is available, this is not currently implemented in this library. Instead,
a simple function fit is used to get a very rough approximation.

The 4-quadrant model has 3 main coefficients, which are somewhat similar to the 1-quadrant model:
- The load angle: 
  - `beta = atan(speed / 0.7 / pi / rotation_speed / diameter)`
  - `beta = atan(j / 0.7 / pi)`
- The thrust coefficient:
  - `ct = 8 * thrust / (speed^2 + (0.7 * pi * rotation_speed * diameter)^2) / pi / rho / diameter^2`
  - `ct = 8 * kt / pi / (j^2 + 0.7^2 * pi^2)`
- The torque coefficient:
  - `cq = 8 * torque / (speed^2 + (0.7 * pi * rotation_speed * diameter)^2) / pi / rho / diameter^3`
  - `cq = 8 * kq / pi / (j^2 + 0.7^2 * pi^2)`

The code below shows how the 4-quadrant model can give an estimate of the thrust and drag for a working point with a
very low rotation speed. At this working point, the advance ratio would be higher than `j_max`, so the 1-quadrant model
cannot be used.

```python
>>> from propy import WageningenBPropeller
>>> from math import atan, pi
>>>
>>> speed = 10  # 10 m/s speed
>>> rotation_speed = 10  # 10 Hz rotation speed
>>> rho = 1000  # 1000 kg/m3 water density
>>>
>>> prop = WageningenBPropeller()
>>>
>>> j = speed / rotation_speed / prop.diameter
>>> beta = atan(j / 0.7 / pi)
>>> thrust = prop.ct(beta) * (speed**2 + (0.7 * pi * rotation_speed * prop.diameter)**2) * pi * rho * prop.diameter**2 / 8
>>> torque = prop.cq(beta) * (speed**2 + (0.7 * pi * rotation_speed * prop.diameter)**2) * pi * rho * prop.diameter**3 / 8
>>>
>>> print(f'{j=:1.2}, {beta=:1.2} rad, {thrust=:1.2} N, {torque=:1.2} Nm')
j=1.0, beta=0.43 rad, thrust=-4.7e+03 N, torque=-2.7e+01 Nm

```

## Development

### Installation
To work on the development of this project, the project needs to be cloned locally. It can be convenient to install the 
package in "editable" mode. It's also recommended to install all the dependencies in a local venv.

**MacOS/Linux**
```commandline
git clone git@github.com:GijsB/propy.git
cd propy
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Windows**
```commandline
git clone git@github.com:GijsB/propy.git
cd propy
python -m venv .venv
source .venv\Scripts\activate.bat
pip install -e ".[dev]"
```


### Git workflow
The main branch only contains releases, which are also tagged with the version number. When it's a good time for a new 
release, the changes from the develop branch are pulled into the main branch. New features are developed in 
`feature/...` branches. These changes are reviewed in a GitHub pull-request. After all the checks are passed, they can 
be merged into the develop branch. 


### Testing & validation
The following steps could be performed manually during development, but are also tested upon each pull-request.
- The code style is tested using `flake8 .`
- The type checking can be performed using `mypy .`
- The unit tests can be executed using `pytest .`


### Test status:
- Main branch: ![example workflow](https://github.com/GijsB/propy/actions/workflows/tests.yml/badge.svg?branch=main)
- Develop branch: ![example workflow](https://github.com/GijsB/propy/actions/workflows/tests.yml/badge.svg?branch=develop)