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

```python
>>> from propy import WageningenBPropeller
>>> WageningenBPropeller()
WageningenBPropeller(blades=4, diameter=1.0, area_ratio=0.5, pd_ratio=0.8)

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