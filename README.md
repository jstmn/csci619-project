# CSCI 619 - Class Project

## Project Layout

```
csci619_class_project/
├── pyproject.toml        # Build + tool configuration
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   └── pusht619/         # Main package
│       ├── __init__.py
│       └── __main__.py
└── tests/                # Pytest suite
    ├── __init__.py
    └── test_smoke.py
```

## Installation

Create and activate the conda environment, then install the package in editable mode.

```bash
conda create -n pusht619 python=3.10
conda activate pusht619
conda install jaxsim -c conda-forge
pip install -e ".[dev]"
```
