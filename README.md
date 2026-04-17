# CSCI 619 - Class Project

**T block:**
```
# Dimensions:

<-     20cm    ->
_________________
|                |   ^ 5 cm   ^ 25cm
|____       _____|   |        |
     |     |    ^             |
     |     |    | 15 cm       |
     |     |    |             |
     |_____|    |             |
     < 5cm >

+z height is 5cm


# Links
# - all p_i are on the xy plane (i.e. floor level)

p4 _____________ p5
|                 |
p3 _ p2     p7 _ p6
     |     |
     |     |
     |     |
     p1___ p8


# Reference frame:
# - centered at the midway point of the T 
# - origin-x is +10cm from the left edge
# - origin-y is +12.5cm from the bottom edge
# - origin-z is on the floor level

_________________
|                |
|____       _____|
    |      |
       +y
       ^
       |--> +x

    |_____|
```

Note that urdfs can be easily visualized using the [URDF Visualizer](https://marketplace.cursorapi.com/items/?itemName=morningfrog.urdf-visualizer) VSCode extension.


## Installation

Create and activate the conda environment, then install the package in editable mode.

```bash
conda create -n pusht619 python=3.10
conda activate pusht619
pip install -e ".[dev]"
```


## Running the project

```bash
# Run with claude code:
! source ~/miniconda3/etc/profile.d/conda.sh && conda activate pusht619 && python scripts/jaxsim_reference.py
```