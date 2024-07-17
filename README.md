# EMPOT
This is an implementation of EMPOT, an algorithm designed for solving the partial alignment of cryo-EM density maps. More detailes can be found on our manuscript https://arxiv.org/abs/2311.00850.

## Requirements
The following python packages are required to run our code:
- matplotlib (3.7.1)
- mrcfile (1.4.3)
- numpy (1.23.5)
- pandas (2.0.0)
- Pillow (9.5.0)
- POT (0.9.0)
- scipy (1.10.1)
- seaborn (0.12.2)
- numba (0.56.4)
- Bio (1.5.9)
- tornado (4.5.3)

## Usage
The user can download the python files in [src](https://github.com/artajmir3/EMPOT/tree/main/src) folder. This implementation can be used in a couple of ways. The user either can use it just as an alignment method or as an model building method featuring the options to select best alignment for subunits and generate corresponding PDB files for them. A comprehensive example for both usages is available in [examples](https://github.com/artajmir3/EMPOT/tree/main/examples) folder. For now, we did not implement the refinement step of the alignment in python. For this step, user can generate the output of alignment as a PDB file and perform the refinement via ChimeraX.
