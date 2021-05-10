# Pattern Formation
Pattern Formation code and tests for the paper, 
(Minimizing The Maximum Distance Traveled To Form Patterns With Systems of 
 Mobile Robots)[https://www.cccg.ca/proceedings/2020/proceedings.pdf#page=83].


## Dependencies
- Python >= 3.7 
- Python dependencies (installed automatically with package):
  - numpy
  - matplotlib
  - cvxopt

## Installation
To install the pafo (pattern formation) python package, run:
```bash
pip install git+https://github.com/jaredraycoleman/pafo # to install from git 
pip install ./ # if you cloned the repository
```

## Tests
After installation, you can run the test scripts in the ```tests/``` directory:
```bash
    # creates diretory of PNGs in tests/visualize_solution
    python tests/visualize_solution.py 
```

## Archive
Additional scratch code with ad-hoc tests can be found in ```archive/```.