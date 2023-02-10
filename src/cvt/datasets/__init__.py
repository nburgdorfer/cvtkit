# datasets/__init__.py

"""A collection of dataset-specific functions.

This package includes the following modules:

- `dtu`: A collection of functions for the DTU dataset.
- `tanks`: A collection of functions for the Tanks and Temples dataset.
- `blended`: A collection of functions for the BlendedMVS(MVG) dataset.
"""
import os
import sys

FILE_ROOT=os.path.abspath(os.path.dirname(__file__))
PYTHON_ROOT=os.path.abspath(os.path.dirname(FILE_ROOT))

sys.path.append(PYTHON_ROOT)

from .dtu import *
from .tanks import *
from .blended import *
