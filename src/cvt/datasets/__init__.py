# datasets/__init__.py

"""A collection of dataset-specific functions.

This package includes the following modules:

- `dtu`: A collection of functions for the DTU dataset.
"""
import os
import sys

FILE_ROOT=os.path.abspath(os.path.dirname(__file__))
PYTHON_ROOT=os.path.abspath(os.path.dirname(FILE_ROOT))

sys.path.append(PYTHON_ROOT)

from .DTU import *
