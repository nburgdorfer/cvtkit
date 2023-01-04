# metrics/__init__.py

"""Sub-package including 2D and 3D metrics.

This sub-package includes the following modules:

- `m2d`: 2D metrics.
- `m3d`: 3D metrics.
- `util`: Metric utilities.
"""

import os
import sys

FILE_ROOT=os.path.abspath(os.path.dirname(__file__))
PYTHON_ROOT=os.path.abspath(os.path.dirname(FILE_ROOT))
SRC_ROOT=os.path.abspath(os.path.dirname(PYTHON_ROOT))
sys.path.append(SRC_ROOT)

from .m2d import *
from .m3d import *
from .util import *
