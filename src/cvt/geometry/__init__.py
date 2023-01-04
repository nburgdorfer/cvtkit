# geometry/__init__.py

"""Sub-package including functions relating to 2D and 3D geometry.

This sub-package includes the following modules:

- `g2d`: 2D geometric functions.
- `g3d`: 3D geometric functions.
"""

import os
import sys

FILE_ROOT=os.path.abspath(os.path.dirname(__file__))
PYTHON_ROOT=os.path.abspath(os.path.dirname(FILE_ROOT))
SRC_ROOT=os.path.abspath(os.path.dirname(PYTHON_ROOT))
RENDERING_ROOT=os.path.abspath(os.path.join(SRC_ROOT, "cpp/rendering/build/"))

sys.path.append(PYTHON_ROOT)
sys.path.append(RENDERING_ROOT)


from .g2d import *
from .g3d import *
