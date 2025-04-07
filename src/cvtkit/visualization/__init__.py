# visualization/__init__.py

"""Sub-package including functions used for visualization.

This sub-package includes the following modules:

- `latex`: Latex visaul creation functions.
- `util`: General visualization utilities.
- `video`: Video-based visualization functions.
"""

import os
import sys

FILE_ROOT=os.path.abspath(os.path.dirname(__file__))
PYTHON_ROOT=os.path.abspath(os.path.dirname(FILE_ROOT))
SRC_ROOT=os.path.abspath(os.path.dirname(PYTHON_ROOT))
RENDERING_ROOT=os.path.abspath(os.path.join(SRC_ROOT, "cpp/rendering/build/"))

sys.path.append(PYTHON_ROOT)
sys.path.append(RENDERING_ROOT)

from .latex import *
from .util import *
from .video import *
