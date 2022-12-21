import os
import sys

FILE_ROOT=os.path.abspath(os.path.dirname(__file__))
PYTHON_ROOT=os.path.abspath(os.path.dirname(FILE_ROOT))
SRC_ROOT=os.path.abspath(os.path.dirname(PYTHON_ROOT))
RENDERING_ROOT=os.path.abspath(os.path.join(SRC_ROOT, "cpp/rendering/build/"))

sys.path.append(PYTHON_ROOT)
sys.path.append(RENDERING_ROOT)

from .latex_util import *
from .util import *
from .video import *
