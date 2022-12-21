import os
import sys

FILE_ROOT=os.path.abspath(os.path.dirname(__file__))
PYTHON_ROOT=os.path.abspath(os.path.dirname(FILE_ROOT))
SRC_ROOT=os.path.abspath(os.path.dirname(PYTHON_ROOT))
sys.path.append(SRC_ROOT)

from .metrics_2d import *
from .metrics_3d import *
from .util import *
