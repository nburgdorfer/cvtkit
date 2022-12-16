import os
import sys

FILE_ROOT=os.path.abspath(os.path.dirname(__file__))
PYTHON_ROOT=os.path.abspath(os.path.dirname(FILE_ROOT))
SRC_ROOT=os.path.abspath(os.path.dirname(PYTHON_ROOT))
sys.path.append(SRC_ROOT)

from metrics.metrics_2d import *
from metrics.metrics_3d import *
from metrics.util import *
