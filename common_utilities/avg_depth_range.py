import numpy as np
import sys
import os


# custom imports
sys.path.append("../common_utilities")
from utils import *

data_path = sys.argv[1]

cams = load_cams(data_path)
ranges = []
print("Number of views:",len(cams))

for i,cam in enumerate(cams):
    rng = cam[1,3,3] - cam[1,3,0]
    ranges.append(rng)
np.asarray(ranges)

avg = np.mean(ranges)
mn = np.min(ranges)
mx = np.max(ranges)

print("Average: {}   Min: {}   Max: {}".format(avg,mn,mx))
