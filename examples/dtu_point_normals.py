import sys
import os
from tqdm import tqdm

from cvt.visualization.util import to_normal

cloud_path = sys.argv[1]

cloud_files = os.listdir(cloud_path)
cloud_files.sort()

with tqdm(cloud_files) as cfs:
    for cf in cfs:
        if cf[-3:] == "ply":
            cfs.set_postfix(cloud=cf)
            to_normal(os.path.join(cloud_path,cf), os.path.join(cloud_path,"Normals",cf)) 

