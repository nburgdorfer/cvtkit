import numpy as np
import os
from scipy.spatial.transform import Rotation as R


t1 = np.asarray([-0.003525181, -0.014807079, -0.001079156])
q1 = [-0.016358169, 0.878568456, -0.476443744, -0.029175984]

t2 = np.asarray([-0.003740556, -0.014919663, -0.000981085])
q2 = [-0.016861910, 0.877006554, -0.479100057, -0.032222899]


rot1 = R.from_quat(q1).as_matrix()
rot2 = R.from_quat(q2).as_matrix()

P1 = np.eye(4)
P2 = np.eye(4)

P1[:3,:3] = rot1
P1[:3,3] = -rot1@t1
P2[:3,:3] = rot2
P2[:3,3] = -rot2@t2

P1 = np.linalg.inv(P1)
P2 = np.linalg.inv(P2)

with open("data3/P1.txt", "w") as f:
    f.write(f"{P1[0,0]} {P1[0,1]} {P1[0,2]} {P1[0,3]}\n")
    f.write(f"{P1[1,0]} {P1[1,1]} {P1[1,2]} {P1[1,3]}\n")
    f.write(f"{P1[2,0]} {P1[2,1]} {P1[2,2]} {P1[2,3]}\n")

with open("data3/P2.txt", "w") as f:
    f.write(f"{P2[0,0]} {P2[0,1]} {P2[0,2]} {P2[0,3]}\n")
    f.write(f"{P2[1,0]} {P2[1,1]} {P2[1,2]} {P2[1,3]}\n")
    f.write(f"{P2[2,0]} {P2[2,1]} {P2[2,2]} {P2[2,3]}\n")
