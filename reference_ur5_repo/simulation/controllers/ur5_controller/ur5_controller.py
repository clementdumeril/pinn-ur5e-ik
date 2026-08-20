"""
Pick & Place UR5e - Saisie top-down, camera + Robotiq 3F
"""
import os
import sys
from math import pi

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from ur5 import UR5

PI = pi

ur5 = UR5()

# Vision desactivee par defaut : la calibration pixels -> coordonnees reelles du
# VGG16 est fausse pour ce monde (~148 mm d'erreur, cible hors de portee).
# Voir GUIDE_DES_CODES.md 15. Passer a True pour retester.
USE_VISION = False

# Position initiale stable
ur5.move_to_config([0, 0, 0, 0, 0, 0])

# ============================================================
# Pose de lecture camera
# ============================================================
ur5.actuate_gripper(0)
ur5.move_to_pose([-0.1, -.68, .45], [PI, 0, -PI/2], wrist='up')

for _ in range(30):
    ur5.supervisor.step(ur5.timestep)

# ============================================================
# Detection des positions
# ============================================================
if USE_VISION and ur5.model is not None:
    bottle_position = ur5.predict_bottle_position()
else:
    frame = ur5.get_bottle_frame()
    bottle_position = (frame[0, 3], frame[1, 3])

cx, cy = bottle_position[0], bottle_position[1]

tray_frame = ur5.get_node_frame("BLUE_TRAY")
if tray_frame is not None:
    tx, ty = tray_frame[0, 3], tray_frame[1, 3]
else:
    tx, ty = cx - 0.4, cy

print(f"Cube: ({cx:.3f}, {cy:.3f}) | Zone bleue: ({tx:.3f}, {ty:.3f})")
print("Lancement pick & place...")

# ============================================================
# PICK - Saisie top-down
# ============================================================
ur5.move_to_pose([cx, cy, .40], [PI, 0, -PI/2], wrist='up')
ur5.move_to_pose([cx, cy, .12], [PI, 0, -PI/2], wrist='up', duration=4)
ur5.actuate_gripper(1)

# ============================================================
# TRANSPORT
# ============================================================
ur5.move_to_pose([cx, cy, .40], [PI, 0, -PI/2], wrist='up')
ur5.move_to_pose([tx, ty, .40], [PI, 0, -PI/2], wrist='up')

# ============================================================
# PLACE
# ============================================================
ur5.move_to_pose([tx, ty, .12], [PI, 0, -PI/2], wrist='up', duration=4)
ur5.actuate_gripper(0)
ur5.move_to_pose([tx, ty, .40], [PI, 0, -PI/2], wrist='up')

# ============================================================
# Retour pose de lecture
# ============================================================
ur5.move_to_pose([-0.1, -.68, .45], [PI, 0, -PI/2], wrist='up')

print("Termine !")
