"""
Controleur Webots — VALIDATION du PINN IK sous contraintes (evitement d'obstacle).

A/B en direct sur le vrai UR5e de Webots :
  - PINN  : atteint la cible EN EVITANT l'obstacle (angles issus du reseau, numpy pur)
  - ikpy  : solveur standard, atteint la cible mais FONCE dans l'obstacle

Aucune dependance a PyTorch : le reseau est charge depuis pinn_ik_ur5e.npz (numpy).
L'obstacle est lu EN DIRECT via le superviseur -> deplacez-le, le PINN replanifie.
"""

import os
import sys
import json
import numpy as np

HERE = os.path.dirname(__file__)
SHARED = os.path.join(HERE, "..", "detection.json")   # ecrit par overhead_vision
CUBE_Z = 0.32
PARENT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # pinn_ik_project
if PARENT not in sys.path:
    sys.path.append(PARENT)

ROBOT_BASE_Z = 0.45          # le UR5e est monte a z=0.45 (repere base = monde translate)
OBS_RADIUS = 0.09            # doit correspondre au rayon de DEF OBSTACLE dans le .wbt
APPROACH_H = 0.12            # on vise un point d'approche au-dessus du cube (evite la table)
MARGIN = 0.02

MOTOR_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


# ---------------------------------------------------------------- reseau (numpy)
class NumpyPINN:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.n = int(d["n_linear"])
        self.W = [d[f"W{i}"] for i in range(self.n)]
        self.b = [d[f"b{i}"] for i in range(self.n)]
        self.in_mean, self.in_std = d["in_mean"], d["in_std"]

    @staticmethod
    def _silu(x):
        return x / (1.0 + np.exp(-x))

    def __call__(self, target_r, obs_r):
        x = np.concatenate([target_r, obs_r]).astype(np.float64)
        x = (x - self.in_mean) / self.in_std
        for i in range(self.n - 1):
            x = self._silu(self.W[i] @ x + self.b[i])
        return self.W[-1] @ x + self.b[-1]   # 6 angles


# ------------------------------------------------------- cinematique directe numpy
# DH calibree sur le vrai UR5e de Webots (cf. ur5e_kinematics.py). Angles == moteurs.
_PI = np.pi
_A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
_ALPHA = [_PI / 2, 0.0, 0.0, _PI / 2, -_PI / 2, 0.0]
_D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]


def _dh(theta, a, alpha, d):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0, sa, ca, d],
                     [0, 0, 0, 1]])


def fk(thetas):
    """Retourne ee(3,) (flange) et joint_pos (7,3) dans le repere base de la FK."""
    T = np.eye(4)
    pts = [T[:3, 3].copy()]
    for i in range(6):
        T = T @ _dh(thetas[i], _A[i], _ALPHA[i], _D[i])
        pts.append(T[:3, 3].copy())
    return T[:3, 3].copy(), np.array(pts)


def min_link_distance(joint_pos, obs):
    dmin = 1e9
    for i in range(len(joint_pos) - 1):
        a, b = joint_pos[i], joint_pos[i + 1]
        ab = b - a
        t = np.clip(np.dot(obs - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
        dmin = min(dmin, np.linalg.norm(obs - (a + t * ab)))
    return dmin


def dls_baseline(target, iters=300, damping=0.05, step=0.5, seed=0):
    """IK numerique (DLS) dans la MEME convention que le PINN (ignore l'obstacle)."""
    rng = np.random.default_rng(seed)
    theta = 0.1 * rng.standard_normal(6)
    for _ in range(iters):
        ee, _ = fk(theta)
        err = target - ee
        J = np.zeros((3, 6))
        eps = 1e-6
        for i in range(6):
            t = theta.copy()
            t[i] += eps
            eei, _ = fk(t)
            J[:, i] = (eei - ee) / eps
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), err)
        theta = theta + step * dq
    return theta


def report(tag, q, target_r, obs_r):
    ee, jp = fk(np.asarray(q))
    err = np.linalg.norm(ee - target_r) * 1000
    d = min_link_distance(jp, obs_r)
    hit = "COLLISION" if d < OBS_RADIUS else "OK (evite)"
    print(f"[{tag:5s}] erreur cible {err:6.1f} mm | dist. min obstacle {d*100:5.1f} cm -> {hit}")
    return d < OBS_RADIUS


# ------------------------------------------------------------------------- main
def main():
    from controller import Supervisor
    robot = Supervisor()
    ts = int(robot.getBasicTimeStep())
    dt = ts / 1000.0

    motors = [robot.getDevice(n) for n in MOTOR_NAMES]
    sensors = [robot.getDevice(n + "_sensor") for n in MOTOR_NAMES]
    for s in sensors:
        s.enable(ts)

    box = robot.getFromDef("RED_BOX").getField("translation")
    obs_node = robot.getFromDef("OBSTACLE")
    if obs_node is None:
        print("ERREUR: DEF OBSTACLE introuvable dans le monde.")
        return
    obs_field = obs_node.getField("translation")
    tool_node = robot.getFromDef("TOOL")   # PandaHand -> vraie position du flange

    pinn = NumpyPINN(os.path.join(HERE, "pinn_ik_ur5e.npz"))

    try:
        from ur5e_trajectory import QuinticTrajectory
    except Exception:
        QuinticTrajectory = None

    def world_to_robot(p):
        # socle Webots tourne de 180 deg autour de z -> on inverse x et y
        return np.array([-p[0], -p[1], p[2] - ROBOT_BASE_Z])

    def goto(q_target, duration=3.0):
        """Mouvement lisse (quintique si dispo, sinon direct) puis attente."""
        q0 = np.array([s.getValue() for s in sensors])
        q_target = np.asarray(q_target, dtype=float)
        if QuinticTrajectory is not None:
            traj = QuinticTrajectory(q0, q_target, duration)
            t = 0.0
            while robot.step(ts) != -1 and t < duration:
                t += dt
                q = traj.get_position(t)
                for i in range(6):
                    motors[i].setPosition(float(q[i]))
        else:
            for i in range(6):
                motors[i].setPosition(float(q_target[i]))
            for _ in range(int(duration / dt)):
                if robot.step(ts) == -1:
                    break

    def real_ee_error_mm(target_world):
        """Vraie position du flange dans Webots vs cible (verite terrain du simulateur)."""
        if tool_node is None:
            return None
        p = np.array(tool_node.getPosition())
        return np.linalg.norm(p - np.array(target_world)) * 1000.0

    def hold(sec):
        for _ in range(int(sec / dt)):
            if robot.step(ts) == -1:
                return False
        return True

    def read_detection():
        try:
            return json.load(open(SHARED))
        except Exception:
            return None

    print("=== PIPELINE VISION + PINN IK sous contraintes (Webots) ===")
    print("Phase 1 (VISION) : la camera reconnait le cube ; Phase 2 (PINN) : atteindre en evitant.")
    print("Astuce: deplacez le cube ROUGE ou la sphere ORANGE a la souris.\n")

    # Attendre que la vision ait fini sa calibration
    while robot.step(ts) != -1:
        det = read_detection()
        if det and det.get("calibrated"):
            break
        print("[bras] attente de la calibration vision...")
        if not hold(0.5):
            return

    while robot.step(ts) != -1:
        det = read_detection()
        if not det or not det.get("ok"):
            if not hold(0.3):
                return
            continue

        # --- Phase reconnaissance : position du cube ESTIMEE PAR LA VISION ---
        cube_w = [det["x"], det["y"], CUBE_Z]
        true = box.getSFVec3f()
        err_vision = np.hypot(cube_w[0] - true[0], cube_w[1] - true[1]) * 1000
        print(f"[VISION] cube reconnu a ({cube_w[0]:.3f}, {cube_w[1]:.3f}) "
              f"| erreur vs reel {err_vision:.0f} mm")

        target_w = [cube_w[0], cube_w[1], CUBE_Z + APPROACH_H]  # point d'approche
        obs_w = obs_field.getSFVec3f()
        target_r = world_to_robot(target_w)
        obs_r = world_to_robot(obs_w)

        # --- PINN : doit EVITER ---
        q_pinn = pinn(target_r, obs_r)
        goto(q_pinn)
        if not hold(1.0):
            return
        report("PINN", q_pinn, target_r, obs_r)
        e = real_ee_error_mm(target_w)
        if e is not None:
            print(f"        -> Webots reel : flange a {e:.1f} mm de la cible")

        # --- DLS (meme convention, ignore l'obstacle) : doit COLLISIONNER ---
        q_dls = dls_baseline(target_r)
        goto(q_dls)
        if not hold(1.0):
            return
        report("DLS ", q_dls, target_r, obs_r)
        e = real_ee_error_mm(target_w)
        if e is not None:
            print(f"        -> Webots reel : flange a {e:.1f} mm de la cible")
        print("-" * 60)


if __name__ == "__main__":
    main()
