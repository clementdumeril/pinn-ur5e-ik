"""
Controleur UR5e Webots basé sur la bibliothèque Python IKPY (Standard Industrie).
Utilise IKPY pour le calcul de Cinématique Inverse et QuinticTrajectory pour la fluidité.
"""
import sys
import os
import numpy as np

from controller import Supervisor

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from ikpy_ur5e_solver import IKPYUR5eSolver
from ur5e_trajectory import QuinticTrajectory

# --- Machine a etats ---
STATE_INIT = "INIT"
STATE_FIND_OBJECT = "FIND_OBJECT"
STATE_APPROACH = "APPROACH"
STATE_GRASP = "GRASP"
STATE_LIFT = "LIFT"
STATE_MOVE_TO_TRAY = "MOVE_TO_TRAY"
STATE_RELEASE = "RELEASE"
STATE_GO_HOME = "GO_HOME"

# --- Configuration Ergonomique du Poste de Travail (Repere Monde Webots) ---
ROBOT_BASE_Z = 0.45      # Socle robot a z=0.45m

# Repere Monde (Table top a z=0.30m, Cube a z=0.32m)
GRASP_Z_WORLD = 0.305    # Hauteur d'enrobage exacte du cube (sans frotter la table)
LIFT_Z_WORLD  = 0.550    # Levage vertical a 55 cm du sol (25 cm au-dessus de la table)
TRAY_WORLD    = [0.50, -0.10, 0.40]  # Au-dessus du bac bleu
HOME_WORLD    = [0.35,  0.00, 0.55]  # Position de repos haute


def world_to_robot(world_pos):
    """Convertit position Monde [x, y, z] en position relative a la Base Robot."""
    return [world_pos[0], world_pos[1], world_pos[2] - ROBOT_BASE_Z]


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0   # Pas de temps en secondes

    print("\n=================================================================")
    print("🤖 UR5e Controller — Python IKPY Standard Library Kinematics")
    print("=================================================================\n")

    # Connecter les 6 moteurs du UR5e
    motor_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]
    motors = [robot.getDevice(name) for name in motor_names]
    sensors = [robot.getDevice(name + "_sensor") for name in motor_names]
    for s in sensors:
        s.enable(timestep)

    # Pince (moteurs dynamiques avec force 50 N)
    gripper_left = robot.getDevice("gripper_left")
    gripper_right = robot.getDevice("gripper_right")

    def open_gripper():
        gripper_left.setPosition(0.0)
        gripper_right.setPosition(0.0)

    def close_gripper():
        gripper_left.setPosition(0.02)
        gripper_right.setPosition(-0.02)

    open_gripper()

    # Superviseur pour lire la position du cube
    box_node = robot.getFromDef("RED_BOX")
    box_trans_field = box_node.getField("translation")

    # Solveur IKPY
    urdf_file = os.path.join(PARENT_DIR, "ur5e.urdf")
    ikpy_solver = IKPYUR5eSolver(urdf_file)

    # Position initiale
    q_home, _, _ = ikpy_solver.solve_ik(world_to_robot(HOME_WORLD))
    current_q = np.array(q_home)
    target_q = np.array(q_home)

    for i in range(6):
        motors[i].setPosition(float(current_q[i]))

    # Machine a etats & Trajectoire
    current_state = STATE_INIT
    active_trajectory = None
    traj_time = 0.0
    state_timer = 0

    def start_trajectory(q_target_new, duration):
        nonlocal active_trajectory, traj_time, target_q, current_q
        for i in range(6):
            current_q[i] = sensors[i].getValue()
        target_q = np.array(q_target_new)
        active_trajectory = QuinticTrajectory(current_q, target_q, duration)
        traj_time = 0.0

    print("✓ Initialisation IKPY terminee. Demarrage du Pick and Place...")

    # ==================== Boucle de Controle ====================
    while robot.step(timestep) != -1:
        state_timer += 1

        # Avancement du temps de la trajectoire quintique
        if active_trajectory is not None:
            traj_time += dt
            q_step = active_trajectory.get_position(traj_time)
            for i in range(6):
                motors[i].setPosition(float(q_step[i]))

        if current_state == STATE_INIT:
            if state_timer > 60:
                current_state = STATE_FIND_OBJECT
                state_timer = 0

        elif current_state == STATE_FIND_OBJECT:
            # 1. Position du cube dans le monde
            box_pos = box_trans_field.getSFVec3f()
            print(f"[Vision] Cube localise -> X={box_pos[0]:.3f}, Y={box_pos[1]:.3f}, Z={box_pos[2]:.3f}")

            # 2. Cible d'approche (doigts entourent le cube)
            target_world = [box_pos[0], box_pos[1], GRASP_Z_WORLD]
            target_robot_xyz = world_to_robot(target_world)

            # 3. Resolution IKPY
            q_approach, _, err_mm = ikpy_solver.solve_ik(target_robot_xyz)
            print(f"[IKPY Solver] Approche Cube -> Erreur Math: {err_mm:.4f} mm")

            open_gripper()
            start_trajectory(q_approach, duration=2.5)

            current_state = STATE_APPROACH
            state_timer = 0

        elif current_state == STATE_APPROACH:
            if traj_time >= active_trajectory.T:
                print("[Pince] Position d'approche atteinte. Serrage physique du cube (50N)...")
                current_state = STATE_GRASP
                state_timer = 0

        elif current_state == STATE_GRASP:
            close_gripper()
            if state_timer > 65:
                # 4. Levage vertical du cube
                box_pos = box_trans_field.getSFVec3f()
                target_world = [box_pos[0], box_pos[1], LIFT_Z_WORLD]
                target_robot_xyz = world_to_robot(target_world)
                q_lift, _, _ = ikpy_solver.solve_ik(target_robot_xyz)

                print(f"[Pince] Cube enserre ! Levage vertical IKPY a Z={LIFT_Z_WORLD}m...")
                start_trajectory(q_lift, duration=2.0)

                current_state = STATE_LIFT
                state_timer = 0

        elif current_state == STATE_LIFT:
            if traj_time >= active_trajectory.T:
                # 5. Transbordement vers le bac bleu
                target_robot_xyz = world_to_robot(TRAY_WORLD)
                q_tray, _, _ = ikpy_solver.solve_ik(target_robot_xyz)

                print(f"[Transport] Deplacement vers le bac bleu {TRAY_WORLD}...")
                start_trajectory(q_tray, duration=2.5)

                current_state = STATE_MOVE_TO_TRAY
                state_timer = 0

        elif current_state == STATE_MOVE_TO_TRAY:
            if traj_time >= active_trajectory.T:
                print("[Depot] Au-dessus du bac. Relâchement du cube...")
                current_state = STATE_RELEASE
                state_timer = 0

        elif current_state == STATE_RELEASE:
            open_gripper()
            if state_timer > 50:
                # 6. Retour a la position Home
                target_robot_xyz = world_to_robot(HOME_WORLD)
                q_home_new, _, _ = ikpy_solver.solve_ik(target_robot_xyz)

                print("[Home] Retour lissé a la position de repos.")
                start_trajectory(q_home_new, duration=2.5)

                current_state = STATE_GO_HOME
                state_timer = 0

        elif current_state == STATE_GO_HOME:
            if traj_time >= active_trajectory.T:
                print("\n[Loop] Reinitialisation du cube pour la prochaine boucle IKPY.")
                box_trans_field.setSFVec3f([0.50, 0.10, 0.32])
                current_state = STATE_FIND_OBJECT
                state_timer = 0


if __name__ == "__main__":
    main()
