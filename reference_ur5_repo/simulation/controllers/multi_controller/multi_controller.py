"""
Controleur Comparatif Multi-Robots Webots (Pure ASCII).
Pilote simultanement les 3 robots UR5e avec leurs 3 technologies respectives :
- Robot A : Option A (Trigo DH Analytique)
- Robot B : Option B (Algebre de Lie SE3 Produit d'Exponentielles)
- Robot C : Option C (Reseau de Neurones PINN AI)
"""
import os
import sys
import numpy as np
from math import pi

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PARENT_DIR = os.path.abspath(os.path.join(REPO_DIR, '..'))

for p in [REPO_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ur5 import UR5, forward_kinematics, inverse_kinematics, build_matrix
from ur5e_se3_ik import UR5eSE3

PI = pi

# Rendre importable l'architecture du reseau (training/) et les utilitaires
for extra in [os.path.join(PARENT_DIR, "training"),
              os.path.join(PARENT_DIR, "robotics_utils")]:
    if extra not in sys.path:
        sys.path.insert(0, extra)

# Charger le modele PINN 6-DOF.
# On cherche en priorite le modele entraine sur la cinematique DH COMPLETE
# (pinn_model_true_physics.pth) : c'est la seule convention compatible avec
# forward_kinematics() de ur5.py, qui sert ici a mesurer l'erreur cartesienne.
# pinn_model_ur5e_6dof.pth a ete entraine sur la FK planaire simplifiee de
# ur5e_6dof_ik.py et donne des positions fausses s'il est evalue avec le DH complet.
PINN_CANDIDATES = [
    os.path.join(PARENT_DIR, "models", "pinn_model_true_physics.pth"),
    os.path.join(PARENT_DIR, "models", "pinn_model_webots.pth"),
    os.path.join(PARENT_DIR, "models", "archive_models", "pinn_model_ur5e_6dof.pth"),
    os.path.join(PARENT_DIR, "pinn_model_ur5e_6dof.pth"),
    os.path.join(REPO_DIR, "pinn_model_ur5e_6dof.pth"),
]

PINN_PATH = next((p for p in PINN_CANDIDATES if os.path.exists(p)), None)

pinn_model = None

if PINN_PATH is not None:
    try:
        import torch
        from train_pinn_6dof import PINN6DOF
        pinn_model = PINN6DOF.from_file(PINN_PATH)
        pinn_model.eval()
        print(f"[PINN] Modele charge pour le Robot C : {os.path.basename(PINN_PATH)}")
    except Exception as e:
        print(f"[PINN Info] Chargement impossible ({e}) -> repli sur l'IK analytique.")
else:
    print("[PINN Info] Aucun modele .pth trouve -> le Robot C utilisera l'IK analytique.")
    print(f"           Cherche dans : {PINN_CANDIDATES[0]}")


def solve_ik_technology(robot_name, target_pos, ur5_instance):
    """
    Resout la cinematique inverse selon la technologie attribuee au robot.
    """
    T_mat = build_matrix(target_pos, [PI, 0, PI/2])

    if "OptionA" in robot_name:
        # Option A : Trigo DH Analytique
        q_sol = inverse_kinematics(T_mat, wrist='up')
        tech_name = "Option A (Trigo DH)"
    elif "OptionB" in robot_name:
        # Option B : Lie Algebra SE3 Exponentielles
        se3_solver = UR5eSE3()
        q_sol, _, _ = se3_solver.solve_ik_se3(target_pos, ur5_instance)
        tech_name = "Option B (Lie SE3 PoE)"
    else:
        # Option C : Reseau de Neurones PINN AI
        tech_name = "Option C (PINN AI)"
        if pinn_model is not None:
            try:
                import torch
                with torch.no_grad():
                    inp = torch.tensor(target_pos, dtype=torch.float32).unsqueeze(0)
                    q_sol = pinn_model(inp).numpy()[0]
            except Exception:
                q_sol = inverse_kinematics(T_mat, wrist='up')
        else:
            q_sol = inverse_kinematics(T_mat, wrist='up')

    # Calcul de la position reelle pour l'erreur cartesienne
    fk_res, _ = forward_kinematics(q_sol)
    err_mm = np.linalg.norm(fk_res[:3, 3] - target_pos) * 1000.0

    return q_sol, tech_name, err_mm


def main():
    ur5 = UR5()
    robot_name = ur5.supervisor.getName()

    print("\n=================================================================")
    print(f"DEMARRAGE DU ROBOT : {robot_name}")
    print("=================================================================\n")

    # Activer la camera si presente
    if hasattr(ur5, "camera") and ur5.camera is not None:
        try:
            ur5.camera.enable(ur5.timestep)
            print(f"[{robot_name}] Flux Camera active !")
        except Exception:
            pass

    # Determiner les noms d'objets associes a ce poste
    if "OptionA" in robot_name:
        box_def, tray_def = "RED_BOX_A", "BLUE_TRAY_A"
    elif "OptionB" in robot_name:
        box_def, tray_def = "RED_BOX_B", "BLUE_TRAY_B"
    else:
        box_def, tray_def = "RED_BOX_C", "BLUE_TRAY_C"

    # Posture de depart face a la table
    q_home = [0, -PI/3, PI/2, -PI/6, -PI/2, 0]
    ur5.move_to_config(q_home)

    # 1. Localisation du Cube et du Bac dans le repere de ce robot
    cube_frame = ur5.get_node_frame(box_def)
    tray_frame = ur5.get_node_frame(tray_def)

    if cube_frame is not None:
        cx, cy = cube_frame[0, 3], cube_frame[1, 3]
    else:
        cx, cy = 0.73, 0.25

    if tray_frame is not None:
        tx, ty = tray_frame[0, 3], tray_frame[1, 3]
    else:
        tx, ty = 0.73, -0.25

    # 2. Saisie et Depose synchronisees
    print(f"[{robot_name}] Approche au-dessus du cube...")
    q_app, tech, err_app = solve_ik_technology(robot_name, [cx, cy, 0.35], ur5)
    print(f"  -> [{tech}] Approche Cube | Erreur Cartesienne: {err_app:.2f} mm")
    ur5.move_to_config(q_app)

    print(f"[{robot_name}] Descente sur le cube (Z=0.05m)...")
    q_grasp, _, err_grasp = solve_ik_technology(robot_name, [cx, cy, 0.05], ur5)
    print(f"  -> [{tech}] Prise Cube | Erreur Cartesienne: {err_grasp:.2f} mm")
    ur5.move_to_config(q_grasp)

    print(f"[{robot_name}] Serrage de la pince...")
    ur5.actuate_gripper(1)

    print(f"[{robot_name}] Levage vertical...")
    ur5.move_to_config(q_app)

    print(f"[{robot_name}] Transport vers le bac...")
    q_tray, _, err_tray = solve_ik_technology(robot_name, [tx, ty, 0.35], ur5)
    ur5.move_to_config(q_tray)

    print(f"[{robot_name}] Descente dans le bac (Z=0.08m)...")
    q_drop, _, err_drop = solve_ik_technology(robot_name, [tx, ty, 0.08], ur5)
    print(f"  -> [{tech}] Depose Bac | Erreur Cartesienne: {err_drop:.2f} mm")
    ur5.move_to_config(q_drop)

    print(f"[{robot_name}] Ouverture pince...")
    ur5.actuate_gripper(0)

    print(f"[{robot_name}] Retour a la position Home...")
    ur5.move_to_config(q_tray)
    ur5.move_to_config(q_home)

    print(f"\n[{robot_name}] SEQUENCE COMPLETEE AVEC SUCCES !\n")


if __name__ == "__main__":
    main()
