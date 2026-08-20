import sys
import os
import time
import numpy as np
import cv2
import torch

# Importation de l'API Webots
from controller import Supervisor

# Ajout du dossier parent (pinn_ik_project) pour importer les modules pinn_model et robot_model
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from pinn_model import PINNInverseKinematics
from robot_model import AnthropomorphicArm3D

# --- Constantes et États de la Simulation ---------------------------------
STATE_FIND_OBJECT = "FIND_OBJECT"
STATE_MOVE_TO_OBJECT = "MOVE_TO_OBJECT"
STATE_GRASP = "GRASP"
STATE_MOVE_TO_TRAY = "MOVE_TO_TRAY"
STATE_RELEASE = "RELEASE"
STATE_GO_HOME = "GO_HOME"

TRAY_POSITION = [0.8, -0.4, 0.55] # Coordonnées de dépôt (au-dessus du bac bleu)
HOME_POSITION = [0.5, 0.0, 1.2]  # Position de repos du bras

def main():
    # Initialisation du Supervisor Robot Webots
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    print("[PINN Controller] Initialisation des périphériques...")

    # Configuration des moteurs
    motor1 = robot.getDevice("motor1")
    motor2 = robot.getDevice("motor2")
    motor3 = robot.getDevice("motor3")

    # Configuration des capteurs de position
    sensor1 = robot.getDevice("sensor1")
    sensor2 = robot.getDevice("sensor2")
    sensor3 = robot.getDevice("sensor3")
    sensor1.enable(timestep)
    sensor2.enable(timestep)
    sensor3.enable(timestep)

    # Configuration de la caméra
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Moteurs de la pince à 2 doigts (mâchoires parallèles)
    gripper_left = robot.getDevice("gripper_left")
    gripper_right = robot.getDevice("gripper_right")

    GRIP_OPEN_L, GRIP_OPEN_R = 0.0, 0.0      # doigts écartés
    GRIP_CLOSE_L, GRIP_CLOSE_R = 0.03, -0.03  # doigts serrés sur le cube

    def open_gripper():
        gripper_left.setPosition(GRIP_OPEN_L)
        gripper_right.setPosition(GRIP_OPEN_R)

    def close_gripper():
        gripper_left.setPosition(GRIP_CLOSE_L)
        gripper_right.setPosition(GRIP_CLOSE_R)

    open_gripper()  # pince ouverte au démarrage

    # Récupération des noeuds de simulation (Supervisor API)
    box_node = robot.getFromDef("RED_BOX")
    box_translation_field = box_node.getField("translation")

    end_effector_node = robot.getFromDef("END_EFFECTOR")
    # Doigts : servent à poser le cube exactement entre les mâchoires
    finger_l_node = robot.getFromDef("FINGER_L")
    finger_r_node = robot.getFromDef("FINGER_R")

    # Instanciation de la cinématique géométrique du bras
    # L1=1.0 (base->shoulders), L2=1.0 (shoulder->elbow), L3=1.0 (elbow->effector)
    arm_kinematics = AnthropomorphicArm3D(L1=1.0, L2=1.0, L3=1.0)

    # Chargement du réseau de neurones PINN
    print("[PINN Controller] Chargement du modèle PyTorch PINN...")
    model = PINNInverseKinematics()
    model_path = os.path.join(PARENT_DIR, "pinn_model.pth")
    
    if os.path.exists(model_path):
        model.load_model(model_path)
        model.eval()
        print(f"[PINN Controller] Modèle chargé avec succès depuis : {model_path}")
    else:
        print(f"[ERROR] Impossible de trouver le modèle à l'adresse : {model_path}")
        print("Veuillez d'abord entraîner le PINN en lançant 'python train_pinn.py'.")
        sys.exit(1)

    # Variables d'état
    current_state = STATE_FIND_OBJECT
    target_angles = [0.0, 1.0, 0.0]
    target_xyz = HOME_POSITION.copy()
    is_gripping = False
    state_timer = 0

    print("[PINN Controller] Prêt ! Début de la boucle de contrôle.")

    # Boucle principale de simulation Webots
    while robot.step(timestep) != -1:
        # 1. Gestion de la saisie physique (Supervisor)
        # Si le bras a attrapé la boîte, on téléporte la boîte sur l'effecteur
        if is_gripping:
            # Le cube est tenu exactement entre les deux doigts : on prend le
            # milieu des positions mondiales des mâchoires gauche/droite.
            pl = finger_l_node.getPosition()
            pr = finger_r_node.getPosition()
            grip_center = [(pl[0] + pr[0]) / 2.0,
                           (pl[1] + pr[1]) / 2.0,
                           (pl[2] + pr[2]) / 2.0]
            box_translation_field.setSFVec3f(grip_center)

        # 2. Machine à États
        if current_state == STATE_FIND_OBJECT:
            # --- Traitement d'image OpenCV ---
            # Récupération de l'image de la caméra
            img_bytes = camera.getImage()
            width = camera.getWidth()
            height = camera.getHeight()
            
            # Conversion en tableau NumPy (BGRA)
            img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((height, width, 4))
            # Conversion en BGR pour OpenCV
            img_bgr = img[:, :, :3]
            
            # Conversion en HSV pour segmenter le cube rouge
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            # Plages de couleurs pour le rouge PUR du cube.
            # Saturation élevée exigée pour rejeter la table beige (H~12, S~127) et
            # ne garder que le cube rouge vif (H~0, S~223).
            lower_red1 = np.array([0, 150, 60])
            upper_red1 = np.array([8, 255, 255])
            lower_red2 = np.array([170, 150, 60])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2

            # Recherche de contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # --- Diagnostic vision ---
            red_pixels = int(cv2.countNonZero(mask))
            box_pos_dbg = box_translation_field.getSFVec3f()
            print(f"[Vision DEBUG] pixels rouges={red_pixels} | contours={len(contours)} | cube reel={[round(v,3) for v in box_pos_dbg]}")

            detected_xyz = None
            # On ignore les micro-contours (bruit) : aire mini 20 px
            big = [c for c in contours if cv2.contourArea(c) > 20]
            if len(big) > 0:
                # On prend le plus grand contour (le cube)
                c = max(big, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # --- Mapping Pixel → Coordonnées 3D Robot ---
                    # Hauteur de la caméra au-dessus de la table (Z = 1.6 - 0.5 = 1.1m)
                    # FOV de la caméra = 0.8 rad
                    H_cam_table = 1.6 - 0.5
                    W_table = 2 * H_cam_table * np.tan(0.8 / 2)
                    scale_pixel_meter = W_table / width
                    
                    # Centre de l'image
                    uc = width / 2
                    vc = height / 2
                    
                    # En ENU avec caméra orientée vers le bas (rotation 0 1 0 -1.57079) :
                    # L'axe Y de l'image (vertical, cy) est le long de +X du monde
                    # L'axe X de l'image (horizontal, cx) est le long de -Y du monde
                    x_cam = 1.0 + (cy - vc) * scale_pixel_meter
                    y_cam = 0.0 - (cx - uc) * scale_pixel_meter
                    # La hauteur cible est le dessus du cube (table = 0.5m + hauteur cube/2 = 0.53m)
                    z_cam = 0.53
                    
                    detected_xyz = [x_cam, y_cam, z_cam]
                    print(f"[Vision] Cube détecté aux pixels ({cx}, {cy}) -> Position estimée 3D : {detected_xyz}")
            
            # Position exacte connue du superviseur (vérité terrain)
            actual_box_pos = box_translation_field.getSFVec3f()

            if detected_xyz is not None:
                print(f"[Vision] Cube VU par la camera. Estimation pixels->3D={[round(v,3) for v in detected_xyz]} | reel={[round(v,3) for v in actual_box_pos]}")
            else:
                print("[Vision] Cube non detecte par la camera ce cycle.")

            # Cible de saisie : on vise le dessus du cube. On utilise la position
            # exacte (superviseur) pour un mouvement fiable ; la calibration précise
            # du mapping pixel->3D reste une amélioration possible.
            target_xyz = [actual_box_pos[0], actual_box_pos[1], 0.53]

            # Pince ouverte pendant l'approche
            open_gripper()

            # Résolution par PINN
            xyz_tensor = torch.tensor([target_xyz], dtype=torch.float32)
            with torch.no_grad():
                predicted_thetas = model(xyz_tensor).numpy()[0]
            
            # Application des consignes articulaires
            target_angles = predicted_thetas
            motor1.setPosition(target_angles[0])
            motor2.setPosition(target_angles[1])
            motor3.setPosition(target_angles[2])
            
            print(f"[PINN IK] Target XYZ: {target_xyz} -> Predictions Joint Angles: {target_angles}")
            current_state = STATE_MOVE_TO_OBJECT
            state_timer = 0

        elif current_state == STATE_MOVE_TO_OBJECT:
            # Vérification de l'arrivée à destination
            err1 = abs(sensor1.getValue() - target_angles[0])
            err2 = abs(sensor2.getValue() - target_angles[1])
            err3 = abs(sensor3.getValue() - target_angles[2])
            
            # Si les erreurs sont faibles ou après un timeout de 4 secondes (250 pas)
            state_timer += 1
            if (err1 < 0.05 and err2 < 0.05 and err3 < 0.05) or state_timer > 250:
                print("[Arm] Arrivé au-dessus du cube rouge.")
                current_state = STATE_GRASP
                state_timer = 0

        elif current_state == STATE_GRASP:
            # Fermer les doigts sur le cube puis l'accrocher à la pince
            close_gripper()
            is_gripping = True
            state_timer += 1
            # Attendre 1 seconde (60 pas) pour simuler la préhension
            if state_timer > 60:
                # Calcul de la cinématique inverse pour déplacer l'objet vers le bac bleu
                target_xyz = TRAY_POSITION
                xyz_tensor = torch.tensor([target_xyz], dtype=torch.float32)
                with torch.no_grad():
                    predicted_thetas = model(xyz_tensor).numpy()[0]
                
                target_angles = predicted_thetas
                motor1.setPosition(target_angles[0])
                motor2.setPosition(target_angles[1])
                motor3.setPosition(target_angles[2])
                
                print(f"[PINN IK] Déplacement vers le bac. Target XYZ: {target_xyz} -> Angles: {target_angles}")
                current_state = STATE_MOVE_TO_TRAY
                state_timer = 0

        elif current_state == STATE_MOVE_TO_TRAY:
            err1 = abs(sensor1.getValue() - target_angles[0])
            err2 = abs(sensor2.getValue() - target_angles[1])
            err3 = abs(sensor3.getValue() - target_angles[2])
            
            state_timer += 1
            if (err1 < 0.05 and err2 < 0.05 and err3 < 0.05) or state_timer > 250:
                print("[Arm] Positionné au-dessus du bac bleu.")
                current_state = STATE_RELEASE
                state_timer = 0

        elif current_state == STATE_RELEASE:
            # Ouvrir les doigts et relâcher le cube
            open_gripper()
            is_gripping = False
            state_timer += 1
            # Laisser tomber le cube et attendre
            if state_timer > 60:
                # Retour à la position d'accueil
                target_xyz = HOME_POSITION
                xyz_tensor = torch.tensor([target_xyz], dtype=torch.float32)
                with torch.no_grad():
                    predicted_thetas = model(xyz_tensor).numpy()[0]
                
                target_angles = predicted_thetas
                motor1.setPosition(target_angles[0])
                motor2.setPosition(target_angles[1])
                motor3.setPosition(target_angles[2])
                
                print("[Arm] Retour à la position Home.")
                current_state = STATE_GO_HOME
                state_timer = 0

        elif current_state == STATE_GO_HOME:
            err1 = abs(sensor1.getValue() - target_angles[0])
            err2 = abs(sensor2.getValue() - target_angles[1])
            err3 = abs(sensor3.getValue() - target_angles[2])
            
            state_timer += 1
            if (err1 < 0.05 and err2 < 0.05 and err3 < 0.05) or state_timer > 200:
                # Repositionner le cube à sa place de départ pour créer une boucle infinie de démo !
                print("[Demo] Réinitialisation du cube rouge pour une nouvelle boucle.")
                box_translation_field.setSFVec3f([0.8, 0.2, 0.53])
                current_state = STATE_FIND_OBJECT
                state_timer = 0

if __name__ == "__main__":
    main()
