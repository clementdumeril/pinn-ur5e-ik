"""
Calibration de la camera et generation du dataset -- approche directe.

Principe
--------
La camera est a une pose FIXE. On regarde ce qu'elle voit, et on ne fait
apparaitre le cube que la.

  Etape 1  Le bras va a POSE_CAMERA et on attend qu'il soit reellement
           immobile (la derive est mesuree, pas supposee).
  Etape 2  Balayage : le cube est place sur une grille couvrant la table, et
           on note ou il est effectivement detecte. Aucun modele de projection,
           aucune convention d'axes a deviner -- on observe.
  Etape 3  Homographie ajustee sur ces correspondances mesurees, et zone
           visible deduite des detections reussies.
  Etape 4  Collecte optionnelle du dataset, dans la zone visible uniquement.

Pourquoi cette reecriture
-------------------------
La version precedente essayait de modeliser la camera analytiquement pour
predire, sans rendu, ce que verrait chaque orientation. Elle n'a jamais reussi
a identifier la convention d'axes de maniere fiable (595 px puis 817 px
d'erreur), ce qui la faisait basculer dans un repli ou le critere de verticalite
etait ignore. Mesurer coute quelques minutes de simulation et ne suppose rien.

Sorties (dans reference_ur5_repo/dataset/)
------------------------------------------
  calibration.json    pose de lecture, homographie pixel -> monde, zone visible,
                      transformation outil -> camera
  labels.csv          filename, x_pixel, y_pixel   (repere 256x256)
  images/*.jpg        seulement si COLLECTER_IMAGES = True (pour le VGG16)
"""
import os
import sys
import csv
import json
import glob
import random
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from ur5 import UR5, build_matrix

PI = np.pi

# --- Monde ---------------------------------------------------------------
TABLE_CENTER = (-1.5, 0.0)          # table1, repere Webots
TABLE_SIZE = (0.8, 1.4)             # x, y
CUBE_Z = 0.955                      # hauteur du centre du cube pose sur la table

# --- La pose de lecture, fixe --------------------------------------------
# Repere robot. Le bras est deja a 0.822 m de sa base pour une portee de
# ~0.85 m : on ne peut pas l'eloigner davantage de la table, seulement
# reorienter le poignet. Modifier ces valeurs et relancer suffit a changer le
# point de vue -- tout le reste (zone visible, homographie) est remesure.
POSE_CAMERA = [-0.10, -0.68, 0.45]
ROT_CAMERA = [PI, 0.30, -PI / 2]

# --- Reglages ------------------------------------------------------------
GRILLE = 11                         # balayage GRILLE x GRILLE sur la table
MIN_RED_PIXELS = 20                 # seuil de detection du cube
MARGE = 0.03                        # marge (m) retiree de la zone visible
IMG_SIZE = 256                      # taille enregistree = entree du reseau
STABILISATION = 1.5                 # s de temps simule apres le deplacement

COLLECTER_IMAGES = False            # True uniquement pour reentrainer le VGG16
NB_IMAGES = 1000


def bornes_table():
    cx, cy = TABLE_CENTER
    sx, sy = TABLE_SIZE
    return (cx - sx / 2, cx + sx / 2), (cy - sy / 2, cy + sy / 2)


def noeud_camera(ur5):
    """
    Noeud Webots de la camera.

    getFromDevice() attend le TAG NUMERIQUE, pas l'objet Device (sinon ctypes
    leve "Don't know how to convert parameter 1"). Selon la version de l'API il
    s'obtient par getTag() ou par l'attribut _tag.
    """
    cam = ur5.camera
    if cam is None:
        return None
    for get_tag in (lambda: cam.getTag(), lambda: cam._tag):
        try:
            tag = get_tag()
        except Exception:
            continue
        if isinstance(tag, int):
            try:
                n = ur5.supervisor.getFromDevice(tag)
            except Exception:
                continue
            if n is not None:
                return n
    return None


def pose_noeud(node):
    T = np.eye(4)
    T[:3, :3] = np.array(node.getOrientation()).reshape(3, 3)
    T[:3, 3] = np.array(node.getPosition())
    return T


def detecter_cube(img_rgb):
    """Centroide des pixels rouges, dans le repere IMG_SIZE x IMG_SIZE."""
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    r, g, b = (img[:, :, i].astype(int) for i in range(3))
    mask = (r > 110) & (r - g > 60) & (r - b > 60)
    if int(mask.sum()) < MIN_RED_PIXELS:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def poser_cube(ur5, field, node, x, y):
    """
    Teleporte le cube.

    resetPhysics() est indispensable : sans lui le cube garde sa vitesse, rebondit
    et ne se trouve pas ou on croit -- ce qui fausse silencieusement toutes les
    correspondances.
    """
    field.setSFVec3f([float(x), float(y), CUBE_Z])
    node.resetPhysics()
    for _ in range(4):
        ur5.supervisor.step(ur5.timestep)


def attendre_immobilite(ur5, cam_node, duree=STABILISATION):
    """Attend, puis verifie que la camera ne bouge REELLEMENT plus."""
    t0 = ur5.supervisor.getTime()
    while ur5.supervisor.getTime() - t0 < duree:
        ur5.supervisor.step(ur5.timestep)
    for essai in range(6):
        p1 = np.array(cam_node.getPosition())
        for _ in range(25):
            ur5.supervisor.step(ur5.timestep)
        derive = np.linalg.norm(np.array(cam_node.getPosition()) - p1) * 1000
        if derive < 0.5:
            print(f"  camera immobile (derive {derive:.2f} mm)")
            return True
        print(f"  camera encore en mouvement ({derive:.1f} mm), attente...")
    print("  ATTENTION : la camera ne se stabilise pas.")
    return False


def main():
    random.seed(0)
    np.random.seed(0)

    ur5 = UR5()
    ur5.use_pinn = False             # IK analytique exacte pour la calibration
    ur5.setup_camera()

    box = ur5.bottle
    if box is None:
        print("ERREUR : aucun noeud DEF RED_BOX / bottle dans ce monde.")
        return
    field = box.getField("translation")
    depart = field.getSFVec3f()

    cam_node = noeud_camera(ur5)
    if cam_node is None:
        print("ERREUR : noeud de la camera inaccessible (getFromDevice).")
        return

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    dataset_dir = os.path.join(base, 'dataset')
    img_dir = os.path.join(dataset_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # ETAPE 1 : le bras se place, et on attend qu'il soit immobile
    # ---------------------------------------------------------------
    print()
    print("=== ETAPE 1 : mise en place de la camera ===")
    print(f"  pose demandee : {POSE_CAMERA}  rot {[round(r, 2) for r in ROT_CAMERA]}")
    ur5.move_to_pose(list(POSE_CAMERA), list(ROT_CAMERA), wrist='up')
    attendre_immobilite(ur5, cam_node)

    T_base = pose_noeud(ur5.supervisor.getSelf())
    T_cam = pose_noeud(cam_node)
    T_cmd = T_base @ build_matrix(POSE_CAMERA, ROT_CAMERA, euler='XYZ')
    T_cmd_cam = np.linalg.inv(T_cmd) @ T_cam

    print(f"  camera (monde)     : {np.round(T_cam[:3, 3], 3)}")
    print(f"  pose commandee     : {np.round(T_cmd[:3, 3], 3)}")
    ecart = np.linalg.norm(T_cam[:3, 3] - T_cmd[:3, 3]) * 1000
    print(f"  decalage outil->camera : {np.round(T_cmd_cam[:3, 3], 4)} "
          f"({ecart:.0f} mm)")

    # ---------------------------------------------------------------
    # ETAPE 2 : que voit-elle ? on regarde, on ne calcule pas
    # ---------------------------------------------------------------
    print()
    print(f"=== ETAPE 2 : balayage {GRILLE}x{GRILLE} de la table ===")
    (x0, x1), (y0, y1) = bornes_table()
    src, dst = [], []
    for x in np.linspace(x0 + 0.02, x1 - 0.02, GRILLE):
        for y in np.linspace(y0 + 0.02, y1 - 0.02, GRILLE):
            poser_cube(ur5, field, box, x, y)
            uv = detecter_cube(ur5.get_image())
            if uv is not None:
                src.append(uv)
                dst.append((x, y))

    total = GRILLE * GRILLE
    print(f"  cube visible sur {len(src)}/{total} positions "
          f"({100 * len(src) / total:.0f} % de la table)")
    if len(src) < 12:
        print("  TROP PEU de points visibles pour calibrer.")
        print("  -> modifier POSE_CAMERA / ROT_CAMERA en tete de ce fichier.")
        field.setSFVec3f(depart)
        box.resetPhysics()
        return

    # ---------------------------------------------------------------
    # ETAPE 3 : homographie et zone visible, deduites des mesures
    # ---------------------------------------------------------------
    print()
    print("=== ETAPE 3 : homographie pixel -> monde ===")
    H, _ = cv2.findHomography(np.array(src, dtype=np.float64),
                              np.array(dst, dtype=np.float64), cv2.RANSAC, 0.005)
    if H is None:
        print("  Ajustement impossible.")
        return
    P = np.hstack([np.array(src), np.ones((len(src), 1))]) @ H.T
    P = P[:, :2] / P[:, 2:3]
    res = np.linalg.norm(P - np.array(dst), axis=1) * 1000
    print(f"  residus : moyenne {res.mean():.1f} mm, median {np.median(res):.1f} mm, "
          f"max {res.max():.1f} mm")

    a = np.array(dst)
    zone = {'x_min': float(a[:, 0].min()) + MARGE, 'x_max': float(a[:, 0].max()) - MARGE,
            'y_min': float(a[:, 1].min()) + MARGE, 'y_max': float(a[:, 1].max()) - MARGE}
    print(f"  zone visible : x [{zone['x_min']:.3f}, {zone['x_max']:.3f}]  "
          f"y [{zone['y_min']:.3f}, {zone['y_max']:.3f}]")

    json.dump({
        'reading_pose': {'position': list(POSE_CAMERA), 'orientation': list(ROT_CAMERA)},
        'image_size': IMG_SIZE,
        'camera_resolution': 512,
        'cube_z': CUBE_Z,
        'tool_to_camera': T_cmd_cam.tolist(),
        'pixel_to_world_homography': H.tolist(),
        'fit_rmse_m': float(np.sqrt((res ** 2).mean()) / 1000),
        'residu_median_mm': float(np.median(res)),
        'visible_zone_world': zone,
        'coverage': len(src) / total,
        'points_mesures': len(src),
    }, open(os.path.join(dataset_dir, 'calibration.json'), 'w'), indent=2)
    print("  calibration.json ecrit")

    # ---------------------------------------------------------------
    # ETAPE 4 : dataset, uniquement si on veut reentrainer le VGG16
    # ---------------------------------------------------------------
    if COLLECTER_IMAGES:
        print()
        print(f"=== ETAPE 4 : collecte de {NB_IMAGES} images ===")
        for f in glob.glob(os.path.join(img_dir, '*.jpg')):
            os.remove(f)
        garde = rejet = 0
        with open(os.path.join(dataset_dir, 'labels.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['filename', 'x_pixel', 'y_pixel'])
            while garde < NB_IMAGES:
                x = random.uniform(zone['x_min'], zone['x_max'])
                y = random.uniform(zone['y_min'], zone['y_max'])
                poser_cube(ur5, field, box, x, y)
                img = ur5.get_image()
                uv = detecter_cube(img)
                if uv is None:
                    rejet += 1
                    continue
                nom = f"img_{garde:04d}.jpg"
                cv2.imwrite(os.path.join(img_dir, nom),
                            cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                       (IMG_SIZE, IMG_SIZE)))
                w.writerow([nom, round(uv[0], 3), round(uv[1], 3)])
                garde += 1
                if garde % 100 == 0:
                    print(f"  {garde}/{NB_IMAGES}  ({rejet} rejetees)")
    else:
        print()
        print("=== ETAPE 4 ignoree (COLLECTER_IMAGES = False) ===")
        print("  La detection par couleur n'a besoin que de calibration.json.")

    field.setSFVec3f(depart)
    box.resetPhysics()
    print()
    print("Termine.")
    ur5.supervisor.simulationSetMode(ur5.supervisor.SIMULATION_MODE_PAUSE)


if __name__ == '__main__':
    main()
