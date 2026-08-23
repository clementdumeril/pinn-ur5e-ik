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

# Le deuxieme angle est le PITCH, et c'est exactement l'inclinaison de la
# camera : avec 0.30 rad l'axe optique tombait 135 mm en avant de la verticale
# pour 436 mm de hauteur, soit 17.20 deg -- contre 17.19 deg pour 0.30 rad.
# A 0, la camera regarde droit vers le bas.
ROT_CAMERA = [PI, 0.0, -PI / 2]

# --- Reglages ------------------------------------------------------------
GRILLE = 11                         # balayage GRILLE x GRILLE sur la table
MIN_RED_PIXELS = 20                 # seuil de detection du cube
MARGE = 0.03                        # marge (m) retiree de la zone visible
IMG_SIZE = 256                      # taille enregistree = entree du reseau
STABILISATION = 1.5                 # s de temps simule apres le deplacement
FOV = 1.5708                        # fieldOfView de la camera (rad), cf. le .wbt
DEBORD = 1.45                       # on balaye 45 % au-dela du champ predit,
                                    # pour mesurer la frontiere et non la deviner

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


def bloquer_pince(ur5, ouverture=0.04):
    """
    Maintient les doigts de la PandaHand a une ouverture fixe.

    Sans cela ils restent des articulations LIBRES : ur5.init_handles() vide
    self.finger_joints puis parcourt cette liste vide, si bien qu'aucun doigt
    n'est jamais pilote. Ils ballottent au gre des mouvements du bras, ce qui
    multiplie les contacts et provoque les avertissements
    "physics step could not be computed correctly".

    Les noms changent selon la version du PROTO :
      R2023a : panda_finger_joint1 / panda_finger_joint2
      R2025a : panda_finger::left  / panda_finger::right
    """
    for a, b in (("panda_finger_joint1", "panda_finger_joint2"),
                 ("panda_finger::left", "panda_finger::right")):
        f1, f2 = ur5.supervisor.getDevice(a), ur5.supervisor.getDevice(b)
        if f1 is not None and f2 is not None:
            for f in (f1, f2):
                f.setVelocity(0.1)
                f.setPosition(ouverture)
            print(f"  pince bloquee a {ouverture:.3f} m ({a})")
            return True
    print("  ATTENTION : doigts de la pince introuvables, ils resteront libres")
    return False


def ou_pointe_la_camera(T_cam):
    """
    Determine par la mesure quel axe local de la camera est l'axe optique.

    Deux tentatives d'identifier la convention d'axes de Webots par ajustement
    ont echoue (595 puis 817 px). On procede donc autrement : pour chacun des
    6 axes locaux possibles, on lance un rayon depuis la camera et on regarde
    ou il rencontre le plan de la table. Un seul peut etre l'axe optique --
    celui qui tombe sur la table, pres de ce que l'image montre effectivement.

    Purement informatif : rien dans la calibration n'en depend, puisque
    l'homographie est ajustee sur des correspondances mesurees.
    """
    C = T_cam[:3, 3]
    R = T_cam[:3, :3]
    (x0, x1), (y0, y1) = bornes_table()

    print()
    print("  --- ou pointe chaque axe local de la camera ---")
    print(f"  camera a {np.round(C, 3)}, table a z = {CUBE_Z:.3f}")
    axes = {'+x': (1, 0, 0), '-x': (-1, 0, 0), '+y': (0, 1, 0),
            '-y': (0, -1, 0), '+z': (0, 0, 1), '-z': (0, 0, -1)}
    for nom, a in axes.items():
        d = R @ np.array(a, dtype=float)
        if abs(d[2]) < 1e-6 or (CUBE_Z - C[2]) / d[2] <= 0:
            print(f"    {nom}  ne descend pas vers la table")
            continue
        t = (CUBE_Z - C[2]) / d[2]
        P = C + t * d
        sur = (x0 <= P[0] <= x1) and (y0 <= P[1] <= y1)
        dist = np.linalg.norm(P[:2] - C[:2]) * 1000
        print(f"    {nom}  vise x {P[0]:+.3f}  y {P[1]:+.3f}   "
              f"{'SUR LA TABLE' if sur else 'hors table'}   "
              f"a {dist:.0f} mm de la verticale")
    print("  -> l'axe optique est celui qui vise la table ; l'ecart a la")
    print("     verticale donne l'inclinaison a corriger via ROT_CAMERA.")


def main():
    random.seed(0)
    np.random.seed(0)

    ur5 = UR5()
    ur5.use_pinn = False             # IK analytique exacte pour la calibration
    ur5.setup_camera()
    bloquer_pince(ur5)               # sinon les doigts ballottent

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

    # Passer par une posture neutre connue avant de viser la pose de lecture.
    # Sans cela le bras part de la configuration ou le monde l'a laisse, et le
    # chemin vers la solution d'IK peut etre acrobatique -- c'est le mouvement
    # etrange visible au demarrage.
    ur5.move_to_config([0, 0, 0, 0, 0, 0])
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

    ou_pointe_la_camera(T_cam)

    # ---------------------------------------------------------------
    # ETAPE 2 : que voit-elle ? on regarde, on ne calcule pas
    # ---------------------------------------------------------------
    print()
    print(f"=== ETAPE 2 : balayage {GRILLE}x{GRILLE} ===")

    # On balaye ce que la camera peut voir, pas toute la table.
    #
    # Une camera a hauteur h avec un champ FOV couvre un carre de cote
    # 2*h*tan(FOV/2), centre a sa verticale. Balayer toute la table faisait
    # apparaitre le cube hors cadre 55 fois sur 121 -- du temps perdu, et un
    # spectacle deroutant. On ajoute DEBORD pour depasser un peu le champ
    # predit : la frontiere reste ainsi MESUREE, pas supposee.
    (x0, x1), (y0, y1) = bornes_table()
    h = float(T_cam[2, 3]) - CUBE_Z
    demi = h * np.tan(FOV / 2) * DEBORD
    cx, cy = float(T_cam[0, 3]), float(T_cam[1, 3])
    bx = (max(x0 + 0.02, cx - demi), min(x1 - 0.02, cx + demi))
    by = (max(y0 + 0.02, cy - demi), min(y1 - 0.02, cy + demi))
    print(f"  hauteur {h:.3f} m -> champ predit {2 * h * np.tan(FOV / 2):.3f} m de cote")
    print(f"  zone balayee : x [{bx[0]:.3f}, {bx[1]:.3f}]  y [{by[0]:.3f}, {by[1]:.3f}]")

    src, dst = [], []
    for x in np.linspace(bx[0], bx[1], GRILLE):
        for y in np.linspace(by[0], by[1], GRILLE):
            poser_cube(ur5, field, box, x, y)
            uv = detecter_cube(ur5.get_image())
            if uv is not None:
                src.append(uv)
                dst.append((x, y))

    total = GRILLE * GRILLE
    print(f"  cube visible sur {len(src)}/{total} positions balayees "
          f"({100 * len(src) / total:.0f} %)")
    if len(src) / total > 0.97:
        print("  quasi tout est visible -> le champ deborde la zone balayee,")
        print("     augmenter DEBORD pour mesurer la vraie frontiere.")
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
    # Seuil RANSAC a 20 mm, pas 5.
    # A 5 mm il etait sous la mediane des residus (5.3 mm) : RANSAC classait la
    # moitie des points en aberrants et ajustait sur un sous-ensemble, laissant
    # aux points ecartes de gros residus -- d'ou un RMSE de 38 mm pour une
    # mediane de 5 mm. Le bruit reel etant de l'ordre de 5 mm, 20 mm ne rejette
    # que les vraies aberrations.
    H, inliers = cv2.findHomography(np.array(src, dtype=np.float64),
                                    np.array(dst, dtype=np.float64),
                                    cv2.RANSAC, 0.020)
    if H is None:
        print("  Ajustement impossible.")
        return
    P = np.hstack([np.array(src), np.ones((len(src), 1))]) @ H.T
    P = P[:, :2] / P[:, 2:3]
    res = np.linalg.norm(P - np.array(dst), axis=1) * 1000
    n_in = int(inliers.sum()) if inliers is not None else len(src)
    print(f"  points retenus : {n_in}/{len(src)}")
    print(f"  residus : moyenne {res.mean():.1f} mm, median {np.median(res):.1f} mm, "
          f"max {res.max():.1f} mm")

    # --- Residu en fonction de la distance au centre du champ --------------
    # Le cube depasse de 10 cm au-dessus de la table, alors qu'une homographie
    # suppose une scene PLATE. Le centroide des pixels rouges est celui de la
    # face visible : sous la camera on ne voit que le dessus, au bord on voit
    # aussi un cote et le centroide descend. La hauteur effective varie donc
    # avec la position, ce qu'aucune homographie ne peut representer.
    # L'erreur qui en resulte croit avec la distance au centre -- ce profil le
    # montre, et delimite la zone reellement exploitable.
    d = np.linalg.norm(np.array(dst) - np.array([cx, cy]), axis=1)
    print("  residu selon la distance au centre du champ :")
    for lo in np.arange(0.0, d.max() + 0.001, 0.10):
        m = (d >= lo) & (d < lo + 0.10)
        if m.sum():
            print(f"    {lo:.2f} a {lo + 0.10:.2f} m : {m.sum():3d} pts, "
                  f"median {np.median(res[m]):5.1f} mm, max {res[m].max():5.1f} mm")

    # --- Correction radiale de la parallaxe --------------------------------
    # Le cube depasse de 10 cm au-dessus de la table : le centroide des pixels
    # rouges n'est pas au-dessus de sa position reelle. Sous la camera on ne
    # voit que le dessus (deport nul) ; plus on s'ecarte, plus une face laterale
    # apparait et plus le centroide glisse VERS L'EXTERIEUR -- jusqu'a 57 mm au
    # bord du champ.
    #
    # La camera etant verticale, la scene est symetrique autour de l'axe
    # optique : ce deport ne depend que de la DISTANCE au centre, pas de la
    # direction. Un polynome en r l'absorbe donc entierement.
    #
    # On ajuste r_vrai = r_estime - (a*r + b*r^2) sur les 121 mesures.
    est = P                                   # positions estimees par H seule
    vrai = np.array(dst)
    centre_sol = np.array([cx, cy])
    r_est = np.linalg.norm(est - centre_sol, axis=1)
    r_vrai = np.linalg.norm(vrai - centre_sol, axis=1)
    ok = r_est > 1e-6
    A = np.column_stack([r_est[ok], r_est[ok] ** 2])
    coef, *_ = np.linalg.lstsq(A, (r_est[ok] - r_vrai[ok]), rcond=None)

    # residus apres correction
    fac = np.ones(len(est))
    corr = coef[0] * r_est + coef[1] * r_est ** 2
    fac[ok] = (r_est[ok] - corr[ok]) / r_est[ok]
    est_corr = centre_sol + (est - centre_sol) * fac[:, None]
    res2 = np.linalg.norm(est_corr - vrai, axis=1) * 1000
    print()
    print("  --- correction radiale de la parallaxe ---")
    print(f"  coefficients : a = {coef[0]:+.4f}   b = {coef[1]:+.4f}")
    print(f"  residus AVANT : median {np.median(res):5.1f} mm, max {res.max():5.1f} mm")
    print(f"  residus APRES : median {np.median(res2):5.1f} mm, max {res2.max():5.1f} mm")

    # --- Champ de vue, calcule et non plus cherche -------------------------
    # Camera verticale : elle voit un disque de rayon h*tan(FOV/2) centre a sa
    # verticale. Le disque inscrit est independant de la rotation de l'image,
    # donc toujours a l'interieur du champ reel.
    rayon = h * np.tan(FOV / 2) - MARGE
    print()
    print("  --- champ de vue ---")
    print(f"  disque utilisable : centre ({cx:+.3f}, {cy:+.3f}), rayon {rayon:.3f} m")

    a = np.array(dst)
    zone = {'x_min': float(a[:, 0].min()) + MARGE, 'x_max': float(a[:, 0].max()) - MARGE,
            'y_min': float(a[:, 1].min()) + MARGE, 'y_max': float(a[:, 1].max()) - MARGE}
    print(f"  zone visible : x [{zone['x_min']:.3f}, {zone['x_max']:.3f}]  "
          f"y [{zone['y_min']:.3f}, {zone['y_max']:.3f}]")

    # --- Inclinaison de la camera, mesuree sans modele ---------------------
    # Une camera verticale voit une zone CENTREE sous elle. L'ecart entre sa
    # position au sol et le centre de ce qu'elle voit donne donc directement son
    # decentrage, sans avoir a identifier la convention d'axes de Webots.
    centre = np.array([(zone['x_min'] + zone['x_max']) / 2,
                       (zone['y_min'] + zone['y_max']) / 2])
    sous_camera = T_cam[:3, 3][:2]
    decal = centre - sous_camera
    hauteur = float(T_cam[2, 3]) - CUBE_Z
    angle = np.degrees(np.arctan2(np.linalg.norm(decal), hauteur))
    print()
    print("  --- verticalite de la camera ---")
    print(f"  camera au sol      : x {sous_camera[0]:+.3f}  y {sous_camera[1]:+.3f}")
    print(f"  centre du champ    : x {centre[0]:+.3f}  y {centre[1]:+.3f}")
    print(f"  decentrage         : dx {decal[0]*1000:+.0f} mm  dy {decal[1]*1000:+.0f} mm "
          f"({np.linalg.norm(decal)*1000:.0f} mm)")
    print(f"  hauteur au-dessus de la table : {hauteur:.3f} m")
    print(f"  -> inclinaison apparente : {angle:.1f} deg  "
          f"({'quasi verticale' if angle < 5 else 'penchee'})")

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
        'camera_sol_xy': [float(cx), float(cy)],
        'camera_hauteur_m': float(h),
        'rayon_utilisable_m': float(rayon),
        'correction_radiale': [float(coef[0]), float(coef[1])],
        'residu_median_corrige_mm': float(np.median(res2)),
        'decentrage_mm': float(np.linalg.norm(decal) * 1000),
        'inclinaison_apparente_deg': float(angle),
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
