"""
Generateur de dataset pour la vision (detection du cube rouge) -- MODELE CAMERA EXACT.

Principe
--------
On ne devine plus ou regarde la camera : on le calcule.

La camera n'est pas au bout de l'outil mais decalee sur le poignet
(translation 0 0.08 0.068 + rotation propre, cf. toolSlot du .wbt). Impossible
donc de deduire son champ de vision de la pose de l'effecteur "a vue".
`Supervisor.getFromDevice()` donne sa position et son orientation reelles dans
le monde : tout le reste en decoule analytiquement.

  Phase A  Mesure du montage. A une pose de reference, on lit la pose monde de
           la camera et celle de l'outil, d'ou la transformation rigide
           outil -> camera. Elle est constante : on peut ensuite predire la pose
           de la camera pour N'IMPORTE quelle pose d'outil, sans bouger le bras.
           On determine aussi empiriquement la convention d'axes de la camera
           Webots, en comparant projection theorique et cube reellement detecte.

  Phase B  Recherche d'orientation. La position de lecture est bloquee par la
           portee du bras (0.822 m sur ~0.85 m) : le seul levier est
           l'orientation du poignet. Grace a la phase A, evaluer une orientation
           ne coute plus qu'un calcul -- ni mouvement, ni rendu. On en teste
           donc des centaines et on garde celle qui voit le plus de table.

  Phase C  Verification croisee et homographie. A la pose retenue, on place le
           cube en quelques points connus et on compare projection theorique et
           detection reelle. Puis on ajuste une HOMOGRAPHIE pixel -> monde : une
           camera inclinee regardant un plan produit une transformation
           projective, pas affine -- un ajustement affine aurait un biais
           systematique.

  Phase D  Collecte. Le cube n'est tire que dans l'empreinte reellement visible.

Sorties (dans reference_ur5_repo/dataset/)
------------------------------------------
  images/*.jpg        256x256, l'entree du VGG16
  labels.csv          filename, x_pixel, y_pixel   (repere 256x256)
  calibration.json    pose de lecture, homographie pixel -> monde, zone visible
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

from ur5 import UR5, build_matrix, inverse_kinematics, forward_kinematics

PI = np.pi

# --- Monde ---------------------------------------------------------------
TABLE_CENTER = (-1.5, 0.0)          # table1, repere Webots
TABLE_SIZE = (0.8, 1.4)             # x, y
CUBE_Z = 0.955                      # hauteur du centre du cube pose sur la table
DEMO_CUBE_XY = (-1.55, 0.20)        # ou se trouve le cube dans la demo pick & place

# --- Camera --------------------------------------------------------------
IMG_SIZE = 256                      # taille enregistree = entree du reseau
FOV = 1.5708                        # fieldOfView du .wbt (rad, horizontal)
FOCAL = (IMG_SIZE / 2) / np.tan(FOV / 2)
CX = CY = IMG_SIZE / 2

# --- Reglages ------------------------------------------------------------
NUM_SAMPLES = 1000
MIN_RED_PIXELS = 20
MARGIN = 0.03                       # marge (m) retiree de la zone visible
REF_POSE = ([-0.10, -0.68, 0.45], [PI, 0.0, -PI / 2])   # pose de reference
MAX_REACH = 0.84                    # portee utile UR5e (m)

# Camera perpendiculaire a la table plutot que penchee.
# La phase B choisissait l'orientation qui voit le PLUS de table, ce qui donnait
# un basculement de 17 deg. Avec CAMERA_VERTICALE, elle privilegie les poses ou
# l'axe optique est proche de la verticale, quitte a voir moins de table.
# Une caméra verticale a deux avantages : l'image est une vue de dessus lisible,
# et l'homographie se rapproche d'une simple mise a l'echelle (la parallaxe due
# aux 10 cm de hauteur du cube diminue fortement).
CAMERA_VERTICALE = True
TILT_MAX = np.radians(12)           # inclinaison toleree quand CAMERA_VERTICALE

# Phase D (1000 images) : utile UNIQUEMENT pour reentrainer le VGG16.
# La detection par couleur n'a besoin que de la calibration, donc on la saute.
COLLECTER_IMAGES = False

# Les 24 conventions d'axes possibles pour une camera, exprimees dans son
# repere local sous la forme (avant, droite, bas). On enumere TOUTES les
# combinaisons orthogonales directes plutot que d'en deviner quelques-unes :
# la bonne est forcement dans le lot, et la mesure tranchera.
def _all_conventions():
    axes = {'+x': (1, 0, 0), '-x': (-1, 0, 0), '+y': (0, 1, 0),
            '-y': (0, -1, 0), '+z': (0, 0, 1), '-z': (0, 0, -1)}
    out = {}
    for fn, f in axes.items():
        for dn, d in axes.items():
            if abs(np.dot(f, d)) > 1e-9:        # doivent etre perpendiculaires
                continue
            r = np.cross(d, f)                   # droite x bas = avant
            out[f'fwd{fn}_down{dn}'] = (f, tuple(r), d)
    return out


CONVENTIONS = _all_conventions()


# =========================================================================
# Geometrie
# =========================================================================
def homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def camera_node(ur5):
    """
    Noeud Webots correspondant a la camera.

    Supervisor.getFromDevice() attend le TAG NUMERIQUE du peripherique, pas
    l'objet Device (sinon ctypes leve "Don't know how to convert parameter 1").
    Selon la version de l'API ce tag s'obtient via getTag() ou via l'attribut
    interne _tag : on essaie les deux.
    """
    cam = ur5.camera
    if cam is None:
        return None
    for get_tag in (lambda: cam.getTag(), lambda: cam._tag):
        try:
            tag = get_tag()
        except Exception:
            continue
        if not isinstance(tag, int):
            continue
        try:
            node = ur5.supervisor.getFromDevice(tag)
        except Exception:
            continue
        if node is not None:
            return node
    return None


def node_pose(node):
    """Pose monde 4x4 d'un noeud Webots."""
    R = np.array(node.getOrientation()).reshape(3, 3)
    t = np.array(node.getPosition())
    return homogeneous(R, t)


def table_bounds():
    cx, cy = TABLE_CENTER
    sx, sy = TABLE_SIZE
    return (cx - sx / 2, cx + sx / 2), (cy - sy / 2, cy + sy / 2)


def project(P_world, T_cam, conv):
    """
    Projette un point monde dans l'image.

    Returns:
        (u, v) en pixels, ou None si le point est derriere la camera.
    """
    fwd, right, down = (np.array(a, dtype=float) for a in conv)
    C = T_cam[:3, 3]
    R = T_cam[:3, :3]
    local = R.T @ (np.asarray(P_world, dtype=float) - C)
    z = float(local @ fwd)
    if z <= 1e-6:
        return None
    u = CX + FOCAL * float(local @ right) / z
    v = CY + FOCAL * float(local @ down) / z
    return u, v


def backproject(u, v, T_cam, conv, plane_z):
    """Intersection du rayon pixel (u, v) avec le plan horizontal z = plane_z."""
    fwd, right, down = (np.array(a, dtype=float) for a in conv)
    C = T_cam[:3, 3]
    R = T_cam[:3, :3]
    d_local = fwd + ((u - CX) / FOCAL) * right + ((v - CY) / FOCAL) * down
    d = R @ d_local
    if abs(d[2]) < 1e-9:
        return None
    t = (plane_z - C[2]) / d[2]
    if t <= 0:
        return None
    return C + t * d


def inclinaison(T_cam, conv):
    """
    Angle (rad) entre l'axe optique de la camera et la verticale descendante.

    0 = la camera regarde droit vers le bas.
    """
    fwd = np.array(conv[0], dtype=float)
    axe = T_cam[:3, :3] @ fwd
    axe = axe / np.linalg.norm(axe)
    return float(np.arccos(np.clip(-axe[2], -1.0, 1.0)))


def in_image(uv):
    return uv is not None and 0 <= uv[0] < IMG_SIZE and 0 <= uv[1] < IMG_SIZE


def coverage(T_cam, conv, n=20):
    """
    Fraction de la table visible, et visibilite du cube de la demo.
    Purement analytique : ni mouvement ni rendu.
    """
    (x0, x1), (y0, y1) = table_bounds()
    seen = 0
    total = 0
    for x in np.linspace(x0, x1, n):
        for y in np.linspace(y0, y1, n):
            total += 1
            if in_image(project((x, y, CUBE_Z), T_cam, conv)):
                seen += 1
    demo = in_image(project((*DEMO_CUBE_XY, CUBE_Z), T_cam, conv))
    return seen / total, demo


def visible_zone(T_cam, conv, n=40):
    """Boite englobante, sur la table, de ce que la camera voit reellement."""
    (x0, x1), (y0, y1) = table_bounds()
    pts = [(x, y) for x in np.linspace(x0, x1, n) for y in np.linspace(y0, y1, n)
           if in_image(project((x, y, CUBE_Z), T_cam, conv))]
    if not pts:
        return None
    a = np.array(pts)
    return {'x_min': float(a[:, 0].min()) + MARGIN, 'x_max': float(a[:, 0].max()) - MARGIN,
            'y_min': float(a[:, 1].min()) + MARGIN, 'y_max': float(a[:, 1].max()) - MARGIN}


# =========================================================================
# Perception
# =========================================================================
def detect_cube(img_rgb):
    """Centroide du cube rouge, en pixels de l'image IMG_SIZE x IMG_SIZE."""
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    r, g, b = (img[:, :, i].astype(int) for i in range(3))
    mask = (r > 110) & (r - g > 60) & (r - b > 60)
    if int(mask.sum()) < MIN_RED_PIXELS:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def place_cube(ur5, field, x, y, node=None):
    """
    Teleporte le cube.

    resetPhysics() est indispensable : sans lui le cube conserve sa vitesse et
    son inertie apres teleportation, rebondit ou traverse la table, et se
    retrouve ailleurs qu'a l'endroit demande -- ce qui fausse silencieusement
    toutes les correspondances pixel <-> monde. C'est aussi une des causes des
    avertissements "physics step could not be computed correctly".
    """
    field.setSFVec3f([float(x), float(y), CUBE_Z])
    if node is not None:
        node.resetPhysics()
    for _ in range(4):
        ur5.supervisor.step(ur5.timestep)


def goto(ur5, pos, rot):
    try:
        ur5.move_to_pose(list(pos), list(rot), wrist='up')
        for _ in range(20):
            ur5.supervisor.step(ur5.timestep)
        return True
    except Exception as e:
        print(f"    pose inatteignable : {e}")
        return False


# =========================================================================
# Phases
# =========================================================================
def phase_a(ur5, field, box, cam_node):
    """Repere camera + convention d'axes, mesures a la pose de reference."""
    print()
    print("=== PHASE A : modele de la camera ===")
    if not goto(ur5, *REF_POSE):
        raise SystemExit("La pose de reference est inatteignable.")

    T_base = node_pose(ur5.supervisor.getSelf())
    T_cam = node_pose(cam_node)

    # On repere la camera par rapport a la pose COMMANDEE, pas par rapport a
    # forward_kinematics(). Un premier essai avait montre 0.119 m d'ecart entre
    # les deux : les angles capteurs et le modele DH de ur5.py ne donnent pas
    # exactement le meme repere que celui vise par move_to_pose(). En restant
    # dans le repere commande de bout en bout (ici et en phase B), l'ecart
    # n'intervient plus.
    T_cmd = T_base @ build_matrix(REF_POSE[0], REF_POSE[1], euler='XYZ')
    T_cmd_cam = np.linalg.inv(T_cmd) @ T_cam

    print(f"  camera monde        : {np.round(T_cam[:3, 3], 3)}")
    print(f"  pose commandee      : {np.round(T_cmd[:3, 3], 3)}")
    print(f"  decalage cmd->cam   : {np.round(T_cmd_cam[:3, 3], 4)} "
          f"(|d| = {np.linalg.norm(T_cmd_cam[:3, 3]):.3f} m)")

    # Diagnostics : les trois reperes coincident-ils ?
    try:
        T_gt = T_base @ ur5.get_ground_truth()
        print(f"  ecart commande / reel (superviseur) : "
              f"{np.linalg.norm(T_gt[:3, 3] - T_cmd[:3, 3]) * 1000:6.1f} mm")
    except Exception:
        pass
    T_fk = T_base @ forward_kinematics(ur5.get_joint_angles())[0]
    print(f"  ecart commande / forward_kinematics : "
          f"{np.linalg.norm(T_fk[:3, 3] - T_cmd[:3, 3]) * 1000:6.1f} mm")

    # Convention d'axes : on balaye la table, on detecte, on compare.
    print("  determination de la convention d'axes...")
    (x0, x1), (y0, y1) = table_bounds()
    obs = []
    for x in np.linspace(x0 + 0.05, x1 - 0.05, 5):
        for y in np.linspace(y0 + 0.05, y1 - 0.05, 7):
            place_cube(ur5, field, x, y, box)
            uv = detect_cube(ur5.get_image())
            if uv is not None:
                obs.append(((x, y, CUBE_Z), uv))
    print(f"  {len(obs)} points detectes")
    if len(obs) < 4:
        raise SystemExit("Trop peu de points visibles pour identifier la convention.")

    scored = []
    for name, conv in CONVENTIONS.items():
        errs = []
        for P, (u, v) in obs:
            uv = project(P, T_cam, conv)
            errs.append(1e3 if uv is None else np.hypot(uv[0] - u, uv[1] - v))
        scored.append((float(np.sqrt(np.mean(np.square(errs)))), name))
    scored.sort()
    print("  meilleures conventions :")
    for rms, name in scored[:5]:
        print(f"    {name:22s} RMS = {rms:8.1f} px")

    best_err, best = scored[0]
    print(f"  -> convention retenue : {best} ({best_err:.1f} px)")
    reliable = best_err <= 30
    if not reliable:
        print("  MODELE NON FIABLE : aucune convention ne reproduit les mesures.")
        print("  -> la phase B basculera en mode empirique (mesure reelle).")
    return T_cmd_cam, CONVENTIONS[best], best, best_err, reliable


def phase_b(ur5, T_cmd_cam, conv):
    """Recherche analytique de l'orientation qui voit le plus de table."""
    print("\n=== PHASE B : recherche d'orientation (analytique) ===")
    T_base = node_pose(ur5.supervisor.getSelf())
    base_pos = REF_POSE[0]

    cands, tested, reachable = [], 0, 0
    for dx in (-0.10, -0.05, 0.0, 0.05):
        for dy in (-0.04, 0.0, 0.04):
            for dz in (0.40, 0.45, 0.50):
                pos = [base_pos[0] + dx, base_pos[1] + dy, dz]
                if np.linalg.norm(pos) > MAX_REACH:
                    continue
                # Grille elargie : imposer la verticalite restreint fortement
                # les candidats, il faut donc explorer plus large qu'avant.
                for roll in PI + np.arange(-0.6, 0.61, 0.15):
                    for pitch in np.arange(-0.9, 0.91, 0.15):
                        for yaw in -PI / 2 + np.arange(-0.6, 0.61, 0.3):
                            tested += 1
                            rot = [float(roll), float(pitch), float(yaw)]
                            T = build_matrix(pos, rot, euler='XYZ')
                            try:
                                q = inverse_kinematics(T, wrist='up',
                                                       shoulder='left', elbow='up')
                                if np.any(np.isnan(q)):
                                    continue
                            except Exception:
                                continue
                            reachable += 1
                            T_cam = (T_base @ T) @ T_cmd_cam
                            cov, demo = coverage(T_cam, conv, n=10)
                            cands.append({'pos': pos, 'rot': rot, 'cov': cov,
                                          'demo': demo,
                                          'tilt': inclinaison(T_cam, conv)})

    print(f"  {tested} orientations testees, {reachable} atteignables")
    if not cands:
        raise SystemExit("Aucune pose atteignable.")

    def montre(titre, liste):
        print(f"  {titre} :")
        for c in liste[:4]:
            print(f"    cov {100 * c['cov']:5.1f} %  inclinaison {np.degrees(c['tilt']):5.1f} deg"
                  f"  cube {'OUI' if c['demo'] else 'non'}"
                  f"  pos={[round(v, 2) for v in c['pos']]} rot={[round(v, 2) for v in c['rot']]}")

    # Reference : ce que donnerait le critere "voir le plus de table"
    par_couverture = sorted(cands, key=lambda c: (c['demo'], c['cov']), reverse=True)
    montre("si on maximise la couverture (ancien critere)", par_couverture)

    if CAMERA_VERTICALE:
        droites = [c for c in cands if c['tilt'] <= TILT_MAX and c['demo']]
        if not droites:
            droites = [c for c in cands if c['tilt'] <= TILT_MAX]
        if droites:
            droites.sort(key=lambda c: (c['demo'], c['cov']), reverse=True)
            montre(f"camera verticale (inclinaison <= {np.degrees(TILT_MAX):.0f} deg)",
                   droites)
            best = droites[0]
            perte = par_couverture[0]['cov'] - best['cov']
            print(f"  -> pose verticale retenue : {np.degrees(best['tilt']):.1f} deg "
                  f"d'inclinaison, {100 * best['cov']:.0f} % de table "
                  f"({100 * perte:+.0f} points par rapport au meilleur)")
        else:
            best = par_couverture[0]
            print(f"  AUCUNE pose atteignable sous {np.degrees(TILT_MAX):.0f} deg "
                  "d'inclinaison.")
            print(f"  -> repli sur la meilleure couverture "
                  f"({np.degrees(best['tilt']):.1f} deg)")
    else:
        best = par_couverture[0]
    if not best['demo']:
        print("\n  AUCUNE pose atteignable ne voit le cube de la demo "
              f"{DEMO_CUBE_XY}.")
        print("  -> il faudra deplacer le cube dans la zone visible (voir ci-dessous).")
    return best


def phase_b_empirical(ur5, field, box):
    """
    Repli quand le modele de projection n'est pas valide.

    On ne calcule plus : on mesure. Quelques poses candidates, un balayage
    grossier pour chacune, et on garde celle qui voit le plus de table. C'est
    plus lent (un deplacement de bras par candidat) mais ca ne depend d'aucune
    hypothese sur la camera.
    """
    print()
    print("=== PHASE B (empirique) : mesure reelle de chaque pose ===")
    cands = [
        ([-0.10, -0.68, 0.45], [PI, 0.00, -PI / 2]),
        ([-0.20, -0.64, 0.50], [PI, 0.30, -PI / 2]),
        ([-0.20, -0.64, 0.50], [PI, 0.00, -PI / 2]),
        ([-0.20, -0.64, 0.50], [PI, -0.30, -PI / 2]),
        ([-0.05, -0.65, 0.50], [PI, 0.45, -PI / 2]),
        ([0.00, -0.62, 0.52], [PI, 0.30, -PI / 2]),
        ([-0.15, -0.66, 0.48], [PI, 0.15, -PI / 2]),
    ]
    (x0, x1), (y0, y1) = table_bounds()
    best = None
    for i, (pos, rot) in enumerate(cands, 1):
        print(f"  [{i}/{len(cands)}] pos={pos} rot={[round(r, 2) for r in rot]}")
        if not goto(ur5, pos, rot):
            continue
        seen, demo = [], False
        for x in np.linspace(x0 + 0.04, x1 - 0.04, 5):
            for y in np.linspace(y0 + 0.04, y1 - 0.04, 6):
                place_cube(ur5, field, x, y, box)
                if detect_cube(ur5.get_image()) is not None:
                    seen.append((x, y))
        place_cube(ur5, field, *DEMO_CUBE_XY, box)
        demo = detect_cube(ur5.get_image()) is not None
        cov = len(seen) / 30.0
        print(f"      couverture {100 * cov:5.1f} %   cube demo "
              f"{'OUI' if demo else 'non'}")
        cand = {'pos': pos, 'rot': rot, 'cov': cov, 'demo': demo}
        if best is None or (cand['demo'], cand['cov']) > (best['demo'], best['cov']):
            best = cand
    if best is None:
        raise SystemExit("Aucune pose candidate atteignable.")
    print(f"  -> retenue : pos={best['pos']} rot={[round(r, 2) for r in best['rot']]} "
          f"({100 * best['cov']:.0f} %, cube demo {'OUI' if best['demo'] else 'non'})")
    return best


def phase_c(ur5, field, box, cam_node, best, conv, reliable):
    """
    Calibration finale a la pose retenue, entierement par la MESURE.

    On balaye la table, on detecte le cube, et on ajuste l'homographie sur les
    correspondances observees. Cette phase ne depend donc pas du modele de
    projection : meme si la convention d'axes n'a pas pu etre identifiee, la
    calibration produite reste correcte.
    """
    print()
    print("=== PHASE C : calibration mesuree ===")
    if not goto(ur5, best['pos'], best['rot']):
        raise SystemExit("La pose retenue est finalement inatteignable.")

    T_cam = node_pose(cam_node)
    (x0, x1), (y0, y1) = table_bounds()

    src, dst, errs = [], [], []
    for x in np.linspace(x0 + 0.02, x1 - 0.02, 9):
        for y in np.linspace(y0 + 0.02, y1 - 0.02, 12):
            place_cube(ur5, field, x, y, box)
            uv = detect_cube(ur5.get_image())
            if uv is None:
                continue
            src.append(uv)
            dst.append((x, y))
            if reliable:
                th = project((x, y, CUBE_Z), T_cam, conv)
                if th is not None:
                    errs.append(np.hypot(th[0] - uv[0], th[1] - uv[1]))

    print(f"  {len(src)} correspondances mesurees sur 108 positions balayees")
    if errs:
        print(f"  ecart modele / mesure : moyenne {np.mean(errs):.1f} px, "
              f"max {np.max(errs):.1f} px")
    if len(src) < 10:
        raise SystemExit("Trop peu de correspondances : la camera ne voit "
                         "presque rien a cette pose.")

    a = np.array(dst)
    zone = {'x_min': float(a[:, 0].min()) + MARGIN,
            'x_max': float(a[:, 0].max()) - MARGIN,
            'y_min': float(a[:, 1].min()) + MARGIN,
            'y_max': float(a[:, 1].max()) - MARGIN}
    print(f"  zone visible MESUREE : x [{zone['x_min']:.3f} , {zone['x_max']:.3f}]  "
          f"y [{zone['y_min']:.3f} , {zone['y_max']:.3f}]")

    H, _ = cv2.findHomography(np.array(src, dtype=np.float64),
                              np.array(dst, dtype=np.float64), cv2.RANSAC, 0.01)
    if H is None:
        raise SystemExit("Ajustement de l'homographie impossible.")
    P = np.hstack([np.array(src), np.ones((len(src), 1))]) @ H.T
    P = P[:, :2] / P[:, 2:3]
    rmse = float(np.sqrt(np.mean(np.sum((P - a) ** 2, axis=1))))
    print(f"  homographie pixel -> monde : RMSE = {1000 * rmse:.1f} mm")

    demo_in = (zone['x_min'] <= DEMO_CUBE_XY[0] <= zone['x_max'] and
               zone['y_min'] <= DEMO_CUBE_XY[1] <= zone['y_max'])
    print(f"  cube de la demo {DEMO_CUBE_XY} dans la zone : "
          f"{'OUI' if demo_in else 'NON'}")
    return T_cam, zone, H, rmse, demo_in


def phase_d(ur5, field, box, zone, dataset_dir, img_dir):
    """Collecte du dataset, uniquement dans la zone visible."""
    print(f"\n=== PHASE D : collecte de {NUM_SAMPLES} images ===")
    for f in glob.glob(os.path.join(img_dir, '*.jpg')):
        os.remove(f)

    kept = skipped = 0
    with open(os.path.join(dataset_dir, 'labels.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['filename', 'x_pixel', 'y_pixel'])
        while kept < NUM_SAMPLES:
            x = random.uniform(zone['x_min'], zone['x_max'])
            y = random.uniform(zone['y_min'], zone['y_max'])
            place_cube(ur5, field, x, y, box)
            img = ur5.get_image()
            uv = detect_cube(img)
            if uv is None:                      # cube masque par le bras
                skipped += 1
                continue
            name = f"img_{kept:04d}.jpg"
            cv2.imwrite(os.path.join(img_dir, name),
                        cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   (IMG_SIZE, IMG_SIZE)))
            w.writerow([name, round(uv[0], 3), round(uv[1], 3)])
            kept += 1
            if kept % 100 == 0:
                print(f"  {kept}/{NUM_SAMPLES}  ({skipped} rejetees)")
    return kept, skipped


# =========================================================================
def main():
    random.seed(0)
    np.random.seed(0)

    ur5 = UR5()
    ur5.use_pinn = False            # IK analytique exacte pour la calibration
    ur5.setup_camera()

    box = ur5.bottle
    if box is None:
        print("ERREUR : aucun noeud DEF RED_BOX / bottle dans ce monde.")
        return
    field = box.getField("translation")
    home = field.getSFVec3f()

    cam_node = camera_node(ur5)
    if cam_node is None:
        print("ERREUR : impossible d'acceder au noeud de la camera via "
              "Supervisor.getFromDevice(). Verifier que le robot a bien "
              "supervisor TRUE dans le .wbt.")
        return
    print(f"Camera localisee : {np.round(np.array(cam_node.getPosition()), 3)}")

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    dataset_dir = os.path.join(base, 'dataset')
    img_dir = os.path.join(dataset_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    T_cmd_cam, conv, conv_name, conv_err, reliable = phase_a(ur5, field, box, cam_node)
    if reliable:
        best = phase_b(ur5, T_cmd_cam, conv)
    else:
        best = phase_b_empirical(ur5, field, box)
    T_cam, zone, H, rmse, demo_in = phase_c(ur5, field, box, cam_node,
                                            best, conv, reliable)

    json.dump({
        'reading_pose': {'position': list(best['pos']), 'orientation': list(best['rot'])},
        'image_size': IMG_SIZE,
        'camera_resolution': 512,
        'field_of_view': FOV,
        'cube_z': CUBE_Z,
        # Transformation rigide entre le repere COMMANDE par move_to_pose (bout de
        # la chaine DH) et la camera. Elle est constante : la camera est vissee
        # sur le poignet. Sans elle, commander une position place un repere
        # mathematique invisible, pas l'objectif -- d'ou UR5.move_camera_to().
        'tool_to_camera': T_cmd_cam.tolist(),
        'axis_convention': conv_name,
        'axis_convention_rms_px': conv_err,
        'pixel_to_world_homography': H.tolist(),
        'fit_rmse_m': rmse,
        'visible_zone_world': zone,
        'coverage': best['cov'],
        'sees_demo_cube': bool(demo_in),
        'model_reliable': bool(reliable),
    }, open(os.path.join(dataset_dir, 'calibration.json'), 'w'), indent=2)
    print("  calibration.json ecrit")

    if COLLECTER_IMAGES:
        kept, skipped = phase_d(ur5, field, box, zone, dataset_dir, img_dir)
    else:
        kept = skipped = 0
        print()
        print("=== PHASE D ignoree (COLLECTER_IMAGES = False) ===")
        print("  La detection par couleur n'a besoin que de calibration.json.")
        print("  Repasser a True uniquement pour reentrainer le VGG16.")

    field.setSFVec3f(home)
    print(f"\nTermine : {kept} images, {skipped} rejetees.")
    print(f"Couverture de la table : {100 * best['cov']:.0f} %")
    if not demo_in:
        cx = (zone['x_min'] + zone['x_max']) / 2
        cy = (zone['y_min'] + zone['y_max']) / 2
        print("\n  Le cube de la demo n'est PAS dans le champ de la camera.")
        print("  Deplacer DEF bottle dans les .wbt vers le centre de la zone visible :")
        print(f"      translation {cx:.3f} {cy:.3f} {CUBE_Z}")
    print("\nEtape suivante : reentrainer computer_vision/train_vgg16.ipynb")
    ur5.supervisor.simulationSetMode(ur5.supervisor.SIMULATION_MODE_PAUSE)


if __name__ == '__main__':
    main()
