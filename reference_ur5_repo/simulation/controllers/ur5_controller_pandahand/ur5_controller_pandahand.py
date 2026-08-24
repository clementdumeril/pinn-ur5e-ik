import os
import sys
from math import pi

import numpy as np

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from ur5 import UR5

PI = pi
ur5 = UR5()

# ---------------------------------------------------------
# MODE DE PERCEPTION
# ---------------------------------------------------------
#   "color"      seuillage couleur du cube rouge. Precis (< 3 mm), instantane,
#                aucun entrainement. C'est la methode qui a servi a etiqueter
#                le dataset du VGG16 -- donc au moins aussi bonne que lui.
#   "cnn"        reseau VGG16 (computer_vision/vgg16.h5). Mesure sur les images
#                de validation : ~49 mm d'erreur moyenne, soit 17 % de saisies
#                reussies sur un cube de 50 mm. Garde pour comparaison.
#   "supervisor" position exacte lue dans le moteur Webots. Ce n'est pas de la
#                perception, c'est la verite terrain -- utile pour isoler les
#                problemes d'IK sans melanger avec ceux de vision.
VISION_MODE = "color"

# Mesure de l'erreur reelle du PINN dans Webots, en comparant les poses
# atteintes avec le PINN et avec l'IK analytique sur les memes cibles.
# Allonge le lancement de quelques dizaines de secondes ; a activer quand on
# veut un chiffre defendable, pas a chaque demo.
MESURER_ERREUR_PINN = True


# ---------------------------------------------------------
# ARGUMENTS POUR LA SIMULATION COMPARATIVE
# ---------------------------------------------------------
# Webots passe toujours au moins un argument, et les mondes mono-robot y
# laissent une chaine VIDE (controllerArgs [ "" ]). L'ancien test
# `if len(sys.argv) > 1` la prenait donc pour un mode, mode devenait "",
# different de "pinn", et use_pinn restait a False : le PINN etait charge en
# memoire mais JAMAIS appele. Toute la demo tournait sur l'IK analytique.
# On ignore desormais les arguments vides.
mode = "pinn"
if len(sys.argv) > 1 and sys.argv[1].strip():
    mode = sys.argv[1].strip().lower()

print(f"[Mode] Solveur de cinematique inverse : "
      f"{'PINN (reseau de neurones)' if mode == 'pinn' else 'IK analytique'}")

# Affichage du titre selon le robot
if mode == "pinn":
    ur5.supervisor.setLabel(1, "ROBOT ROUGE : IA TRUE PINN", 0.02, 0.05, 0.1, 0xff0000, 0, "Arial")
else:
    ur5.supervisor.setLabel(2, "ROBOT VERT : IK ANALYTIQUE", 0.02, 0.10, 0.1, 0x00ff00, 0, "Arial")

# --- PandaHand : recuperation des moteurs -------------------------------
# Les noms des moteurs changent selon la version du PROTO PandaHand :
#     R2023a : panda_finger_joint1 / panda_finger_joint2
#     R2025a : panda_finger::left  / panda_finger::right
# Le monde declare la R2023a mais Webots R2025a est installe. L'ancien code
# faisait `if panda_f1: ...` : si le nom ne correspondait pas, la pince ne
# s'actionnait JAMAIS, sans le moindre message. On essaie donc les deux jeux de
# noms, et on le dit franchement si aucun ne marche.
FINGER_OPEN_INIT = 0.04

NOMS_DOIGTS = [
    ("panda_finger_joint1", "panda_finger_joint2"),   # R2023a
    ("panda_finger::left", "panda_finger::right"),    # R2025a
]

panda_f1 = panda_f2 = None
for _a, _b in NOMS_DOIGTS:
    _f1, _f2 = ur5.supervisor.getDevice(_a), ur5.supervisor.getDevice(_b)
    if _f1 is not None and _f2 is not None:
        panda_f1, panda_f2 = _f1, _f2
        print(f"[Pince] Moteurs trouves : {_a} / {_b}")
        break

if panda_f1 is None:
    print("[Pince] ERREUR : aucun moteur de pince trouve.")
    print(f"[Pince] Noms essayes : {NOMS_DOIGTS}")
    print("[Pince] La pince ne se fermera pas -- la saisie echouera forcement.")
panda_s1 = panda_s2 = None
if panda_f1 is None:
    print("[Pince] La pince ne se fermera pas -- la saisie echouera forcement.")
else:
    panda_f1.setPosition(FINGER_OPEN_INIT)
    panda_f2.setPosition(FINGER_OPEN_INIT)
    panda_f1.setVelocity(0.1)
    panda_f2.setVelocity(0.1)
    # Capteurs de position : ils disent l'ouverture REELLE des doigts. C'est le
    # seul moyen de distinguer "la pince ne se ferme pas" de "elle se ferme mais
    # rate le cube" -- deux causes qui donnent le meme symptome.
    for _f, _nom in ((panda_f1, 'panda_s1'), (panda_f2, 'panda_s2')):
        try:
            _s = _f.getPositionSensor()
            if _s is not None:
                _s.enable(ur5.timestep)
                if _nom == 'panda_s1':
                    panda_s1 = _s
                else:
                    panda_s2 = _s
        except Exception as _e:
            print(f"[Pince] Capteur indisponible : {_e}")
    print(f"[Pince] Capteurs de doigts : "
          f"{'actifs' if panda_s1 and panda_s2 else 'INDISPONIBLES'}")


# --- Hauteurs de la sequence (repere robot, en metres) -------------------
# ATTENTION : ces z sont la position du repere 6 (la bride), PAS des doigts.
# La table DH de ur5.py utilise d6 = 0.0996 + 0.1237, ou le 0.1237 est un offset
# d'outil ajoute a la main pour la pince Robotiq 3F du projet d'origine. La
# PandaHand n'a pas la meme longueur, donc GRASP_Z est une valeur empirique a
# recaler empiriquement : verifier_saisie() dit si le cube a ete souleve, et
# propose une correction quand ce n'est pas le cas.
APPROACH_Z = 0.40       # survol, sans risque
# GRASP_Z mesure : avec 0.05, la bride visait 2.8 mm au-dessus du centre du cube
# (mesure : centre du cube a z = +0.0472). Les doigts sont donc pratiquement au
# niveau du repere 6 -- le +0.1237 de d6 dans ur5.py couvre bien la longueur de
# l'outil. On vise desormais le centre du cube.
GRASP_Z = 0.030         # hauteur de saisie -- remplacee si CALIBRER_HAUTEUR
                        # (cube ramene a 5 cm : centre a z = 0.030 en repere robot,
                        #  contre 0.055 quand il faisait 10 cm)

# Cherche automatiquement la bonne hauteur de saisie au lancement.
# Deux tentatives de la deduire par le calcul ont echoue : la position des
# doigts par rapport au bout de la chaine DH n'est pas accessible proprement
# (un moteur Webots n'est pas un noeud Pose, et DEF frame6 est place 10.34 cm
# plus loin que le bout de la chaine). On la MESURE donc : on essaie plusieurs
# hauteurs et on garde celle qui souleve reellement le cube.
# Une fois la valeur connue, la recopier dans GRASP_Z et repasser a False.
# La recherche a converge du premier coup sur 0.030, valeur desormais en dur
# dans GRASP_Z. Repasser a True si le cube ou la pince changent.
CALIBRER_HAUTEUR = False
HAUTEURS_A_TESTER = [0.030, 0.015, 0.000, 0.045, -0.015, 0.060]
RELEASE_Z = 0.05        # hauteur de depose dans le bac

FINGER_OPEN = 0.04      # course d'un doigt (m)
FINGER_SPEED = 0.1      # vitesse imposee plus haut par setVelocity (m/s)
EPAISSEUR_DOIGT_MM = 7.4  # retrait de la face interne de chaque doigt par
                          # rapport a la position lue par son capteur


def hauteur_cube():
    """Hauteur (repere robot) du centre du cube, ou None."""
    f = ur5.get_bottle_frame()
    return None if f is None else float(f[2, 3])


def diagnostic_hauteur():
    """
    Rapporte les hauteurs en jeu, sans pretendre en deduire GRASP_Z.

    Une premiere version comparait GRASP_Z au noeud Webots DEF frame6 et
    proposait une correction. C'etait faux : ce noeud est place 10.34 cm plus
    loin que le bout de la chaine DH (cf. le Transform du toolSlot), alors que
    move_to_pose commande le bout de la chaine. Les "54 mm d'ecart" n'etaient
    que la distance entre deux reperes distincts, et la correction suggeree
    aurait enfonce le bras dans la table.

    La position exacte des doigts n'est pas accessible simplement : un moteur
    Webots n'est pas un noeud Pose, donc getPosition() y est refuse. On se
    contente donc de rapporter, et c'est verifier_saisie() qui tranche.
    """
    cz = hauteur_cube()
    print(f"[Hauteur] commande (bout chaine DH) : z = {GRASP_Z:+.4f}")
    if cz is not None:
        print(f"[Hauteur] centre du cube           : z = {cz:+.4f}")


def verifier_saisie(z_avant):
    """
    Le cube a-t-il reellement ete souleve ?

    C'est le seul test sans ambiguite de la hauteur de saisie : on compare la
    hauteur du cube avant la fermeture de la pince et apres le levage.
    """
    z_apres = hauteur_cube()
    if z_avant is None or z_apres is None:
        print("[Saisie] Position du cube inaccessible, verification impossible.")
        return
    monte = z_apres - z_avant
    print(f"[Saisie] cube : z {z_avant:+.4f} -> {z_apres:+.4f}  "
          f"({1000 * monte:+.0f} mm)")
    if monte > 0.05:
        print("[Saisie] REUSSIE : le cube a suivi le bras.")
    elif monte > 0.005:
        print("[Saisie] PARTIELLE : le cube a bouge mais glisse. "
              f"Essayer GRASP_Z = {GRASP_Z - 0.005:.4f}")
    else:
        print("[Saisie] ECHOUEE : le cube n'a pas ete souleve. "
              f"Essayer GRASP_Z = {GRASP_Z - 0.010:.4f} (pince trop haute) "
              f"ou {GRASP_Z + 0.010:.4f} (pince trop basse, elle pousse le cube)")


def calibrer_hauteur_saisie(cx, cy):
    """
    Trouve la hauteur de saisie en essayant, plutot qu'en calculant.

    Pour chaque hauteur candidate : on remet le cube en place, on descend, on
    ferme, on leve, et on regarde si le cube a suivi. La premiere qui marche est
    la bonne. C'est lent (quelques essais) mais sans ambiguite -- contrairement
    aux deductions geometriques, qui se sont revelees fausses deux fois.

    Returns:
        la hauteur qui fonctionne, ou None si aucune n'a marche
    """
    node = ur5.bottle
    if node is None:
        print("[Calib] Cube introuvable, calibration impossible.")
        return None
    field = node.getField("translation")
    depart = field.getSFVec3f()

    print()
    print("=== CALIBRATION DE LA HAUTEUR DE SAISIE ===")
    for z in HAUTEURS_A_TESTER:
        # remettre le cube exactement a sa place
        field.setSFVec3f(depart)
        node.resetPhysics()
        for _ in range(10):
            ur5.supervisor.step(ur5.timestep)

        actuate_panda(close=False)
        ur5.move_to_pose([cx, cy, APPROACH_Z], [PI, 0, -PI / 2], wrist='up')
        ur5.move_to_pose([cx, cy, z], [PI, 0, -PI / 2], wrist='up', duration=3)
        z0 = hauteur_cube()
        actuate_panda(close=True)
        ur5.move_to_pose([cx, cy, APPROACH_Z], [PI, 0, -PI / 2], wrist='up')
        z1 = hauteur_cube()

        monte = (z1 - z0) if (z0 is not None and z1 is not None) else 0.0
        ok = monte > 0.05
        print(f"  z = {z:+.3f}  ->  cube monte de {1000 * monte:+6.0f} mm   "
              f"{'REUSSI' if ok else 'echec'}")
        if ok:
            print(f"  -> hauteur retenue : GRASP_Z = {z:.3f}")
            print(f"     (recopier dans GRASP_Z et mettre CALIBRER_HAUTEUR = False)")
            field.setSFVec3f(depart)
            node.resetPhysics()
            actuate_panda(close=False)
            for _ in range(10):
                ur5.supervisor.step(ur5.timestep)
            return z

    print("  AUCUNE hauteur n'a fonctionne.")
    print("  -> elargir HAUTEURS_A_TESTER, ou verifier l'ouverture de la pince.")
    field.setSFVec3f(depart)
    node.resetPhysics()
    return None


def pose_reelle():
    """
    Position monde du repere 6, telle que le moteur physique la donne.

    Ce n'est pas le bout de la chaine DH (le noeud est 10,34 cm plus loin),
    mais peu importe : on ne s'en sert que pour COMPARER deux poses, et le
    decalage s'annule dans la difference.
    """
    n = ur5.supervisor.getFromDef("frame6")
    return None if n is None else np.array(n.getPosition())


def mesurer_erreur_pinn(cibles):
    """
    Erreur du PINN mesuree sur le robot, pas sur son modele.

    Pour chaque cible : on y va avec l'IK analytique, on releve la pose reelle,
    puis on refait le trajet avec le PINN et on releve a nouveau. L'IK
    analytique etant une forme fermee exacte, l'ecart entre les deux poses est
    l'erreur du PINN telle que le robot la realise.

    On mesure aussi la repetabilite du bras (meme cible, meme solveur, deux
    fois) : sans elle on ne saurait pas si un ecart vient du reseau ou du
    controle.
    """
    print()
    print("=== ERREUR REELLE DU PINN (mesuree dans Webots) ===")
    if pose_reelle() is None:
        print("  DEF frame6 introuvable, mesure impossible.")
        return

    neutre = [0, -PI / 3, PI / 2, -PI / 6, -PI / 2, 0]
    rot = [PI, 0, -PI / 2]
    resultats = []

    for nom, cible in cibles:
        # meme point de depart pour les deux essais, sinon la trajectoire
        # differe et on mesurerait autre chose que le solveur
        ur5.move_to_config(neutre)
        ur5.use_pinn = False
        ur5.move_to_pose(list(cible), rot, wrist='up')
        p_math = pose_reelle()

        ur5.move_to_config(neutre)
        ur5.use_pinn = False
        ur5.move_to_pose(list(cible), rot, wrist='up')
        p_repet = pose_reelle()          # repetabilite : meme solveur, 2e fois

        ur5.move_to_config(neutre)
        ur5.use_pinn = True
        ur5.move_to_pose(list(cible), rot, wrist='up')
        p_pinn = pose_reelle()

        err = np.linalg.norm(p_pinn - p_math) * 1000
        rep = np.linalg.norm(p_repet - p_math) * 1000
        resultats.append((nom, cible, err, rep))
        print(f"  {nom:18s} {str(tuple(round(v, 3) for v in cible)):24s} "
              f"erreur {err:6.2f} mm   (repetabilite {rep:.2f} mm)")

    if resultats:
        e = np.array([r[2] for r in resultats])
        r = np.array([r[3] for r in resultats])
        print()
        print(f"  erreur PINN     : moyenne {e.mean():.2f} mm, max {e.max():.2f} mm")
        print(f"  repetabilite    : moyenne {r.mean():.2f} mm")
        print("  -> la repetabilite est le plancher : une erreur du meme ordre")
        print("     ne serait pas attribuable au reseau.")
    # Les 12 deplacements de la campagne ne font pas partie du scenario :
    # on remet les compteurs a zero pour que le bilan final reste lisible.
    ur5.n_pinn = 0
    ur5.n_analytique = 0
    ur5.use_pinn = (mode == "pinn")


def actuate_panda(close=False):
    """
    Ouvre ou ferme la PandaHand, en attendant la duree reellement necessaire.

    L'ancienne version attendait 20 pas de simulation. Or un pas ne dure pas
    toujours pareil : avec basicTimeStep a 8 ms, 20 pas ne font que 0.16 s pour
    une course qui en demande 0.40 s -- le bras repartait la pince encore
    ouverte. On calcule donc l'attente a partir de la course et de la vitesse,
    ce qui reste correct quel que soit le basicTimeStep du monde.
    """
    target = 0.0 if close else FINGER_OPEN
    if panda_f1: panda_f1.setPosition(target)
    if panda_f2: panda_f2.setPosition(target)
    duree = FINGER_OPEN / FINGER_SPEED * 1.5      # course complete + 50 % de marge
    t0 = ur5.supervisor.getTime()
    while ur5.supervisor.getTime() - t0 < duree:
        ur5.supervisor.step(ur5.timestep)

    if panda_s1 is not None and panda_s2 is not None:
        o1, o2 = panda_s1.getValue(), panda_s2.getValue()
        etat = "fermeture" if close else "ouverture"
        print(f"[Pince] {etat} -> consigne {target:.3f}, doigts a "
              f"{o1:.4f} / {o2:.4f} m  (ecartement total {1000 * (o1 + o2):.0f} mm)")
        if close:
            if o1 + o2 > 0.07:
                print("[Pince] Les doigts n'ont PAS bouge -> moteur non pilote.")
            elif o1 + o2 < 0.005:
                print("[Pince] Les doigts se sont fermes A VIDE -> le cube "
                      "n'etait pas entre eux.")
            else:
                # Les capteurs donnent la COURSE de chaque doigt, pas l'ecart
                # entre leurs faces internes : celles-ci sont en retrait
                # d'environ 7,4 mm chacune (mesure sur un cube de 50 mm serre
                # avec une somme de course de 65 mm). On retranche donc cette
                # epaisseur pour annoncer une largeur d'objet exploitable.
                largeur = 1000 * (o1 + o2) - 2 * EPAISSEUR_DOIGT_MM
                print(f"[Pince] Les doigts serrent un objet de "
                      f"{largeur:.0f} mm (course totale {1000 * (o1 + o2):.0f} mm).")

# ---------------------------------------------------------
# POSE DE LECTURE CAMERA
# ---------------------------------------------------------
# La calibration pixel -> monde n'est valable qu'a la pose exacte ou elle a ete
# mesuree. Si la vision est active, on reprend donc la pose enregistree par
# data_collector dans dataset/calibration.json plutot qu'une valeur en dur.
READ_POS = [-0.1, -0.68, 0.45]
READ_ROT = [PI, 0, -PI / 2]

if VISION_MODE in ("color", "cnn"):
    try:
        _c = ur5.vision_calibration()
        READ_POS = _c['reading_pose']['position']
        READ_ROT = _c['reading_pose']['orientation']
        _z = _c['visible_zone_world']
        print(f"[Vision] Pose de lecture calibree : {[round(v, 3) for v in READ_POS]}")
        print(f"[Vision] Zone visible monde : x[{_z['x_min']:.2f},{_z['x_max']:.2f}] "
              f"y[{_z['y_min']:.2f},{_z['y_max']:.2f}]")
    except FileNotFoundError as e:
        print(f"[Vision] {e}")
        print("[Vision] Repli sur le superviseur.")
        VISION_MODE = "supervisor"

# Position initiale stable
ur5.move_to_config([0, 0, 0, 0, 0, 0])

# ============================================================
# Pose de lecture camera
# ============================================================
# La pose de lecture reste en IK analytique, deliberement.
# La calibration pixel -> monde n'est valable qu'a la pose EXACTE ou elle a ete
# mesuree ; le PINN y ferait 3,2 mm d'erreur (ce point est hors de sa zone
# d'entrainement en x), ce qui decalerait la camera et fausserait la vision.
# Le PINN prend le relais pour la saisie, ou il est chez lui : 0,30 mm.
ur5.use_pinn = False
actuate_panda(close=False)
ur5.move_to_pose(READ_POS, READ_ROT, wrist='up')

for _ in range(30):
    ur5.supervisor.step(ur5.timestep)

# ============================================================
# Detection des positions
# ============================================================
cx, cy = None, None

# 1. Detection par la camera, selon le mode choisi
if VISION_MODE in ("color", "cnn"):
    try:
        if VISION_MODE == "color":
            found = ur5.detect_cube_color()
        else:
            found = ur5.predict_bottle_position() if ur5.model is not None else None
            if found is None:
                print("[Vision] Modele VGG16 indisponible.")
        if found is not None:
            cx, cy = found
            print(f"📷 Cube localise par la camera ({VISION_MODE}) : "
                  f"({cx:.3f}, {cy:.3f})")
    except Exception as e:
        print(f"[Vision] Echec ({type(e).__name__}: {e}) -> repli superviseur.")

# 2. Sinon, chercher l'objet selon le monde (mono-robot ou duo-robot)
if cx is None:
    # Monde duo (pinn_vs_math.wbt)
    box_def = "RED_BOX_PINN" if mode == "pinn" else "RED_BOX_MATH"
    box_frame = ur5.get_node_frame(box_def)
    
    # Monde mono-robot classique (my_first_simulation_pandahand.wbt)
    if box_frame is None:
        box_frame = ur5.get_bottle_frame()
        
    if box_frame is not None:
        cx, cy = box_frame[0, 3], box_frame[1, 3]
    else:
        cx, cy = 0.73, 0.2

# Detection du plateau bleu
tray_def = "BLUE_TRAY_PINN" if mode == "pinn" else "BLUE_TRAY_MATH"
tray_frame = ur5.get_node_frame(tray_def)
if tray_frame is None:
    tray_frame = ur5.get_node_frame("BLUE_TRAY")

if tray_frame is not None:
    tx, ty = tray_frame[0, 3], tray_frame[1, 3]
else:
    tx, ty = cx - 0.4, cy

print(f"Cube: ({cx:.3f}, {cy:.3f}) | Zone bleue: ({tx:.3f}, {ty:.3f})")
print("Lancement pick & place...")

# ============================================================
# MESURE (optionnelle) DE L'ERREUR REELLE DU PINN
# ============================================================
if MESURER_ERREUR_PINN and ur5.pinn_model is not None:
    mesurer_erreur_pinn([
        ("approche cube", (cx, cy, APPROACH_Z)),
        ("saisie cube", (cx, cy, GRASP_Z)),
        ("approche bac", (tx, ty, APPROACH_Z)),
        ("depose bac", (tx, ty, RELEASE_Z)),
    ])

# ============================================================
# PICK - Saisie top-down avec PandaHand
# ============================================================
if mode == "pinn":
    ur5.use_pinn = True  # On active ton IA PINN
else:
    ur5.use_pinn = False # On force les maths

if CALIBRER_HAUTEUR:
    trouve = calibrer_hauteur_saisie(cx, cy)
    if trouve is not None:
        GRASP_Z = trouve

ur5.move_to_pose([cx, cy, APPROACH_Z], [PI, 0, -PI/2], wrist='up')
ur5.move_to_pose([cx, cy, GRASP_Z], [PI, 0, -PI/2], wrist='up', duration=4)
diagnostic_hauteur()
_z_avant = hauteur_cube()        # reference pour verifier_saisie()
actuate_panda(close=True)

# ============================================================
# TRANSPORT
# ============================================================
ur5.move_to_pose([cx, cy, APPROACH_Z], [PI, 0, -PI/2], wrist='up')
verifier_saisie(_z_avant)        # le cube a-t-il suivi ?
ur5.move_to_pose([tx, ty, APPROACH_Z], [PI, 0, -PI/2], wrist='up')

# ============================================================
# PLACE
# ============================================================
ur5.move_to_pose([tx, ty, RELEASE_Z], [PI, 0, -PI/2], wrist='up', duration=4)
actuate_panda(close=False)
ur5.move_to_pose([tx, ty, APPROACH_Z], [PI, 0, -PI/2], wrist='up')

# ============================================================
# Retour pose de lecture
# ============================================================
ur5.use_pinn = False
ur5.move_to_pose(READ_POS, READ_ROT, wrist='up')

ur5.bilan_solveurs()
print("Termine !")

