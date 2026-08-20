"""
Controleur VISION (camera fixe au-dessus de la table, PAS sur le bras).

Role : reconnaitre le cube rouge par la couleur, estimer sa position (x,y) dans
le monde, et l'ecrire dans un fichier partage lu par le controleur du bras.

Pipeline :
  1. Auto-calibration : on place le cube a 4 positions connues (superviseur),
     on lit le pixel du centre rouge, et on ajuste une transformation affine
     pixel -> monde (absorbe l'orientation/FOV de la camera, aucune geometrie a coder).
  2. Boucle : detection couleur -> pixel -> monde -> ecriture detection.json.

Le bras n'utilise PAS la position "verite" du superviseur : il lit ce que la
VISION a estime. (La verite n'est lue ici que pour afficher l'erreur de detection.)
"""

import os
import json
import numpy as np
from controller import Supervisor

HERE = os.path.dirname(__file__)
SHARED = os.path.join(HERE, "..", "detection.json")   # lu par le controleur du bras
CUBE_Z = 0.32


def main():
    robot = Supervisor()
    ts = int(robot.getBasicTimeStep())
    cam = robot.getDevice("overhead_camera")
    cam.enable(ts)
    W, H = cam.getWidth(), cam.getHeight()
    box = robot.getFromDef("RED_BOX").getField("translation")

    def step(n=1):
        for _ in range(n):
            if robot.step(ts) == -1:
                return False
        return True

    def detect_pixel():
        """Centre (u,v) du blob rouge dans l'image, ou None."""
        raw = cam.getImage()
        if raw is None:
            return None, 0
        arr = np.frombuffer(raw, np.uint8).reshape(H, W, 4).astype(np.int32)
        b, g, r = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        mask = (r > 120) & (g < 90) & (b < 90)
        n = int(mask.sum())
        if n < 5:
            return None, n
        ys, xs = np.where(mask)
        return np.array([xs.mean(), ys.mean()]), n

    def write(d):
        json.dump(d, open(SHARED, "w"))

    write({"calibrated": False})
    step(5)

    # ---- Auto-calibration pixel -> monde ----
    cal_pts = [(0.44, -0.05), (0.58, -0.05), (0.44, 0.18), (0.58, 0.18)]
    pix, wld = [], []
    print("[VISION] Auto-calibration de la camera...")
    for (x, y) in cal_pts:
        box.setSFVec3f([x, y, CUBE_Z])
        step(12)
        px, n = detect_pixel()
        if px is not None:
            pix.append(px)
            wld.append([x, y])
            print(f"  point monde ({x:.2f},{y:.2f}) -> pixel ({px[0]:.0f},{px[1]:.0f}) [{n} px rouges]")
        else:
            print(f"  point monde ({x:.2f},{y:.2f}) -> AUCUN rouge detecte ({n} px)")

    if len(pix) < 3:
        print("[VISION] ECHEC calibration (cube non vu). Verifier la camera.")
        write({"calibrated": False})
        while step():
            pass
        return

    A = np.hstack([np.array(pix), np.ones((len(pix), 1))])
    coef, *_ = np.linalg.lstsq(A, np.array(wld), rcond=None)   # (3,2)
    box.setSFVec3f([0.50, 0.10, CUBE_Z])                        # restaurer le cube
    print("[VISION] Calibration OK.")
    write({"calibrated": True, "ok": False})

    def pixel_to_world(px):
        return np.array([px[0], px[1], 1.0]) @ coef

    # ---- Boucle de reconnaissance ----
    k = 0
    while step():
        px, n = detect_pixel()
        if px is not None:
            wx, wy = pixel_to_world(px)
            write({"calibrated": True, "ok": True, "x": float(wx), "y": float(wy)})
            k += 1
            if k % 40 == 0:
                true = box.getSFVec3f()
                err = np.hypot(wx - true[0], wy - true[1]) * 1000
                print(f"[VISION] cube estime ({wx:.3f},{wy:.3f}) | vrai ({true[0]:.3f},{true[1]:.3f}) "
                      f"| erreur {err:.0f} mm")
        else:
            write({"calibrated": True, "ok": False})


if __name__ == "__main__":
    main()
