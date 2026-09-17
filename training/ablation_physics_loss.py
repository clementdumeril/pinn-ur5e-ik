"""
Ablation : la perte physique sert-elle vraiment a quelque chose ?

Le README affirmait que le PINN leve l'ambiguite des 8 solutions de la
cinematique inverse, la ou "un reseau supervise moyenne ces solutions et
produit des configurations invalides".

Cette affirmation n'etait pas testee. Les deux generateurs de donnees du
projet filtrent sur UNE SEULE branche avant l'entrainement :

    inverse_kinematics(T, wrist='up', shoulder='left', elbow='up')

L'ambiguite etait donc supprimee par la GENERATION DES DONNEES, pas par la
perte physique. Un reseau purement supervise sur ces memes donnees n'a rien
a moyenner : l'experience ne pouvait pas demontrer ce que le texte annoncait.

Ce script teste l'hypothese pour de bon, sur un plan 2 x 3 :

    branche  : 'single' (une seule famille) ou 'mixed' (les 8 melangees)
    w_phys   : 0 (supervise pur), 1 (le PINN du depot), 20 (physique dominante)

Tout le reste est tenu constant -- meme graine, meme architecture, memes
cibles, memes epoques, meme planning de pas, meme critere de selection.
La seule variable est celle qu'on etudie.

Les trois issues sont publiables, y compris "la physique n'aide nulle part".
"""
import os
import sys
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Mock Webots, pour importer ur5.py hors simulation ---------------------
if 'controller' not in sys.modules:
    sys.modules['controller'] = type(
        'controller', (), {'Supervisor': type('Supervisor', (), {})})

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in [os.path.join(base_dir, 'reference_ur5_repo'),
           os.path.join(base_dir, 'robotics_utils'),
           os.path.dirname(os.path.abspath(__file__))]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ur5 import build_matrix, inverse_kinematics          # noqa: E402
from train_pinn_6dof import PINN6DOF                      # noqa: E402
from ur5_pytorch_fk import UR5ForwardKinematicsPyTorch    # noqa: E402

ROT_DOWN = [math.pi, 0.0, -math.pi / 2]
R_OUTIL = 0.05
N_TARGETS = int(os.environ.get('ABL_TARGETS', 25000))
EPOCHS = int(os.environ.get('ABL_EPOCHS', 100))

BRANCHES = [(s, w, e)
            for s in ('left', 'right')
            for w in ('up', 'down')
            for e in ('up', 'down')]          # 2 x 2 x 2 = 8


# ===========================================================================
# 1. Donnees
# ===========================================================================
def build_dataset(n_targets, seed=42):
    """
    Pour chaque cible, resout les 8 branches.

    On ne garde que les cibles ou la branche de reference ET au moins une
    autre branche existent : les deux bras de l'ablation partagent ainsi
    EXACTEMENT les memes cibles. Sans cela on comparerait deux problemes.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, 0.4, n_targets)
    ys = rng.uniform(-0.9, -0.5, n_targets)
    zs = rng.uniform(0.05, 0.45, n_targets)
    targets = np.column_stack((xs, ys, zs))

    fk_check = UR5ForwardKinematicsPyTorch()
    X, q_single, q_mixed = [], [], []
    n_branches_vues = []

    for t in targets:
        T = build_matrix(t, ROT_DOWN, euler='XYZ')
        sols = []
        q_ref = None
        for (sh, wr, el) in BRANCHES:
            try:
                q = inverse_kinematics(T, shoulder=sh, wrist=wr, elbow=el)
            except Exception:
                continue
            if q is None or np.any(np.isnan(q)):
                continue
            q = np.asarray(q, dtype=np.float32)

            # Ne PAS se fier a l'absence de NaN : pour certaines combinaisons
            # le solveur renvoie des angles finis qui n'atteignent pas la
            # cible. On verifie chaque branche par cinematique directe --
            # c'est la definition meme d'une solution.
            with torch.no_grad():
                p = fk_check.forward_pos(torch.tensor(q).unsqueeze(0))[0]
            if torch.norm(p - torch.tensor(t, dtype=torch.float32)).item() > 1e-4:
                continue

            sols.append(q)
            if (sh, wr, el) == ('left', 'up', 'up'):
                q_ref = q

        if q_ref is None or len(sols) < 2:
            continue

        X.append(t)
        q_single.append(q_ref)
        q_mixed.append(sols[rng.integers(len(sols))])
        n_branches_vues.append(len(sols))

    X = np.asarray(X, dtype=np.float32)
    return (X,
            np.asarray(q_single, dtype=np.float32),
            np.asarray(q_mixed, dtype=np.float32),
            float(np.mean(n_branches_vues)))


def verifier_branches(X, q_single, q_mixed, fk):
    """
    Controle de l'instrument avant de s'en servir.

    Les branches melangees doivent atteindre la MEME pose que la branche de
    reference -- c'est toute la definition d'une solution de cinematique
    inverse. Si ce n'etait pas le cas, le bras 'mixed' testerait autre chose
    que ce qu'on croit, et tout le reste serait sans valeur.
    """
    with torch.no_grad():
        p_s = fk.forward_pos(torch.tensor(q_single[:2000]))
        p_m = fk.forward_pos(torch.tensor(q_mixed[:2000]))
        cible = torch.tensor(X[:2000])
        e_s = torch.norm(p_s - cible, dim=1).max().item() * 1000
        e_m = torch.norm(p_m - cible, dim=1).max().item() * 1000
        diff = torch.norm(torch.tensor(q_single[:2000])
                          - torch.tensor(q_mixed[:2000]), dim=1).mean().item()
    print(f"  [verif] ecart max FK(branche unique) -> cible : {e_s:.4f} mm")
    print(f"  [verif] ecart max FK(branches melangees) -> cible : {e_m:.4f} mm")
    print(f"  [verif] distance angulaire moyenne entre les deux : {diff:.3f} rad")
    if max(e_s, e_m) > 1.0:
        print("  [verif] ECHEC : les branches n'atteignent pas la cible.")
        sys.exit(1)
    print("  [verif] OK : les deux jeux d'angles atteignent la meme pose.\n")


# ===========================================================================
# 2. Un bras de l'ablation
# ===========================================================================
def run(nom, X, y, w_phys, fk, epochs=EPOCHS, w_data=0.1, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    split = int(0.9 * len(X))
    Xtr, Xva = torch.tensor(X[:split]), torch.tensor(X[split:])
    ytr, yva = torch.tensor(y[:split]), torch.tensor(y[split:])

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)

    model = PINN6DOF(hidden_dim=512)
    model.mean_in.copy_(Xtr.mean(dim=0))
    model.std_in.copy_(Xtr.std(dim=0).clamp(min=1e-3))
    model.mean_out.copy_(ytr.mean(dim=0))
    model.std_out.copy_(ytr.std(dim=0).clamp(min=1e-3))

    opt = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    best = {'pos_mm': float('inf'), 'rot_deg': float('nan'), 'epoch': -1}
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            pred_q = model(bx)
            loss = w_data * mse(pred_q, by)
            if w_phys > 0:
                T_pred = fk.forward(pred_q)
                T_ref = fk.forward(by)
                loss_pos = mse(T_pred[:, :3, 3], bx)
                loss_rot = mse(T_pred[:, :3, :3], T_ref[:, :3, :3])
                loss = loss + w_phys * (loss_pos + (R_OUTIL ** 2) * loss_rot)
            loss.backward()
            opt.step()

        # --- meme planning de pas pour tous les bras ---
        if epoch == 30:
            for g in opt.param_groups:
                g['lr'] = 5e-4
        if epoch == 60:
            for g in opt.param_groups:
                g['lr'] = 1e-4
        if epoch == 80:
            for g in opt.param_groups:
                g['lr'] = 2e-5

        # --- meme mesure pour tous les bras, en unites physiques ---
        model.eval()
        with torch.no_grad():
            pred = model(Xva)
            T_val = fk.forward(pred)
            T_ref = fk.forward(yva)
            pos_mm = torch.norm(T_val[:, :3, 3] - Xva, dim=1).mean().item() * 1000
            R_res = torch.bmm(T_ref[:, :3, :3].transpose(1, 2), T_val[:, :3, :3])
            cos = ((R_res[:, 0, 0] + R_res[:, 1, 1] + R_res[:, 2, 2]) - 1.0) / 2.0
            rot_deg = torch.rad2deg(
                torch.acos(cos.clamp(-1.0, 1.0))).mean().item()

        if pos_mm < best['pos_mm']:
            best = {'pos_mm': pos_mm, 'rot_deg': rot_deg, 'epoch': epoch + 1}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    ep {epoch+1:03d} | {pos_mm:9.3f} mm | {rot_deg:7.3f} deg",
                  flush=True)

    return {'nom': nom, 'w_phys': w_phys,
            'best_pos_mm': best['pos_mm'], 'best_rot_deg': best['rot_deg'],
            'best_epoch': best['epoch'],
            'final_pos_mm': pos_mm, 'final_rot_deg': rot_deg,
            'secondes': round(time.time() - t0, 1)}


# ===========================================================================
# 3. Campagne
# ===========================================================================
def main():
    print("=" * 74)
    print("ABLATION : la perte physique change-t-elle quelque chose ?")
    print("=" * 74)
    print(f"cibles demandees : {N_TARGETS}   epoques : {EPOCHS}\n")

    fk = UR5ForwardKinematicsPyTorch()

    print("[1/3] Generation du jeu de donnees (8 branches par cible)...")
    t0 = time.time()
    X, q_single, q_mixed, moy_branches = build_dataset(N_TARGETS)
    print(f"  -> {len(X)} cibles retenues "
          f"({moy_branches:.2f} branches valides par cible en moyenne) "
          f"en {time.time()-t0:.0f} s\n")

    print("[2/3] Verification de l'instrument...")
    verifier_branches(X, q_single, q_mixed, fk)

    print("[3/3] Entrainements\n")
    resultats = []
    for branche, y in (('branche unique', q_single), ('8 branches melangees', q_mixed)):
        for w_phys in (0.0, 1.0, 20.0):
            nom = f"{branche} | w_phys={w_phys:g}"
            print(f"  --- {nom} ---", flush=True)
            r = run(nom, X, y, w_phys, fk)
            r['branche'] = branche
            resultats.append(r)
            print(f"    => meilleur : {r['best_pos_mm']:.3f} mm  "
                  f"{r['best_rot_deg']:.3f} deg  "
                  f"(ep {r['best_epoch']}, {r['secondes']:.0f} s)\n", flush=True)

    print("=" * 74)
    print(f"{'Configuration':<34} {'Position':>12} {'Orientation':>13}")
    print("-" * 74)
    for r in resultats:
        print(f"{r['nom']:<34} {r['best_pos_mm']:>9.3f} mm {r['best_rot_deg']:>10.3f} deg")
    print("=" * 74)

    out = os.path.join(base_dir, 'models', 'ablation_physics_loss.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'n_cibles': len(X), 'epoques': EPOCHS,
                   'branches_moyennes_par_cible': moy_branches,
                   'resultats': resultats}, f, indent=2)
    print(f"\nResultats bruts : {out}")


if __name__ == "__main__":
    main()
