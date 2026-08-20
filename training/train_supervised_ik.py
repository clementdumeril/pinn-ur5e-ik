import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math

# Mock du Supervisor de Webots pour pouvoir importer ur5.py sans crasher
sys.modules['controller'] = type('controller', (), {'Supervisor': type('Supervisor', (), {})})

# Ajouter les chemins vers ur5.py (reference_ur5_repo/) et l'architecture du
# reseau (training/). __file__ est dans training/, il faut donc remonter d'un cran.
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in [os.path.join(base_dir, 'reference_ur5_repo'),
           os.path.join(base_dir, 'robotics_utils'),
           os.path.dirname(os.path.abspath(__file__))]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ur5 import build_matrix, inverse_kinematics
from train_pinn_6dof import PINN6DOF

# --- Destination des modeles : toujours pinn_ik_project/models/, jamais le
# --- repertoire courant. C'est la que ur5.py va chercher les poids.
MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)


def model_path(filename):
    return os.path.join(MODELS_DIR, filename)


def generate_webots_dataset(num_samples=100000):
    print(f"Génération de {num_samples} échantillons pour la vraie table Webots...")
    
    # La table Webots est située à :
    # X autour de 0.0 à 0.4 (dans le repère du robot)
    # Y autour de -0.9 à -0.5
    # Z hauteur d'attrape autour de 0.05 à 0.45
    xs = np.random.uniform(0.0, 0.4, num_samples)
    ys = np.random.uniform(-0.9, -0.5, num_samples)
    zs = np.random.uniform(0.05, 0.45, num_samples)
    
    targets = np.column_stack((xs, ys, zs))
    valid_targets = []
    valid_q = []
    
    for i, t in enumerate(targets):
        if i % 5000 == 0:
            print(f"  Progression : {i}/{num_samples}...", flush=True)
        try:
            # Orientation "down" stricte (pince vers le bas)
            T = build_matrix(t, [math.pi, 0, -math.pi/2], euler='XYZ')
            # Calcul via la vraie formule mathématique de Webots !
            q_sol = inverse_kinematics(T, wrist='up', shoulder='left', elbow='up')
            
            # Si on obtient un résultat valide sans erreur
            if not np.any(np.isnan(q_sol)):
                valid_targets.append(t)
                valid_q.append(q_sol)
        except Exception:
            pass # Pose impossible physiquement

    valid_targets = np.array(valid_targets, dtype=np.float32)
    valid_q = np.array(valid_q, dtype=np.float32)

    print(f"  -> {len(valid_targets)} échantillons valides générés.")
    return valid_targets, valid_q

def train():
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Génération des données parfaites
    X, y = generate_webots_dataset(num_samples=25000)

    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_t, y_train_t = torch.tensor(X_train), torch.tensor(y_train)
    X_val_t, y_val_t = torch.tensor(X_val), torch.tensor(y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # 2. Création du PINN (On augmente la taille à 512 neurones pour plus de précision !)
    model = PINN6DOF(hidden_dim=512)
    model.mean_in.copy_(X_train_t.mean(dim=0))
    model.std_in.copy_(X_train_t.std(dim=0).clamp(min=0.01))
    model.mean_out.copy_(y_train_t.mean(dim=0))
    model.std_out.copy_(y_train_t.std(dim=0).clamp(min=0.01))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    mse = nn.MSELoss()

    epochs = 100
    best_err = float('inf')

    print(f"\nEntraînement PINN Webots : {epochs} époques sur {len(X_train)} échantillons...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            optimizer.zero_grad()
            
            # Ici on utilise une simple Data Loss car la Physics Loss nécessiterait
            # de coder la Forward Kinematics parfaite du UR5 dans PyTorch.
            # Avec un gros réseau (512) et beaucoup de données parfaites, ça suffira !
            pred_y = model(bx)
            loss = mse(pred_y, by)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        total_loss /= len(X_train)

        # Validation sur les angles
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            angle_mse = mse(val_pred, y_val_t).item()
            
        scheduler.step(angle_mse)
        lr = optimizer.param_groups[0]['lr']

        print(f"Ep {epoch+1:03d}/{epochs} | Loss (Angles): {total_loss:.5f} | Val MSE: {angle_mse:.5f} | LR: {lr:.1e}")

        if angle_mse < best_err:
            best_err = angle_mse
            model.save_model(model_path("pinn_model_webots.pth"))

    print(f"\n✓ Meilleur Modèle PINN Webots enregistré dans models/pinn_model_webots.pth")

if __name__ == "__main__":
    train()
