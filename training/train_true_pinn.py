import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math

# --- 1. MOCKING POUR WEBOTS ---
# Simulation des modules Webots pour pouvoir importer ur5.py
if 'controller' not in sys.modules:
    sys.modules['controller'] = type('controller', (), {'Supervisor': type('Supervisor', (), {})})
    
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
simulation_dir = os.path.join(base_dir, 'simulation_ur5') # Ou reference_ur5_repo
utils_dir = os.path.join(base_dir, 'robotics_utils')

sys.path.append(simulation_dir)
# Fallback au cas ou le renommage echoue a cause de Webots
sys.path.append(os.path.join(base_dir, 'reference_ur5_repo'))
sys.path.append(utils_dir)

from ur5 import build_matrix, inverse_kinematics
from train_pinn_6dof import PINN6DOF
from ur5_pytorch_fk import UR5ForwardKinematicsPyTorch

# --- Destination des modeles : toujours pinn_ik_project/models/, jamais le
# --- repertoire courant. C'est la que ur5.py va chercher les poids.
MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)


def model_path(filename):
    return os.path.join(MODELS_DIR, filename)


# --- 2. GENERATION DES DONNEES (SUPERVISION LEGERE) ---
def generate_webots_dataset(num_samples=25000):
    print(f"Génération de {num_samples} cibles pour la table Webots...")
    
    # Zone de la table Webots (là où se trouve le cube)
    xs = np.random.uniform(0.0, 0.4, num_samples)
    ys = np.random.uniform(-0.9, -0.5, num_samples)
    zs = np.random.uniform(0.05, 0.45, num_samples)
    
    targets = np.column_stack((xs, ys, zs))
    valid_targets = []
    valid_q = []
    
    for t in targets:
        try:
            # Orientation vers le bas [math.pi, 0, -math.pi/2]
            T = build_matrix(t, [math.pi, 0, -math.pi/2], euler='XYZ')
            # Famille de solutions "Coude en l'air, épaule gauche" pour éviter la table
            q_sol = inverse_kinematics(T, wrist='up', shoulder='left', elbow='up')
            
            if not np.any(np.isnan(q_sol)):
                valid_targets.append(t)
                valid_q.append(q_sol)
        except Exception:
            pass 

    valid_targets = np.array(valid_targets, dtype=np.float32)
    valid_q = np.array(valid_q, dtype=np.float32)
    print(f"-> {len(valid_targets)} échantillons générés.")
    return valid_targets, valid_q


def train():
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Dataset
    X, y = generate_webots_dataset(num_samples=25000)

    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 2. Modèle et normalisation
    model = PINN6DOF(hidden_dim=512)
    model.mean_in = X_train_t.mean(dim=0)
    model.std_in = X_train_t.std(dim=0)
    model.mean_out = y_train_t.mean(dim=0)
    model.std_out = y_train_t.std(dim=0)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    
    # 3. Le Moteur de Physique (Forward Kinematics en PyTorch)
    fk_engine = UR5ForwardKinematicsPyTorch()

    epochs = 100
    best_err = float('inf')

    print(f"\n--- ENTRAÎNEMENT DU VRAI PINN : {epochs} époques ---")
    print("Loss combinée = 0.1 * Data Loss (Famille de solution) + 1.0 * Physics Loss (Précision géométrique mm)\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            optimizer.zero_grad()
            
            # 1. L'IA devine les 6 angles
            pred_q = model(bx)
            
            # 2. SUPERVISION LEGERE : Aide l'IA à rester sur "Coude en l'air"
            loss_data = mse(pred_q, by)
            
            # 3. VERITABLE PINN : On passe les angles dans le simulateur PyTorch
            # pour voir où le bout du bras atterrit !
            pred_pos = fk_engine.forward_pos(pred_q)
            loss_phys = mse(pred_pos, bx)
            
            # 4. On additionne (La Physique est 10x plus importante)
            loss = 0.1 * loss_data + 1.0 * loss_phys
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_pos = fk_engine.forward_pos(val_pred)
            
            # Erreur spatiale absolue en millimètres (MSE de la Physics Loss pure)
            dist_err_mm = torch.mean(torch.norm(val_pos - X_val_t, dim=1)).numpy() * 1000.0
            
            angle_mse = mse(val_pred, y_val_t).item()

        # Decay du Learning Rate
        if epoch == 30:
            for g in optimizer.param_groups: g['lr'] = 5e-4
        if epoch == 60:
            for g in optimizer.param_groups: g['lr'] = 1e-4
        if epoch == 80:
            for g in optimizer.param_groups: g['lr'] = 2e-5

        lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch+1:03d}/{epochs} | Physics Error: {dist_err_mm:.3f} mm | Data Loss (Angles): {angle_mse:.5f} | LR: {lr:.1e}")

        if dist_err_mm < best_err:
            best_err = dist_err_mm
            model.save_model(model_path("pinn_model_true_physics.pth"))

    print(f"\n[OK] Meilleur VRAI PINN enregistré dans models/pinn_model_true_physics.pth (Précision: {best_err:.3f} mm)")

if __name__ == "__main__":
    train()
