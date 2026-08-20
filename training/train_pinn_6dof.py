"""
Ré-entraînement du réseau de neurones PINN sur la cinématique 6-DOF exacte du UR5e.
Utilise la perte physique vectorisée PyTorch (Physics-Informed Loss) pour une convergence ultra-rapide.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ur5e_6dof_ik import UR5e6DOF

# --- Destination des modeles : toujours pinn_ik_project/models/, jamais le
# --- repertoire courant. C'est la que ur5.py va chercher les poids.
MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)


def model_path(filename):
    return os.path.join(MODELS_DIR, filename)


class PINN6DOF(nn.Module):
    def __init__(self, input_dim=3, output_dim=6, hidden_dim=256):
        super(PINN6DOF, self).__init__()
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.register_buffer("mean_in", torch.zeros(input_dim))
        self.register_buffer("std_in", torch.ones(input_dim))
        self.register_buffer("mean_out", torch.zeros(output_dim))
        self.register_buffer("std_out", torch.ones(output_dim))

    def forward(self, x):
        x_norm = (x - self.mean_in) / self.std_in
        out_norm = self.net(x_norm)
        return out_norm * self.std_out + self.mean_out

    def save_model(self, filepath):
        checkpoint = {
            'state_dict': self.state_dict(),
            'hidden_dim': self.hidden_dim,
            'mean_in': self.mean_in,
            'std_in': self.std_in,
            'mean_out': self.mean_out,
            'std_out': self.std_out
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def from_file(cls, filepath):
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
        hidden_dim = checkpoint.get('hidden_dim', 256)
        model = cls(hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint['state_dict'])
        return model


def generate_dataset(ur_kin, num_samples=120000):
    """
    Génère un dataset (x, y, z) -> (q1..q6) dans la zone de travail utile de la table.
    """
    print(f"Génération de {num_samples} échantillons 6-DOF pour le poste de travail...")

    # Espace de travail de la table (repère robot, base z=0.45m)
    # x dans [0.35, 0.65], y dans [-0.25, 0.25], z dans [-0.25, 0.10]
    xs = np.random.uniform(0.35, 0.65, num_samples)
    ys = np.random.uniform(-0.25, 0.25, num_samples)
    zs = np.random.uniform(-0.25, 0.10, num_samples)

    targets = np.column_stack((xs, ys, zs))
    valid_targets = []
    valid_q = []

    for t in targets:
        q_sol, success, _ = ur_kin.solve_ik_downwards(t)
        if success:
            valid_targets.append(t)
            valid_q.append(q_sol)

    valid_targets = np.array(valid_targets, dtype=np.float32)
    valid_q = np.array(valid_q, dtype=np.float32)

    print(f"  -> {len(valid_targets)} échantillons 6-DOF valides générés.")
    return valid_targets, valid_q


def train():
    np.random.seed(42)
    torch.manual_seed(42)

    ur_kin = UR5e6DOF()
    X, y = generate_dataset(ur_kin, num_samples=120000)

    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_t, y_train_t = torch.tensor(X_train), torch.tensor(y_train)
    X_val_t, y_val_t = torch.tensor(X_val), torch.tensor(y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = PINN6DOF(hidden_dim=256)
    model.mean_in.copy_(X_train_t.mean(dim=0))
    model.std_in.copy_(X_train_t.std(dim=0).clamp(min=0.01))
    model.mean_out.copy_(y_train_t.mean(dim=0))
    model.std_out.copy_(y_train_t.std(dim=0).clamp(min=0.01))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
    mse = nn.MSELoss()

    w_data = 1.0
    w_phys = 20.0

    epochs = 60
    best_err = float('inf')

    print(f"\nEntraînement PINN 6-DOF : {epochs} époques sur {len(X_train)} échantillons...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            optimizer.zero_grad()

            pred_y = model(bx)
            loss_data = mse(pred_y, by)

            pred_pos = ur_kin.forward_kinematics_pytorch(pred_y)
            loss_phys = mse(pred_pos, bx)

            loss = w_data * loss_data + w_phys * loss_phys
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(bx)

        total_loss /= len(X_train)

        # Validation PyTorch vectorisée ultra-rapide
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_pos = ur_kin.forward_kinematics_pytorch(val_pred)
            dist_err_mm = torch.norm(val_pos - X_val_t, dim=1).numpy() * 1000.0

            mean_err = np.mean(dist_err_mm)
            p50 = np.percentile(dist_err_mm, 50)
            p95 = np.percentile(dist_err_mm, 95)

        scheduler.step(mean_err)
        lr = optimizer.param_groups[0]['lr']

        print(f"Ep {epoch+1:02d}/{epochs} | Loss: {total_loss:.5f} | "
              f"Err moy: {mean_err:.2f} mm  P50: {p50:.2f}  P95: {p95:.2f}  LR: {lr:.1e}")

        if mean_err < best_err:
            best_err = mean_err
            model.save_model(model_path("pinn_model_ur5e_6dof.pth"))

    print(f"\n✓ Meilleur Modèle PINN 6-DOF enregistré : Erreur = {best_err:.2f} mm -> models/pinn_model_ur5e_6dof.pth")


if __name__ == "__main__":
    train()
