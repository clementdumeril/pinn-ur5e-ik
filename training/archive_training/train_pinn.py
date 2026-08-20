import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from robot_model import UR5e3Axis
from pinn_model import PINNInverseKinematics


def generate_dataset(arm, num_samples=50000):
    """
    Genere un dataset (position_cible, angles_joints) pour le UR5e.

    Convention FK corrigee (z = d1 - a2*sin(q2) - a3*sin(q2+q3)) :
        q2 < 0  -> bras monte (z augmente)
        q2 > 0  -> bras descend (z diminue)
        q2 = 0  -> bras horizontal

    Plages restreintes a la zone de travail FRONTALE (single-valued IK) :
        q1 : [-pi/2, pi/2]   moitie avant (pas derriere le robot)
        q2 : [-2.5, 0.5]     du haut vers legerement sous l'horizontal
        q3 : [-pi, 0]        coude "up" uniquement (branche unique)
    """
    print(f"Generation de {num_samples} echantillons UR5e...")

    q1 = np.random.uniform(-np.pi / 2, np.pi / 2, num_samples)
    q2 = np.random.uniform(-2.5, 0.5, num_samples)
    q3 = np.random.uniform(-np.pi, 0.0, num_samples)

    thetas = np.column_stack((q1, q2, q3))

    # FK vectorisee pour rapidite
    positions = np.array([arm.forward_kinematics_numpy(t)[3] for t in thetas])

    # Filtrer : effecteur au-dessus du sol (z > -0.2 dans repere robot)
    # et devant le robot (x > 0) pour rester dans la zone de travail
    valid = (positions[:, 2] > -0.2) & (positions[:, 0] > -0.1)
    positions = positions[valid]
    thetas = thetas[valid]
    print(f"  -> {len(positions)} echantillons valides.")

    return positions.astype(np.float32), thetas.astype(np.float32)


def train():
    np.random.seed(42)
    torch.manual_seed(42)

    arm = UR5e3Axis()

    # Verification FK rapide
    p_horiz = arm.forward_kinematics_numpy([0, 0, 0])[3]
    p_up = arm.forward_kinematics_numpy([0, -np.pi/2, 0])[3]
    print(f"FK check: q=[0,0,0] -> {[round(x,3) for x in p_horiz]}  (attendu: ~[0.817, 0, 0.163])")
    print(f"FK check: q=[0,-pi/2,0] -> {[round(x,3) for x in p_up]}  (attendu: ~[0, 0, 0.980])")

    # Generer le dataset
    positions, thetas = generate_dataset(arm, num_samples=150000)

    # Separation train/val
    split = int(0.8 * len(positions))
    X_train, X_val = positions[:split], positions[split:]
    y_train, y_val = thetas[:split], thetas[split:]

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # Modele PINN (256 neurones caches)
    model = PINNInverseKinematics(hidden_dim=256)

    # Normalisation basee sur les donnees d'entrainement
    model.mean_in.copy_(X_train_t.mean(dim=0))
    model.std_in.copy_(X_train_t.std(dim=0).clamp(min=0.01))
    model.mean_out.copy_(y_train_t.mean(dim=0))
    model.std_out.copy_(y_train_t.std(dim=0).clamp(min=0.01))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    mse_loss = nn.MSELoss()

    w_data = 1.0
    w_phys = 25.0   # poids physique eleve

    num_epochs = 80
    best_error = float('inf')
    patience_counter = 0
    print(f"\nEntrainement PINN UR5e : {num_epochs} epoques, {len(X_train)} samples...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            pred_y = model(batch_x)
            loss_data = mse_loss(pred_y, batch_y)

            pred_pos = arm.forward_kinematics_pytorch(pred_y)
            loss_phys = mse_loss(pred_pos, batch_x)

            loss = w_data * loss_data + w_phys * loss_phys
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        total_loss /= len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_pos = arm.forward_kinematics_pytorch(val_pred)
            dist_err = torch.norm(val_pos - X_val_t, dim=1)
            mean_err_mm = dist_err.mean().item() * 1000.0
            max_err_mm = dist_err.max().item() * 1000.0
            p50 = torch.quantile(dist_err, 0.5).item() * 1000.0
            p95 = torch.quantile(dist_err, 0.95).item() * 1000.0

        scheduler.step(mean_err_mm)
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Ep {epoch+1:02d}/{num_epochs} | Loss: {total_loss:.5f} | "
              f"Err moy: {mean_err_mm:.1f} mm  P50: {p50:.1f}  P95: {p95:.1f}  Max: {max_err_mm:.1f}  "
              f"LR: {lr_now:.1e}")

        if mean_err_mm < best_error:
            best_error = mean_err_mm
            model.save_model("pinn_model_ur5e.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter > 15:
            print(f"Early stopping a l'epoque {epoch+1}")
            break

    print(f"\nMeilleur modele : erreur = {best_error:.2f} mm")
    print(f"Fichier : 'pinn_model_ur5e.pth'")


if __name__ == "__main__":
    train()
