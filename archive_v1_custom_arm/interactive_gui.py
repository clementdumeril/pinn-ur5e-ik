import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from robot_model import AnthropomorphicArm3D
from pinn_model import PINNInverseKinematics

def main():
    # 1. Vérifier si le modèle PINN est déjà entraîné
    model_path = "pinn_model.pth"
    if not os.path.exists(model_path):
        print("Erreur : Le modèle PINN entraîné 'pinn_model.pth' n'a pas été trouvé !")
        print("Veuillez d'abord exécuter le script d'entraînement : python train_pinn.py")
        return

    # 2. Charger le robot et le modèle PINN
    arm = AnthropomorphicArm3D()
    pinn = PINNInverseKinematics()
    pinn.load_model(model_path)
    pinn.eval()

    # 3. Initialiser la figure interactive
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title("Simulateur de Cinématique Inverse 3D (PINN vs DLS)")
    
    # Espace 3D pour le robot (occupant le haut)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.35)  # Laisser de la place pour les widgets

    # Limites du graphique
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([0.0, 3.0])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid(True)

    # Initialisation des éléments de tracé
    # Bras PINN (Bleu)
    line_pinn, = ax.plot([], [], [], 'b-o', linewidth=4, markersize=8, label="IA (PINN)")
    # Bras Numérique DLS (Rouge)
    line_dls, = ax.plot([], [], [], 'r--o', linewidth=3, markersize=6, label="Solveur DLS")
    
    # Cible (Étoile Verte)
    target_point, = ax.plot([], [], [], 'g*', markersize=15, label="Cible")
    
    ax.legend(loc="upper left")

    # 4. Ajout des Sliders pour contrôler la position X, Y, Z de la cible
    ax_x = plt.axes([0.15, 0.22, 0.65, 0.03])
    ax_y = plt.axes([0.15, 0.17, 0.65, 0.03])
    ax_z = plt.axes([0.15, 0.12, 0.65, 0.03])

    slider_x = Slider(ax_x, 'Cible X (m)', -2.0, 2.0, valinit=1.2)
    slider_y = Slider(ax_y, 'Cible Y (m)', -2.0, 2.0, valinit=1.0)
    slider_z = Slider(ax_z, 'Cible Z (m)',  0.1, 2.8, valinit=1.5)

    # Ajout de boutons radio pour le mode d'affichage
    ax_mode = plt.axes([0.15, 0.02, 0.25, 0.08])
    radio_mode = RadioButtons(ax_mode, ('IA (PINN) uniquement', 'Solveur DLS uniquement', 'Comparer les deux'), active=2)

    # Titre dynamique pour afficher les erreurs
    title_text = ax.set_title("Initialisation...", fontsize=12, fontweight='bold')

    # 5. Fonction de mise à jour lors de l'interaction
    def update(val):
        # Récupérer les valeurs des sliders
        tgt_x = slider_x.val
        tgt_y = slider_y.val
        tgt_z = slider_z.val
        target_pos = np.array([tgt_x, tgt_y, tgt_z])
        
        mode = radio_mode.value_selected
        
        # Mettre à jour l'affichage de la cible verte
        target_point.set_data([tgt_x], [tgt_y])
        target_point.set_3d_properties([tgt_z])

        # --- A. Résolution par l'IA (PINN) ---
        target_tensor = torch.tensor([[tgt_x, tgt_y, tgt_z]], dtype=torch.float32)
        with torch.no_grad():
            theta_pinn = pinn(target_tensor).numpy()[0]
            
        p0_p, p1_p, p2_p, p3_p = arm.forward_kinematics_numpy(theta_pinn)
        err_pinn = np.linalg.norm(p3_p - target_pos) * 1000.0 # en mm

        # --- B. Résolution par le Solveur Numérique (DLS) ---
        # On utilise une estimation initiale neutre
        theta0 = np.array([0.0, 0.5, 0.0])
        theta_dls, success_dls = arm.inverse_kinematics_numerical(target_pos, theta0)
        p0_d, p1_d, p2_d, p3_d = arm.forward_kinematics_numpy(theta_dls)
        err_dls = np.linalg.norm(p3_d - target_pos) * 1000.0 # en mm

        # --- C. Gestion de l'affichage selon le mode ---
        if mode == 'IA (PINN) uniquement':
            # Afficher PINN
            line_pinn.set_data([p0_p[0], p1_p[0], p2_p[0], p3_p[0]], [p0_p[1], p1_p[1], p2_p[1], p3_p[1]])
            line_pinn.set_3d_properties([p0_p[2], p1_p[2], p2_p[2], p3_p[2]])
            line_pinn.set_visible(True)
            # Masquer DLS
            line_dls.set_visible(False)
            
            title_text.set_text(f"Mode : PINN (IA) | Erreur Pince : {err_pinn:.1f} mm")
            
        elif mode == 'Solveur DLS uniquement':
            # Masquer PINN
            line_pinn.set_visible(False)
            # Afficher DLS
            line_dls.set_data([p0_d[0], p1_d[0], p2_d[0], p3_d[0]], [p0_d[1], p1_d[1], p2_d[1], p3_d[1]])
            line_dls.set_3d_properties([p0_d[2], p1_d[2], p2_d[2], p3_d[2]])
            line_dls.set_visible(True)
            
            title_text.set_text(f"Mode : Numérique DLS | Erreur Pince : {err_dls:.1f} mm")
            
        else: # Comparer les deux
            # Afficher les deux bras
            line_pinn.set_data([p0_p[0], p1_p[0], p2_p[0], p3_p[0]], [p0_p[1], p1_p[1], p2_p[1], p3_p[1]])
            line_pinn.set_3d_properties([p0_p[2], p1_p[2], p2_p[2], p3_p[2]])
            line_pinn.set_visible(True)
            
            line_dls.set_data([p0_d[0], p1_d[0], p2_d[0], p3_d[0]], [p0_d[1], p1_d[1], p2_d[1], p3_d[1]])
            line_dls.set_3d_properties([p0_d[2], p1_d[2], p2_d[2], p3_d[2]])
            line_dls.set_visible(True)
            
            title_text.set_text(f"Erreur IA (PINN) : {err_pinn:.1f} mm | Erreur DLS : {err_dls:.1f} mm")

        fig.canvas.draw_idle()

    # Relier les évènements de modification des widgets à la fonction update
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)
    radio_mode.on_clicked(update)

    # Lancer le tracé initial
    update(None)
    
    plt.show()

if __name__ == "__main__":
    main()
