import os
import json
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import torch
from robot_model import UR5e3Axis, AnthropomorphicArm3D
from pinn_model import PINNInverseKinematics

# 1. Initialisation globale du modele de robot et de l'IA (PINN)
# On essaie d'abord le modele UR5e, sinon fallback sur l'ancien
model_path_ur5e = "pinn_model_ur5e.pth"
model_path_old = "pinn_model.pth"

if os.path.exists(model_path_ur5e):
    arm = UR5e3Axis()
    pinn = PINNInverseKinematics.from_file(model_path_ur5e)
    print("[OK] Modele PINN UR5e charge.")
elif os.path.exists(model_path_old):
    arm = AnthropomorphicArm3D()
    pinn = PINNInverseKinematics()
    pinn.load_model(model_path_old)
    pinn.eval()
    print("[OK] Modele PINN ancien (bras orange) charge.")
else:
    arm = UR5e3Axis()
    pinn = None
    print("[!] Aucun modele PINN trouve. Lancez 'python train_pinn.py'." )


# 2. Classe du Serveur HTTP (Utilise uniquement la bibliothèque standard de Python)
class SimulatorHandler(BaseHTTPRequestHandler):
    
    # Masquer les logs de requêtes individuelles dans le terminal pour garder la console propre
    def log_message(self, format, *args):
        return

    # Gestion des requêtes GET (Chargement de la page web)
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            # Servir le fichier index.html local
            html_path = os.path.join(os.path.dirname(__file__), 'index.html')
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
            else:
                self.wfile.write(b"Fichier index.html introuvable.")
        else:
            self.send_error(404, "Page non trouvée")

    # Gestion des requêtes POST (Calcul de la cinématique inverse en direct)
    def do_POST(self):
        if self.path == '/api/ik':
            # Lire les données brutes envoyées par la page web (Three.js)
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Extraire les coordonnées cibles X, Y, Z
            tgt_x = float(data.get("x", 0.4))
            tgt_y = float(data.get("y", 0.0))
            tgt_z = float(data.get("z", 0.3))
            target_pos = np.array([tgt_x, tgt_y, tgt_z])
            
            # --- A. Résolution par l'IA (PINN) ---
            pinn_angles = [0.0, 0.0, 0.0]
            pinn_error = 999.0
            pinn_trained = False
            
            if pinn is not None:
                pinn_trained = True
                target_tensor = torch.tensor([[tgt_x, tgt_y, tgt_z]], dtype=torch.float32)
                with torch.no_grad():
                    pred_angles = pinn(target_tensor).numpy()[0]
                pinn_angles = [float(a) for a in pred_angles]
                
                # Calcul de la FK pour mesurer l'erreur réelle
                _, _, _, p3_pinn = arm.forward_kinematics_numpy(pred_angles)
                pinn_error = float(np.linalg.norm(p3_pinn - target_pos) * 1000.0)
                
            # --- B. Résolution par le Solveur Numérique (DLS) ---
            theta0 = np.array([0.0, -1.0, 0.5]) # Pose de depart
            dls_angles, _ = arm.inverse_kinematics_numerical(target_pos, theta0)
            dls_angles_list = [float(a) for a in dls_angles]
            
            # Calcul de la FK pour mesurer l'erreur réelle
            _, _, _, p3_dls = arm.forward_kinematics_numpy(dls_angles)
            dls_error = float(np.linalg.norm(p3_dls - target_pos) * 1000.0)

            # --- C. Positions 3D des articulations (pour le tracé des cylindres) ---
            p0_p, p1_p, p2_p, p3_p = arm.forward_kinematics_numpy(np.array(pinn_angles))
            p0_d, p1_d, p2_d, p3_d = arm.forward_kinematics_numpy(np.array(dls_angles_list))
            
            response_data = {
                "pinn_trained": pinn_trained,
                "pinn": {
                    "angles": pinn_angles,
                    "joints": [p0_p.tolist(), p1_p.tolist(), p2_p.tolist(), p3_p.tolist()],
                    "error_mm": pinn_error
                },
                "dls": {
                    "angles": dls_angles_list,
                    "joints": [p0_d.tolist(), p1_d.tolist(), p2_d.tolist(), p3_d.tolist()],
                    "error_mm": dls_error
                }
            }
            
            # Renvoyer la réponse au format JSON
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))


# 3. Lancement du simulateur
def run_simulator(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimulatorHandler)
    
    print(f"\n=======================================================")
    print(f"🚀 Simulateur 3D Démarré sur : http://localhost:{port}")
    print(f"=======================================================\n")
    
    # Ouvrir automatiquement le navigateur internet par défaut sur le simulateur
    webbrowser.open(f"http://localhost:{port}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping simulator server...")
        httpd.server_close()

if __name__ == '__main__':
    run_simulator()
