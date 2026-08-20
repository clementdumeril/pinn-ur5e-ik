import numpy as np
import torch

class AnthropomorphicArm3D:
    def __init__(self, L1=1.0, L2=1.0, L3=1.0):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        
    def forward_kinematics_numpy(self, theta):
        """
        Calcule la position en 3D de chaque articulation et de la pince.
        theta = [theta1, theta2, theta3]
        Renvoie:
            p0: Base [0, 0, 0]
            p1: Épaule [0, 0, L1]
            p2: Coude [x2, y2, z2]
            p3: Pince (End-Effector) [x3, y3, z3]
        """
        t1, t2, t3 = theta
        
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 0.0, self.L1])
        
        # Position du coude (Joint 3)
        x2 = self.L2 * np.cos(t1) * np.cos(t2)
        y2 = self.L2 * np.sin(t1) * np.cos(t2)
        z2 = self.L1 + self.L2 * np.sin(t2)
        p2 = np.array([x2, y2, z2])
        
        # Position de la pince
        x3 = np.cos(t1) * (self.L2 * np.cos(t2) + self.L3 * np.cos(t2 + t3))
        y3 = np.sin(t1) * (self.L2 * np.cos(t2) + self.L3 * np.cos(t2 + t3))
        z3 = self.L1 + self.L2 * np.sin(t2) + self.L3 * np.sin(t2 + t3)
        p3 = np.array([x3, y3, z3])
        
        return p0, p1, p2, p3

    def forward_kinematics_pytorch(self, thetas):
        """
        Version PyTorch vectorisée et différentiable de la FK (calcul de la pince).
        thetas: Tenseur PyTorch de taille (N, 3)
        Renvoie: Tenseur de taille (N, 3) contenant les positions (x, y, z) de la pince.
        """
        t1 = thetas[:, 0]
        t2 = thetas[:, 1]
        t3 = thetas[:, 2]
        
        x = torch.cos(t1) * (self.L2 * torch.cos(t2) + self.L3 * torch.cos(t2 + t3))
        y = torch.sin(t1) * (self.L2 * torch.cos(t2) + self.L3 * torch.cos(t2 + t3))
        z = self.L1 + self.L2 * torch.sin(t2) + self.L3 * torch.sin(t2 + t3)
        
        return torch.stack([x, y, z], dim=1)

    def analytical_jacobian(self, theta):
        """
        Calcule la Jacobienne analytique de position (3x3) pour la pince.
        """
        t1, t2, t3 = theta
        L2, L3 = self.L2, self.L3
        
        J = np.zeros((3, 3))
        
        # Effet de theta1
        J[0, 0] = -np.sin(t1) * (L2 * np.cos(t2) + L3 * np.cos(t2 + t3))
        J[1, 0] =  np.cos(t1) * (L2 * np.cos(t2) + L3 * np.cos(t2 + t3))
        J[2, 0] = 0.0
        
        # Effet de theta2
        J[0, 1] = np.cos(t1) * (-L2 * np.sin(t2) - L3 * np.sin(t2 + t3))
        J[1, 1] = np.sin(t1) * (-L2 * np.sin(t2) - L3 * np.sin(t2 + t3))
        J[2, 1] = L2 * np.cos(t2) + L3 * np.cos(t2 + t3)
        
        # Effet de theta3
        J[0, 2] = np.cos(t1) * (-L3 * np.sin(t2 + t3))
        J[1, 2] = np.sin(t1) * (-L3 * np.sin(t2 + t3))
        J[2, 2] = L3 * np.cos(t2 + t3)
        
        return J

    def inverse_kinematics_numerical(self, target_pos, theta0, max_iter=100, tol=1e-4, damping=0.1):
        """
        Solveur numérique classique par Damped Least Squares (DLS) pour comparaison.
        """
        theta = np.array(theta0, dtype=float)
        for _ in range(max_iter):
            _, _, _, p_current = self.forward_kinematics_numpy(theta)
            error = target_pos - p_current
            
            if np.linalg.norm(error) < tol:
                return theta, True
                
            J = self.analytical_jacobian(theta)
            # Damped Least Squares update
            d_theta = np.dot(J.T, np.linalg.solve(np.dot(J, J.T) + damping**2 * np.eye(3), error))
            theta += d_theta
            
            # Butées articulaires physiques pour un rendu réaliste (cohérent avec l'entraînement)
            theta[0] = np.clip(theta[0], -2.0, 2.0) # Pivot base
            theta[1] = np.clip(theta[1], 0.1, 2.9) # Élévation épaule (évite de taper le sol)
            theta[2] = np.clip(theta[2], -2.0, 2.0) # Pliage coude
            
        return theta, False


# =============================================================================
#  UR5e — Cinématique réelle 3 axes (shoulder_pan, shoulder_lift, elbow)
#  Paramètres DH nominaux Universal Robots, poignets verrouillés à 0.
#
#  Convention UR5e (Webots) :
#    q2 = 0      → bras horizontal vers +X
#    q2 = -π/2   → bras vertical vers le haut (+Z)
#    q2 > 0      → bras incliné vers le bas
#  Le signe de sin(q2) est INVERSÉ dans le calcul de z pour respecter
#  cette convention : z = d1 - a2·sin(q2) - a3·sin(q2+q3)
# =============================================================================
class UR5e3Axis:
    """
    Cinématique avant du UR5e réduite à 3 axes actifs.

    Paramètres DH nominaux UR5e :
        d1 = 0.1625 m   (hauteur base → épaule)
        a2 = 0.425  m   (longueur bras supérieur)
        a3 = 0.3922 m   (longueur avant-bras)

    Joints actifs :
        q1 = shoulder_pan_joint   (rotation autour de Z)
        q2 = shoulder_lift_joint  (rotation autour de Y)
        q3 = elbow_joint          (rotation autour de Y)

    Vérifications de cohérence :
        q = [0, 0, 0]       → effecteur à (0.8172, 0, 0.1625)   bras horizontal
        q = [0, -π/2, 0]    → effecteur à (0, 0, 0.9797)        bras vertical haut
        q = [0, π/2, 0]     → effecteur à (0, 0, -0.6547)       bras vertical bas
    """

    def __init__(self):
        self.d1 = 0.1625   # base → épaule (hauteur)
        self.a2 = 0.425    # longueur bras supérieur
        self.a3 = 0.3922   # longueur avant-bras

    def forward_kinematics_numpy(self, theta):
        """
        FK 3 axes du UR5e.

        theta = [q1, q2, q3] (radians)

        Renvoie : (p_base, p_shoulder, p_elbow, p_effector)
        """
        q1, q2, q3 = theta

        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q2 + q3), np.sin(q2 + q3)

        p_base = np.array([0.0, 0.0, 0.0])
        p_shoulder = np.array([0.0, 0.0, self.d1])

        # Coude : bout du bras supérieur
        x_elbow = self.a2 * c2 * c1
        y_elbow = self.a2 * c2 * s1
        z_elbow = self.d1 - self.a2 * s2          # ← signe inversé
        p_elbow = np.array([x_elbow, y_elbow, z_elbow])

        # Effecteur : bout de l'avant-bras
        reach_x = self.a2 * c2 + self.a3 * c23
        x_eff = reach_x * c1
        y_eff = reach_x * s1
        z_eff = self.d1 - self.a2 * s2 - self.a3 * s23   # ← signe inversé
        p_effector = np.array([x_eff, y_eff, z_eff])

        return p_base, p_shoulder, p_elbow, p_effector

    def forward_kinematics_pytorch(self, thetas):
        """
        Version PyTorch vectorisée et différentiable (position effecteur).
        thetas : Tenseur (N, 3)
        Renvoie : Tenseur (N, 3) [x, y, z]
        """
        q1 = thetas[:, 0]
        q2 = thetas[:, 1]
        q3 = thetas[:, 2]

        c1, s1 = torch.cos(q1), torch.sin(q1)
        c2, s2 = torch.cos(q2), torch.sin(q2)
        c23, s23 = torch.cos(q2 + q3), torch.sin(q2 + q3)

        reach_x = self.a2 * c2 + self.a3 * c23

        x = reach_x * c1
        y = reach_x * s1
        z = self.d1 - self.a2 * s2 - self.a3 * s23       # ← signe inversé

        return torch.stack([x, y, z], dim=1)

    def analytical_jacobian(self, theta):
        """
        Jacobienne analytique 3×3 de position.
        """
        q1, q2, q3 = theta
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q2 + q3), np.sin(q2 + q3)

        reach_x = self.a2 * c2 + self.a3 * c23

        J = np.zeros((3, 3))

        # ∂/∂q1
        J[0, 0] = -reach_x * s1
        J[1, 0] = reach_x * c1
        J[2, 0] = 0.0

        # ∂/∂q2  (dérivées de z avec le signe inversé)
        d_reach_x_q2 = -self.a2 * s2 - self.a3 * s23
        J[0, 1] = d_reach_x_q2 * c1
        J[1, 1] = d_reach_x_q2 * s1
        J[2, 1] = -self.a2 * c2 - self.a3 * c23    # ← d/dq2 de (−a2·s2 −a3·s23)

        # ∂/∂q3
        d_reach_x_q3 = -self.a3 * s23
        J[0, 2] = d_reach_x_q3 * c1
        J[1, 2] = d_reach_x_q3 * s1
        J[2, 2] = -self.a3 * c23                    # ← d/dq3 de (−a3·s23)

        return J

    def inverse_kinematics_numerical(self, target_pos, theta0, max_iter=200, tol=1e-4, damping=0.05):
        """
        Solveur IK numérique (Damped Least Squares) pour le UR5e 3 axes.
        """
        theta = np.array(theta0, dtype=float)
        for _ in range(max_iter):
            _, _, _, p_current = self.forward_kinematics_numpy(theta)
            error = target_pos - p_current

            if np.linalg.norm(error) < tol:
                return theta, True

            J = self.analytical_jacobian(theta)
            d_theta = np.dot(J.T, np.linalg.solve(
                np.dot(J, J.T) + damping**2 * np.eye(3), error))
            theta += d_theta

            # Butées articulaires (zone de travail frontale)
            theta[0] = np.clip(theta[0], -np.pi, np.pi)
            theta[1] = np.clip(theta[1], -np.pi, np.pi)
            theta[2] = np.clip(theta[2], -np.pi, np.pi)

        return theta, False

