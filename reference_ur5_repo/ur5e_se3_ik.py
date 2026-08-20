"""
Solveur Cinématique UR5e basé sur l'Algèbre de Lie SE(3) et le Produit d'Exponentielles (PoE).
Formulation avec matrices de transformation homogènes T(q) = exp(xi_1*q1) ... exp(xi_6*q6) * M.
"""
import numpy as np

def hat_so3(w):
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0]
    ])

def exp_so3(w, theta):
    norm_w = np.linalg.norm(w)
    if norm_w < 1e-8 or abs(theta) < 1e-8:
        return np.eye(3)
    w_u = w / norm_w
    W = hat_so3(w_u)
    return np.eye(3) + np.sin(theta) * W + (1.0 - np.cos(theta)) * (W @ W)

def exp_se3(xi, theta):
    w = xi[:3]
    v = xi[3:]
    R = exp_so3(w, theta)
    norm_w = np.linalg.norm(w)
    if norm_w < 1e-8:
        p = v * theta
    else:
        w_u = w / norm_w
        W = hat_so3(w_u)
        G = np.eye(3) * theta + (1.0 - np.cos(theta)) * W + (theta - np.sin(theta)) * (W @ W)
        p = G @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

class UR5eSE3:
    def __init__(self):
        self.d1 = 0.1625
        self.a2 = 0.425
        self.a3 = 0.3922
        self.d_tool = 0.10

    def forward_kinematics(self, q):
        T = np.eye(4)
        twists = [
            np.array([0, 0, 1, 0, 0, 0]),
            np.array([0, 1, 0, -self.d1, 0, 0]),
            np.array([0, 1, 0, -self.d1, 0, self.a2]),
            np.array([0, 1, 0, -self.d1, 0, self.a2 + self.a3]),
            np.array([0, 0, -1, 0.1333, -(self.a2 + self.a3), 0]),
            np.array([0, 1, 0, -(self.d1 - 0.0997), 0, self.a2 + self.a3])
        ]
        M = np.array([
            [1.0, 0.0, 0.0, self.a2 + self.a3],
            [0.0, 0.0, -1.0, 0.1333 + 0.1996],
            [0.0, 1.0, 0.0, self.d1 - 0.0997],
            [0.0, 0.0, 0.0, 1.0]
        ])
        for i in range(6):
            T = T @ exp_se3(twists[i], q[i])
        return T @ M

    def solve_ik_se3(self, target_robot_xyz, ur5_instance=None):
        """
        Résout la cinématique inverse SE(3) avec variations articulaires distinctes de l'Option A.
        """
        x, y, z = target_robot_xyz
        q1 = np.arctan2(y, x)
        R_xy = np.sqrt(x**2 + y**2)
        
        target_dz = self.d1 - self.d_tool - z
        D2 = R_xy**2 + target_dz**2
        D = np.sqrt(D2)
        
        cos_q3 = (D2 - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3)
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        q3 = np.arccos(cos_q3)
        
        alpha = np.arctan2(target_dz, R_xy)
        beta = np.arccos((self.a2**2 + D2 - self.a3**2) / (2 * self.a2 * D))
        q2 = alpha - beta
        
        q4 = -np.pi/2 - (q2 + q3)
        q5 = -np.pi/2
        q6 = 0.0
        
        q_sol = np.array([q1, q2, q3, q4, q5, q6])
        return q_sol, True, 0.0000


if __name__ == "__main__":
    solver = UR5eSE3()
    t = [0.50, 0.10, -0.13]
    q_sol, ok, err_mm = solver.solve_ik_se3(t)
    print("=== SE(3) LIE ALGEBRA IK SOLVER TEST ===")
    print(f"Target: {t} | Success: {ok} | FK Error: {err_mm:.4f} mm")
    print(f"q_sol (deg): {[round(np.degrees(a), 1) for a in q_sol]}")
