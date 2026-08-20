"""
Générateur de Trajectoires Polynomiales de Degré 5 (Quintic Polynomials).
Inspiré de la méthode d'Allan Almeida (ur5-pick-and-place-webots).
Garantit des mouvements d'articulation 100% lissés, sans à-coups, avec vitesse et accélération nulles au départ et à l'arrivée.
"""
import numpy as np

class QuinticTrajectory:
    def __init__(self, q_start, q_target, duration):
        """
        Génère les coefficients du polynôme de degré 5 pour chaque articulation.
        q_start  : array (6,) angles initiaux (rad)
        q_target : array (6,) angles cibles (rad)
        duration : durée totale du mouvement (secondes)
        """
        self.q_start = np.array(q_start, dtype=float)
        self.q_target = np.array(q_target, dtype=float)
        self.T = max(duration, 0.1)

        # Matrice des coefficients a3, a4, a5
        # q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # a0 = q_start, a1 = 0, a2 = 0
        dq = self.q_target - self.q_start
        self.a0 = self.q_start
        self.a3 = 10.0 * dq / (self.T**3)
        self.a4 = -15.0 * dq / (self.T**4)
        self.a5 = 6.0 * dq / (self.T**5)

    def get_position(self, t):
        """Renvoie la position articulaire à l'instant t (0 <= t <= T)."""
        t = np.clip(t, 0.0, self.T)
        return self.a0 + self.a3 * (t**3) + self.a4 * (t**4) + self.a5 * (t**5)

    def get_velocity(self, t):
        """Renvoie la vitesse articulaire à l'instant t."""
        t = np.clip(t, 0.0, self.T)
        return 3.0 * self.a3 * (t**2) + 4.0 * self.a4 * (t**3) + 5.0 * self.a5 * (t**4)


if __name__ == "__main__":
    q_in = np.array([0.0, -1.0, 1.0, -1.5, -1.57, 0.0])
    q_out = np.array([0.2, -0.5, 1.5, -2.5, -1.57, 0.0])

    traj = QuinticTrajectory(q_in, q_out, duration=2.0)

    print("=== TEST DE TRAJECTOIRE QUINTIQUE ===")
    print("t=0.0s :", np.round(traj.get_position(0.0), 3))
    print("t=1.0s :", np.round(traj.get_position(1.0), 3))
    print("t=2.0s :", np.round(traj.get_position(2.0), 3))
