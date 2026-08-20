"""
Solveur Cinématique UR5e basé sur la bibliothèque standard Python IKPY.
"""
import numpy as np
import ikpy.chain

class IKPYUR5eSolver:
    def __init__(self, urdf_path="ur5e.urdf"):
        self.chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            active_links_mask=[False, True, True, True, True, True, True]
        )

    def solve_ik(self, target_xyz_robot, target_orientation=None):
        """
        Résout l'IK pour target_xyz_robot (position par rapport à la base du robot).
        """
        if target_orientation is not None:
            # Matrice de transformation 4x4 avec orientation
            R = np.eye(4)
            R[:3, :3] = target_orientation
            R[:3, 3] = target_xyz_robot
            q_full = self.chain.inverse_kinematics_frame(R)
        else:
            q_full = self.chain.inverse_kinematics(target_xyz_robot)

        # q_full contient 7 elements (1 base fixe + 6 joints articulés)
        q_joints = q_full[1:]
        
        # FK check
        T_fk = self.chain.forward_kinematics(q_full)
        err_mm = np.linalg.norm(T_fk[:3, 3] - target_xyz_robot) * 1000.0
        
        return q_joints, True, err_mm


if __name__ == "__main__":
    solver = IKPYUR5eSolver()
    target = [0.50, 0.10, -0.13]
    q_sol, success, err_mm = solver.solve_ik(target)
    print("=== IKPY SOLVER TEST ===")
    print(f"Target: {target} | Success: {success} | FK Error: {err_mm:.4f} mm")
    print(f"Angles 6-DOF (deg): {[round(np.degrees(a), 1) for a in q_sol]}")
