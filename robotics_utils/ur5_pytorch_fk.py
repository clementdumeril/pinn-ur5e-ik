import torch
import math

class UR5ForwardKinematicsPyTorch:
    """
    Batched Forward Kinematics for UR5 in PyTorch.
    Matches exactly the reference UR5 DH parameters.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.pi = math.pi
        
        # UR5 DH parameters (a, alpha, d, theta_offset)
        self.d1 = 0.1625
        self.a2 = 0.425
        self.a3 = 0.3922
        self.d4 = 0.1333
        self.d5 = 0.0997
        self.d6 = 0.0996 + 0.1237
        
        # Format: [a, alpha, d, theta_offset]
        self.dh_params = [
            (0, self.pi / 2, self.d1, 0),
            (self.a2, 0, 0, self.pi / 2),
            (self.a3, 0, 0, 0),
            (0, -self.pi / 2, self.d4, -self.pi / 2),
            (0, self.pi / 2, self.d5, 0),
            (0, 0, self.d6, 0),
        ]
        
    def get_transform_matrix(self, theta, a, alpha, d, theta_offset):
        """
        Creates a batched 4x4 transformation matrix.
        theta: Tensor of shape (batch_size,)
        """
        batch_size = theta.shape[0]
        q = theta + theta_offset
        
        cos_q = torch.cos(q)
        sin_q = torch.sin(q)
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)
        
        T = torch.zeros((batch_size, 4, 4), dtype=theta.dtype, device=theta.device)
        
        # Row 1
        T[:, 0, 0] = cos_q
        T[:, 0, 1] = -sin_q * cos_alpha
        T[:, 0, 2] = sin_q * sin_alpha
        T[:, 0, 3] = a * cos_q
        
        # Row 2
        T[:, 1, 0] = sin_q
        T[:, 1, 1] = cos_q * cos_alpha
        T[:, 1, 2] = -cos_q * sin_alpha
        T[:, 1, 3] = a * sin_q
        
        # Row 3
        T[:, 2, 0] = 0
        T[:, 2, 1] = sin_alpha
        T[:, 2, 2] = cos_alpha
        T[:, 2, 3] = d
        
        # Row 4
        T[:, 3, 3] = 1.0
        
        return T

    def forward(self, joint_angles):
        """
        Computes forward kinematics for a batch of joint angles.
        joint_angles: Tensor of shape (batch_size, 6)
        Returns:
            T: Tensor of shape (batch_size, 4, 4) - End effector poses
        """
        batch_size = joint_angles.shape[0]
        # Initialize with Identity matrices
        T_total = torch.eye(4, dtype=joint_angles.dtype, device=joint_angles.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        for i in range(6):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta_i = joint_angles[:, i]
            T_i = self.get_transform_matrix(theta_i, a, alpha, d, theta_offset)
            T_total = torch.bmm(T_total, T_i)
            
        return T_total
        
    def forward_pos(self, joint_angles):
        """
        Returns only the XYZ position (batch_size, 3)
        """
        T = self.forward(joint_angles)
        return T[:, :3, 3]

if __name__ == "__main__":
    # Test
    fk = UR5ForwardKinematicsPyTorch()
    q = torch.tensor([[0.0, -math.pi/2, 0.0, -math.pi/2, 0.0, 0.0]])
    pos = fk.forward_pos(q)
    print("Test pose XYZ:", pos)
