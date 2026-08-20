import torch
import torch.nn as nn

class PINNInverseKinematics(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=128):
        super(PINNInverseKinematics, self).__init__()
        
        # Architecture simple à 3 couches cachées avec activation SiLU (Swish)
        # SiLU est lisse et infiniment différentiable, ce qui aide au calcul des gradients physiques
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Paramètres d'échelle pour l'entraînement
        self.register_buffer("mean_in", torch.tensor([0.0, 0.0, 1.5]))
        self.register_buffer("std_in", torch.tensor([1.5, 1.5, 1.5]))
        self.register_buffer("mean_out", torch.tensor([0.0, 1.0, 0.0]))
        self.register_buffer("std_out", torch.tensor([1.5, 1.0, 1.0]))

    def forward(self, x):
        """
        x: Tenseur de taille (N, 3) représentant (x, y, z) cibles
        """
        # Normalisation interne
        x_scaled = (x - self.mean_in) / self.std_in
        
        # Prédiction des angles normalisés
        out_scaled = self.net(x_scaled)
        
        # Dénormalisation des angles de sortie
        angles = out_scaled * self.std_out + self.mean_out
        return angles

    def save_model(self, filepath):
        torch.save({
            'state_dict': self.state_dict(),
            'mean_in': self.mean_in,
            'std_in': self.std_in,
            'mean_out': self.mean_out,
            'std_out': self.std_out,
            'hidden_dim': self.net[0].out_features  # sauvegarder la taille du réseau
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
        self.load_state_dict(checkpoint['state_dict'])
        self.mean_in.copy_(checkpoint['mean_in'])
        self.std_in.copy_(checkpoint['std_in'])
        self.mean_out.copy_(checkpoint['mean_out'])
        self.std_out.copy_(checkpoint['std_out'])

    @classmethod
    def from_file(cls, filepath):
        """Charge un modèle PINN depuis un fichier en détectant automatiquement hidden_dim."""
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        model = cls(hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint['state_dict'])
        model.mean_in.copy_(checkpoint['mean_in'])
        model.std_in.copy_(checkpoint['std_in'])
        model.mean_out.copy_(checkpoint['mean_out'])
        model.std_out.copy_(checkpoint['std_out'])
        model.eval()
        return model

