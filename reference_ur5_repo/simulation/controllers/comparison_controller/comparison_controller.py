import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from controller import Supervisor

def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    
    print("\n=======================================================")
    print("🥊 COMBAT : TRUE PINN vs MATHÉMATIQUES CLASSIQUES")
    print("=======================================================\n")
    
    # Textes HUD Statiques (Titres)
    robot.setLabel(0, "UR5e A (Rouge) : IA TRUE PINN", 0.02, 0.02, 0.1, 0xff0000, 0, "Arial")
    robot.setLabel(1, "UR5e B (Vert) : IK ANALYTIQUE", 0.02, 0.06, 0.1, 0x00ff00, 0, "Arial")
    
    # Boucle infinie pour maintenir le controleur en vie
    while robot.step(timestep) != -1:
        pass

if __name__ == "__main__":
    main()
