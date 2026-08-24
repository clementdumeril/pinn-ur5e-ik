# 📘 Rapport Technique et Guide d'Exportation : Projet UR5e True PINN

Ce document synthétise l'ensemble du travail réalisé sur le solveur de **Cinématique Inverse 3D par Réseau de Neurones Informé par la Physique (Physics-Informed Neural Network - PINN)** appliqué au robot industriel **Universal Robots UR5e (6-DOF)** sous **Webots**.

![Capture Webots Simulation Robot Unique](assets/webots_single_robot.png)

---

## 💡 1. Résumé Exécutif & Vision du Projet

Les méthodes classiques de résolution de la cinématique inverse (IK) pour les bras articulés à 6 degrés de liberté reposent sur des calculs matriciels itératifs (Jacobienne inverse, Levenberg-Marquardt) ou des équations géométriques analytiques. Bien que précises, ces méthodes souffrent de deux limites majeures :
1. **Multiplicité des solutions (Discontinuité)** : Pour une même position $(X,Y,Z)$, il peut exister jusqu'à 8 configurations d'angles différentes pour le UR5e, causant des sauts brusques d'articulations lors des déplacements.
2. **Temps de calcul & Non-différentiabilité** : Les méthodes itératives sont coûteuses en ressources processeur et ne peuvent pas être intégrées dans des boucles d'apprentissage profond de bout en bout.

**Notre Solution :**
Un réseau de neurones artificiels profond entraîné avec une **perte hybride informée par la physique (True PINN)**. Le réseau prend directement la position cible $(X,Y,Z)$ et prédit de manière instantanée et fluide les 6 angles articulaires $(q_1, q_2, q_3, q_4, q_5, q_6)$, tout en garantissant le respect des lois géométriques réelles du robot grâce à la cinématique directe différenciable codée sous PyTorch.

---

## 🏗️ 2. Architecture Globale du Système

Le système complet s'articule autour de trois composants interconnectés :

```
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│  1. VISION ARTIFICIELLE │ ───> │   2. RESOLUTION PINN    │ ───> │  3. ACTIONNEMENT WEBOTS │
│   Modèle YOLO / VGG16   │      │   PyTorch PINN6DOF Net  │      │   PandaHand Gripper &   │
│  Détection du Cube (XYZ)│      │  Prédit (q1..q6) en 0.4ms│      │   Moteurs UR5e (Webots) │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

### Étape 1 : Perception (Vision par Ordinateur)
- Le robot commence par se placer dans une posture haute de lecture `[-0.1, -0.68, 0.45]`.
- La caméra embarquée capture l'image de la zone de travail.
- Le modèle de vision détecte le cube rouge et convertit ses coordonnées pixels en coordonnées 3D réelles cartésiennes $(X,Y,Z)$ par rapport à la base du robot.

### Étape 2 : Inférence Cinématique (True PINN)
- La coordonnée $(X,Y,Z)$ est transmise au modèle `PINN6DOF` entraîné (`pinn_model_true_physics.pth`).
- Le réseau de neurones réalise un unique passage avant (Forward Pass) en **0,25 ms** (médiane, après échauffement) et renvoie les 6 angles $(q_1, ..., q_6)$. Voir le chapitre 6 pour le détail des mesures.

### Étape 3 : Exécution & Saisie (Simulation Webots)
- Les angles sont convertis en trajectoire polynomiale fluide (`ur5.move_to_config`).
- Le robot descend top-down sur l'objet, ferme la pince **PandaHand**, transporte l'objet et le dépose dans le plateau bleu.

---

## 🧮 3. Formulations Mathématiques du True PINN

La fonction de perte (Loss Function) utilisée pour l'entraînement du réseau combine l'apprentissage supervisé et le modèle physique du robot :

$$\mathcal{L}_{total} = w_{data} \cdot \mathcal{L}_{data} + w_{phys} \cdot \mathcal{L}_{phys}$$

### Perte Données ($\mathcal{L}_{data}$)
Évalue l'erreur quadratique moyenne par rapport aux angles de référence du dataset :
$$\mathcal{L}_{data} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{6} (q_{pred, j}^{(i)} - q_{true, j}^{(i)})^2$$

### Perte Physique ($\mathcal{L}_{phys}$)
Fait passer les angles prédits par le modèle géométrique direct différenciable PyTorch ($\text{FK}_{PyTorch}$) et mesure la distance cartesienne réelle entre la pince et la cible :
$$\mathcal{L}_{phys} = \frac{1}{N} \sum_{i=1}^{N} \left\| \text{FK}_{PyTorch}(q_{pred}^{(i)}) - \mathbf{P}_{cible}^{(i)} \right\|^2_2$$

> **Pondération appliquée lors de l'entraînement :** $w_{data} = 0.1$ et $w_{phys} = 1.0$. Cela force le réseau à privilégier la précision physique dans l'espace 3D tout en conservant une configuration articulaires lisse et naturelle.

---

## 📂 4. Arborescence du Projet pour Export (GitHub & Web)

Pour publier le projet proprement sur GitHub ou votre portfolio web, conservez la structure organisée suivante :

```
pinn_ik_project/
├── models/                               # Modèles et poids entraînés
│   └── pinn_model_true_physics.pth
├── robotics_utils/                       # Utilitaires de cinématique et fonctions PyTorch
│   ├── ur5_pytorch_fk.py                 # Cinématique directe (FK) différenciable PyTorch
│   └── ur5e_6dof_ik.py                   # Utilitaires géométriques
├── training/                             # Scripts d'entraînement
│   ├── train_true_pinn.py                # Script d'entraînement principal du True PINN
│   ├── train_pinn_6dof.py                # Générateur de dataset et architecture réseau
│   └── train_supervised_ik.py            # Baseline d'entraînement supervisé classique
├── reference_ur5_repo/                   # Environnement Webots complet
│   ├── ur5.py                            # Interface et contrôleur maître du robot
│   └── simulation/
│       ├── controllers/
│       │   ├── ur5_controller_pandahand/ # Contrôleur principal Pick & Place
│       │   └── comparison_controller/   # Contrôleur d'affichage HUD du comparatif
│       └── worlds/
│           ├── my_first_simulation_pandahand.wbt # Monde Webots à 1 robot
│           └── pinn_vs_math.wbt                  # Monde Webots comparatif (PINN vs Math)
├── README.md                             # Documentation principale GitHub
└── DOCUMENTATION.md                      # Le présent document de synthèse
```

---

## 🛠️ 5. Instructions d'Exécution

### Dépendances Python
```bash
pip install torch numpy scipy ikpy matplotlib
```

### Entraîner le Modèle True PINN
```bash
python training/train_true_pinn.py
```

### Lancer la Simulation Webots
1. **Simulation Monoposte (Pick & Place avec True PINN)** :
   Ouvrir dans Webots : `reference_ur5_repo/simulation/worlds/my_first_simulation_pandahand.wbt`
2. **Simulation Comparative (True PINN vs IKPY Math)** :
   Ouvrir dans Webots : `reference_ur5_repo/simulation/worlds/pinn_vs_math.wbt`

---

## 📈 6. Résultats et Comparatif de Performance

Mesures sur 1000 appels chronométrés, après 200 appels d'échauffement jetés,
un seul thread, sur 300 cibles atteignables tirées dans la zone d'entraînement.

| Critère | IKPY (itératif) | Analytique en forme fermée | **True PINN** |
| :--- | ---: | ---: | ---: |
| **Temps médian** | 45,0 ms | **0,223 ms** | 0,248 ms |
| Temps moyen | 56,4 ms | 0,280 ms | 0,322 ms |
| Erreur au point de saisie | — | exacte | **0,30 mm** |
| Différentiable | Non | Non | **Oui** |
| Continu en fonction de la cible | Non (8 branches) | Non (8 branches) | **Oui** |

> **À lire honnêtement.** Le PINN n'est **pas** plus rapide que la solution
> analytique en forme fermée : il est 11 % plus lent, et aucun réseau de
> neurones ne battra quelques dizaines d'opérations trigonométriques. Le
> premier appel coûte en outre ~4,5 ms, le temps que PyTorch démarre.

Ce que le PINN apporte réellement, et ce que ce projet démontre :

- **182× plus rapide qu'IKPY**, le solveur numérique itératif qu'il remplace
  effectivement — 0,25 ms contre 45 ms ;
- **différentiable**, donc intégrable dans une chaîne d'apprentissage de bout
  en bout, ce qu'aucune des deux autres méthodes ne permet ;
- **continu** en fonction de la position cible, là où les méthodes analytiques
  sautent entre leurs 8 branches de solution.

Une version antérieure de ce tableau annonçait 0,35-0,45 ms pour le PINN contre
0,50-0,85 ms pour les maths. Ces chiffres avaient été relevés sans échauffement
et sur quelques appels : ils étaient faux dans les deux sens.
