# 🧭 Guide des Codes — Projet PINN IK / UR5e

Ce document explique **chaque fichier de code du projet**, ce qu'il fait, comment il
s'articule avec les autres, et comment publier le tout proprement sur GitHub.

> **Fichier de simulation principal (à ouvrir dans Webots) :**
> `reference_ur5_repo/simulation/worlds/my_first_simulation_pandahand.wbt`

---

## 1. Vue d'ensemble en une image

```
                        ┌──────────────────────────────────────┐
                        │  WEBOTS (moteur physique 3D)         │
                        │  .wbt  →  charge un « controller »   │
                        └───────────────┬──────────────────────┘
                                        │
             ┌──────────────────────────▼──────────────────────────┐
             │  controllers/ur5_controller_pandahand.py            │
             │  = le SCÉNARIO (pick & place, étape par étape)      │
             └──────────────────────────┬──────────────────────────┘
                                        │  appelle
             ┌──────────────────────────▼──────────────────────────┐
             │  reference_ur5_repo/ur5.py   = la CLASSE PILOTE     │
             │  • lit la caméra / la position des objets           │
             │  • résout l'IK  →  2 chemins possibles :            │
             │        (a) use_pinn=True  → réseau de neurones      │
             │        (b) use_pinn=False → maths analytiques       │
             │  • génère la trajectoire quintique et pousse les    │
             │    consignes dans les moteurs Webots                │
             └───────┬──────────────────────────────┬──────────────┘
                     │                              │
      ┌──────────────▼────────────┐   ┌─────────────▼──────────────┐
      │ models/                   │   │ robotics_utils/            │
      │ pinn_model_true_physics   │   │ solveurs mathématiques     │
      │ .pth  (poids entraînés)   │   │ (DH, SE(3), IKPY, quintic) │
      └──────────────▲────────────┘   └────────────────────────────┘
                     │ produit par
      ┌──────────────┴────────────┐
      │ training/train_true_pinn  │
      │ .py  (entraînement PINN)  │
      └───────────────────────────┘
```

**Le fil rouge du projet :** au lieu de résoudre la cinématique inverse (trouver les
6 angles articulaires qui amènent la pince à une position `(x, y, z)`) par des
équations matricielles itératives, on entraîne un réseau de neurones à le faire —
mais en lui imposant pendant l'entraînement de **respecter la géométrie réelle du
robot**, via une cinématique directe écrite en PyTorch et donc dérivable.

---

## 2. Arborescence commentée

```
pinn_ik_project/
├── README.md                     ← vitrine GitHub (anglais)
├── DOCUMENTATION.md              ← rapport technique (français)
├── GUIDE_DES_CODES.md            ← CE DOCUMENT
├── requirements.txt              ← dépendances Python
│
├── robotics_utils/               ← 🧮 BOÎTE À OUTILS MATHÉMATIQUE
│   ├── ur5_pytorch_fk.py         ← FK dérivable PyTorch  ⭐ cœur du PINN
│   ├── ur5e_6dof_ik.py           ← FK/IK analytique simplifiée
│   ├── ur5e_se3_ik.py            ← FK par algèbre de Lie SE(3)
│   ├── ikpy_ur5e_solver.py       ← IK numérique via la lib IKPY
│   ├── ur5e_trajectory.py        ← trajectoires polynomiales quintiques
│   └── ur5e.urdf                 ← description du robot (pour IKPY)
│
├── training/                     ← 🧠 ENTRAÎNEMENT DES RÉSEAUX
│   ├── train_true_pinn.py        ← ⭐ script principal (PINN hybride)
│   ├── train_pinn_6dof.py        ← définit l'architecture PINN6DOF + variante
│   ├── train_supervised_ik.py    ← baseline sans physique (pour comparer)
│   └── archive_training/         ← anciennes versions 3-DOF
│
├── models/                       ← 💾 POIDS ENTRAÎNÉS (.pth)
│   ├── pinn_model_true_physics.pth  ← ⭐ celui que charge ur5.py
│   ├── pinn_model_webots.pth        ← baseline supervisée
│   └── archive_models/
│
├── reference_ur5_repo/           ← 🤖 ENVIRONNEMENT WEBOTS COMPLET
│   ├── ur5.py                    ← ⭐ classe pilote du robot (966 lignes)
│   ├── ur5e_se3_ik.py            ← copie locale du solveur SE(3)
│   ├── computer_vision/          ← modèle VGG16 de détection du cube
│   ├── dataset/                  ← images + labels générés en simulation
│   └── simulation/
│       ├── controllers/          ← les « scénarios » exécutés par Webots
│       │   ├── ur5_controller_pandahand/  ← ⭐ pick & place principal
│       │   ├── ur5_controller/            ← variante pince Robotiq 3F
│       │   ├── comparison_controller/     ← superviseur HUD du duel
│       │   ├── multi_controller/          ← duel à 3 robots
│       │   └── data_collector/            ← génération du dataset vision
│       └── worlds/               ← ⭐ LES FICHIERS DE SIMULATION .wbt
│
├── archive_v1_custom_arm/        ← 📦 prototype v1 (bras 3-DOF maison)
└── assets/                       ← captures d'écran pour la doc
```

---

## 3. 🧮 `robotics_utils/` — la boîte à outils mathématique

### 3.1 `ur5_pytorch_fk.py` ⭐ *(le fichier le plus important du projet)*

**Rôle :** recalculer, **en PyTorch et de façon dérivable**, où se trouve la pince
quand on donne les 6 angles articulaires.

C'est ce qui rend le PINN possible. Une fonction `numpy` classique n'est pas
dérivable par `autograd` : impossible de rétropropager l'erreur à travers elle.
En réécrivant la cinématique directe avec des opérations `torch`, le gradient
traverse la géométrie du robot et le réseau peut apprendre « en regardant où sa
main atterrit vraiment ».

| Élément | Explication |
| :-- | :-- |
| `d1=0.1625, a2=0.425, a3=0.3922, d4=0.1333, d5=0.0997, d6=0.2233` | Paramètres Denavit-Hartenberg réels du UR5e (en mètres). Identiques à ceux de `ur5.py`, sinon le PINN apprendrait un robot différent de celui simulé. |
| `dh_params` | Les 6 lignes `(a, alpha, d, theta_offset)` du tableau DH. |
| `get_transform_matrix(...)` | Construit la matrice homogène 4×4 d'un seul joint, **en batch** : shape `(B, 4, 4)`. Les cases sont remplies une par une plutôt que via `torch.stack`, pour rester lisible. |
| `forward(q)` | Multiplie les 6 matrices avec `torch.bmm` (batch matrix multiply) → pose complète `(B, 4, 4)` de l'effecteur. |
| `forward_pos(q)` | Ne garde que la colonne de translation `T[:, :3, 3]` → position XYZ `(B, 3)`. **C'est cette fonction qu'appelle la loss physique.** |

```python
fk = UR5ForwardKinematicsPyTorch()
pos = fk.forward_pos(q)          # q: (B,6) → pos: (B,3), dérivable
```

### 3.2 `ur5e_6dof_ik.py` — solveur analytique simplifié

Classe `UR5e6DOF`, modèle **planaire simplifié** (le bras est traité comme un bras
plan tourné de `q1` autour de l'axe vertical, avec un outil de 10 cm) :

- `forward_kinematics(q)` — version numpy, ne dépend que de `q1..q4`.
- `forward_kinematics_pytorch(q)` — même formule vectorisée pour un tenseur `(N,6)`.
- `solve_ik_downwards(target)` — **IK en forme fermée**, avec la contrainte que
  l'outil pointe strictement vers le bas :
  1. `q1 = atan2(y, x)` — l'orientation de la base est immédiate,
  2. loi des cosinus sur le triangle épaule-coude-poignet → `q3` (coude),
  3. `q2 = alpha − beta` (angle vers la cible moins l'angle interne du triangle),
  4. `q4 = −π/2 − (q2+q3)` force le poignet à la verticale, `q5 = −π/2`, `q6 = 0`.

Ce solveur sert de **générateur d'étiquettes** pour `train_pinn_6dof.py`.

> ⚠️ À ne pas surinterpréter : le `0.0000 mm` retourné par la fonction est une
> **constante écrite en dur**, pas une erreur mesurée. C'est cohérent (la solution
> est exacte *pour ce modèle simplifié*), mais ce n'est pas une validation numérique.

### 3.3 `ur5e_se3_ik.py` — algèbre de Lie SE(3)

Approche « produit d'exponentielles » (PoE), l'alternative académique aux
paramètres DH :

- `hat_so3(w)` — matrice antisymétrique 3×3 associée à un vecteur (opérateur `^`),
- `exp_so3(w, θ)` — **formule de Rodrigues** : `I + sin(θ)·W + (1−cos θ)·W²`,
- `exp_se3(ξ, θ)` — exponentielle du twist complet (rotation **et** translation),
- `UR5eSE3.forward_kinematics(q)` — `T = exp(ξ₁q₁)···exp(ξ₆q₆)·M`, où `M` est la
  pose de l'effecteur au repos.

> ⚠️ **Point à savoir :** `solve_ik_se3()` ne fait *pas* d'inversion SE(3) — elle
> recopie exactement la trigonométrie de `ur5e_6dof_ik.py`. La contribution SE(3)
> du fichier porte uniquement sur la **cinématique directe**. À présenter tel quel
> dans un rapport, sous peine d'être pris en défaut à la soutenance.

### 3.4 `ikpy_ur5e_solver.py` — le concurrent « maths classiques »

Enveloppe la bibliothèque standard `ikpy` : charge `ur5e.urdf`, masque la base fixe
(`active_links_mask`), appelle `inverse_kinematics()` (résolution numérique
itérative), puis vérifie le résultat par FK et renvoie l'erreur **en mm**.
C'est le baseline auquel le PINN est comparé dans le benchmark.

### 3.5 `ur5e_trajectory.py` — trajectoires quintiques

Classe `QuinticTrajectory`. On ne veut pas envoyer un angle cible brut au moteur
(à-coup violent) : on interpole avec un polynôme de degré 5

`q(t) = a₀ + a₃t³ + a₄t⁴ + a₅t⁵`

choisi pour que **vitesse et accélération soient nulles au départ et à l'arrivée**.
D'où les coefficients classiques `a₃ = 10Δq/T³`, `a₄ = −15Δq/T⁴`, `a₅ = 6Δq/T⁵`.
`get_position(t)` et `get_velocity(t)` échantillonnent la courbe.

*(Note : `ur5.py` réimplémente la même chose en résolvant un système 6×6 avec
`np.linalg.solve` — cette classe-ci est la version autonome et plus lisible.)*

---

## 4. 🧠 `training/` — les scripts d'entraînement

### 4.1 `train_pinn_6dof.py` — l'architecture du réseau

Contient la classe **`PINN6DOF`**, importée par tous les autres scripts et par
`ur5.py`. C'est le fichier à lire en premier.

```
Entrée (3)  →  Linear → SiLU → Linear → SiLU → Linear → SiLU → Linear  →  Sortie (6)
  (x,y,z)              [hidden_dim neurones, 256 ou 512]                 (q1..q6)
```

| Détail | Pourquoi |
| :-- | :-- |
| Activation **SiLU** (Swish) et non ReLU | SiLU est lisse et infiniment dérivable — indispensable quand le gradient doit traverser des `sin`/`cos` dans la loss physique. ReLU introduirait des cassures. |
| Buffers `mean_in/std_in/mean_out/std_out` | Normalisation intégrée **au modèle** : `forward()` normalise l'entrée puis dé-normalise la sortie. Les statistiques sont sauvegardées dans le `.pth`, donc pas besoin de les recharger séparément à l'inférence. Déclarés via `register_buffer` → suivent le modèle sur GPU/CPU sans être des paramètres entraînables. |
| `save_model()` / `from_file()` | Sauvegarde un dictionnaire complet (poids + `hidden_dim` + stats). `from_file()` relit `hidden_dim` et reconstruit l'objet à la bonne taille automatiquement. |

Le `train()` de ce fichier est une **variante** : il entraîne sur les étiquettes du
solveur simplifié `UR5e6DOF`, avec `w_data = 1.0` et `w_phys = 20.0`, 60 époques,
`ReduceLROnPlateau`. Sortie → `pinn_model_ur5e_6dof.pth`.

### 4.2 `train_true_pinn.py` ⭐ — le script principal

C'est **celui qui produit le modèle réellement utilisé** en simulation.

**a) L'astuce du mock Webots (lignes 10-15).** `ur5.py` fait `from controller import
Supervisor`, un module qui n'existe que dans Webots. Le script injecte donc un faux
module dans `sys.modules` **avant** l'import, ce qui permet de réutiliser les
fonctions mathématiques de `ur5.py` (`build_matrix`, `inverse_kinematics`) depuis un
terminal Python normal. Élégant et fréquent en robotique.

**b) Génération du dataset.** 25 000 cibles tirées uniformément dans la zone de la
table Webots (repère robot) :

| Axe | Plage |
| :-- | :-- |
| X | `0.0 → 0.4` |
| Y | `−0.9 → −0.5` |
| Z | `0.05 → 0.45` |

Pour chaque cible on impose l'orientation « pince vers le bas » `[π, 0, −π/2]`, et
on résout avec la **vraie IK analytique du UR5e** en forçant la branche
`wrist='up', shoulder='left', elbow='up'`. Ce forçage est essentiel : il existe
jusqu'à **8 solutions** pour une même position, et si on mélangeait les branches, le
réseau apprendrait leur moyenne — une configuration invalide. On n'apprend donc
qu'**une seule famille de solutions**, cohérente et sans collision avec la table.

**c) La loss hybride — le cœur du sujet.**

```python
pred_q    = model(bx)                       # l'IA devine 6 angles
loss_data = mse(pred_q, by)                 # « ressemble aux angles de référence »
pred_pos  = fk_engine.forward_pos(pred_q)   # où la main atterrit VRAIMENT
loss_phys = mse(pred_pos, bx)               # « la main est-elle sur la cible ? »
loss = 0.1 * loss_data + 1.0 * loss_phys
```

$$\mathcal{L} = 0{,}1\cdot\underbrace{\|q_{pred}-q_{true}\|^2}_{\text{guide la branche}} + 1{,}0\cdot\underbrace{\|\mathrm{FK}(q_{pred})-P_{cible}\|^2}_{\text{impose la physique}}$$

Le poids **10× plus faible sur les données** est délibéré : la partie supervisée
sert seulement de garde-fou pour rester dans la bonne famille de solutions, tandis
que la précision spatiale réelle est portée par la physique. Un réseau purement
supervisé peut avoir une faible erreur d'angle et pourtant rater la cible ; ici,
c'est l'erreur en millimètres qui est optimisée directement.

**d) Boucle d'entraînement.** 100 époques, Adam, `lr` dégradé manuellement
(`1e-3 → 5e-4` à l'époque 30, `1e-4` à 60, `2e-5` à 80). La validation calcule
l'**erreur euclidienne moyenne en mm** et le modèle n'est sauvegardé que s'il
s'améliore (early-save) → `pinn_model_true_physics.pth`.

### 4.3 `train_supervised_ik.py` — la baseline de comparaison

Même dataset, même architecture (512 neurones), mais **loss MSE sur les angles
uniquement**, sans terme physique. Existe précisément pour démontrer par
l'expérience l'apport du terme physique. Sortie → `pinn_model_webots.pth`.

### 4.4 `archive_training/`

`train_pinn.py` et `pinn_simulator.py` : versions historiques sur un bras 3-DOF,
avec un mini serveur HTTP pour la visualisation web. Conservées comme trace de la
démarche, non utilisées par le pipeline actuel.

---

## 5. 🤖 `reference_ur5_repo/ur5.py` — la classe pilote (966 lignes)

Base issue du travail d'**Allan Souza Almeida** (2023), étendue ici avec
l'intégration PINN. C'est le pont entre les maths et Webots.

### 5.1 Fonctions libres (hors classe)

| Fonction | Rôle |
| :-- | :-- |
| `rot_x / rot_y / rot_z(θ)` | Matrices de rotation élémentaires 3×3. |
| `limit_angle(a)` | Ramène un angle dans `[−π, π]`. |
| `build_matrix(pos, rot, euler)` | `(position, angles d'Euler)` → matrice homogène 4×4. Utilise `scipy.spatial.transform.Rotation`. |
| `matrix_error(T1, T2)` | Écart entre deux poses → `(erreur angulaire en °, erreur de position en mm)`. Sert à la validation. |
| `forward_kinematics(θ)` | FK numpy via le tableau DH, produit des 6 matrices avec `reduce(np.dot, A)`. Renvoie `(T_total, A)`. |
| `transform(θ, idx)` | Une seule matrice de passage `T_{i-1,i}`. |
| `inverse_kinematics(T, shoulder, wrist, elbow)` | **IK analytique en forme fermée.** Les 3 arguments sélectionnent la branche parmi les 8 solutions : épaule gauche/droite, poignet haut/bas, coude haut/bas. |

### 5.2 La classe `UR5`

**`__init__`** — instancie le `Supervisor` Webots, lit le `timestep`, puis charge
**deux modèles d'IA** de façon tolérante aux pannes (`try/except` partout, pour que
la simulation démarre même sans TensorFlow ou sans PyTorch) :

1. `computer_vision/vgg16.h5` (Keras) → détection du cube, attribut `self.model` ;
2. `models/pinn_model_true_physics.pth` (PyTorch) → `self.pinn_model`.

L'attribut **`self.use_pinn`** (initialisé à `True`) est l'interrupteur qui décide,
à chaque mouvement, quel solveur est utilisé.

| Méthode | Ce qu'elle fait |
| :-- | :-- |
| `init_handles()` | Récupère les 6 moteurs (`shoulder_pan_joint` … `wrist_3_joint`), la caméra, le nœud du cube (`RED_BOX` ou `bottle`). |
| `setup_control_mode()` | Passe les moteurs en **mode vitesse** (`setPosition(inf)`) et active les capteurs de position — le contrôle fin est fait à la main par la trajectoire. |
| `get_image()` | Image caméra → tableau numpy. |
| `get_joint_angles()` | Lecture des 6 capteurs. |
| `get_ground_truth()` | Pose réelle de l'effecteur donnée par le moteur physique (vérité terrain). |
| `get_jacobian(v)` | Jacobienne géométrique → vitesses linéaires/angulaires de l'effecteur. |
| **`move_to_config(q)`** | Interpolation quintique : construit la matrice 6×6 des conditions aux limites (position/vitesse/accélération à `t₀` et `t_f`), résout avec `np.linalg.solve`, puis pousse les consignes à chaque pas de simulation. Si `duration=None`, la durée est calculée à partir du plus grand déplacement articulaire (minimum 1,5 s). |
| **`move_to_pose(pos, rot, …)`** ⭐ | **L'aiguillage du projet.** Voir ci-dessous. |
| `actuate_gripper(close)` | Même interpolation quintique sur les 9 joints de la pince Robotiq 3F. |
| `predict_bottle_position()` | Pipeline vision complet : image → redimensionnement 256×256 → VGG16 → `(px, py)` → `np.interp` vers les coordonnées réelles → changement de repère monde→robot. |
| `get_bottle_frame()` / `get_node_frame(def)` | Position d'un objet **dans le repère du robot** : `T_objet/robot = (T_robot/monde)⁻¹ · T_objet/monde`. |

**Le cœur de `move_to_pose` :**

```python
if self.use_pinn and self.pinn_model is not None:
    with torch.no_grad():
        pred = self.pinn_model(torch.tensor([[x, y, z]])).numpy()[0]   # 1 forward pass
        q_list = list(pred)
else:
    q_list = inverse_kinematics(T, shoulder=..., wrist=..., elbow='up') # maths
```

Les deux branches sont **chronométrées avec `time.perf_counter()`**, et le temps est
affiché en direct dans le viewport 3D via `supervisor.setLabel()`. C'est ce qui
produit le HUD comparatif de la démo.

> Remarque de lecture : le message console affiche `[IKPY Solver]` dans la branche
> `else`, mais le calcul est en réalité fait par la fonction `inverse_kinematics`
> analytique de `ur5.py`, pas par la bibliothèque IKPY. Libellé trompeur, facile à
> corriger.

---

## 6. 🎬 Les contrôleurs Webots (`simulation/controllers/`)

Un contrôleur = le **script exécuté par un robot** quand la simulation démarre.
Webots le lance automatiquement selon le champ `controller "..."` du `.wbt`.

### 6.1 `ur5_controller_pandahand/` ⭐ — le scénario principal

Séquence complète de pick & place avec la pince parallèle Franka PandaHand :

1. **Argument de mode** — `sys.argv[1]` vaut `"pinn"` ou `"math"`, ce qui permet au
   *même* script de piloter les deux robots du duel avec des solveurs différents.
2. **Configuration neutre** `[0,0,0,0,0,0]`, pince ouverte (`0.04` m par doigt).
3. **Pose de lecture caméra** `[-0.1, -0.68, 0.45]`, en forçant `use_pinn = False` —
   choix pragmatique : les maths sûres pour la manœuvre d'observation, l'IA pour la
   saisie qu'on veut démontrer.
4. **Localisation** de la cible : vision VGG16 si disponible, sinon lecture directe
   de la position par le superviseur (fallback ultime codé en dur `(0.73, 0.2)`).
5. **Pick** : approche à `z = 0.40`, descente à `z = 0.05` en 4 s, fermeture.
6. **Transport** puis **Place** au-dessus du plateau bleu, ouverture.
7. Retour à la pose de lecture.

### 6.2 `ur5_controller/` — la variante Robotiq 3F

Même scénario, sans argument de mode, avec la pince Robotiq à 3 doigts
(`actuate_gripper(0/1)`) et une hauteur de saisie de `0.12` au lieu de `0.05`.

### 6.3 `comparison_controller/` — l'arbitre du duel

Robot superviseur **sans corps** dans `pinn_vs_math.wbt`. Il n'actionne rien : il
affiche les titres HUD (« UR5e A (Rouge) : IA TRUE PINN » / « UR5e B (Vert) : MATH »)
puis tourne en boucle `while robot.step(timestep) != -1` pour rester vivant pendant
toute la simulation.

### 6.4 `multi_controller/` — le comparatif à 3 technologies

Version la plus ambitieuse : trois UR5e côte à côte, chacun avec un solveur
différent, sélectionné d'après le **nom du robot** :

| Robot | Technologie | Fonction appelée |
| :-- | :-- | :-- |
| `…OptionA` | Trigonométrie DH analytique | `inverse_kinematics()` |
| `…OptionB` | Algèbre de Lie SE(3) / PoE | `UR5eSE3.solve_ik_se3()` |
| `…OptionC` | Réseau de neurones PINN | `PINN6DOF.forward()` |

Après chaque résolution, le script recalcule la FK et **imprime l'erreur cartésienne
en mm** — c'est la source des chiffres du benchmark.

### 6.5 `data_collector/` — la fabrique du dataset vision

Déplace le cube rouge à des positions aléatoires sur la table, capture l'image
caméra à chaque fois, et écrit `dataset/images/img_XXXX.jpg` + `dataset/labels.csv`.
C'est ce dataset qui alimente `computer_vision/train_vgg16.ipynb`.

---

## 7. 🌍 Les fichiers de simulation (`simulation/worlds/`)

**Chemin complet de la simulation principale :**

```
reference_ur5_repo\simulation\worlds\my_first_simulation_pandahand.wbt
```

Chemin relatif depuis la racine du projet :
`reference_ur5_repo/simulation/worlds/my_first_simulation_pandahand.wbt`

| Fichier `.wbt` | Contrôleur lancé | À quoi ça sert |
| :-- | :-- | :-- |
| **`my_first_simulation_pandahand.wbt`** ⭐ | `ur5_controller_pandahand` | **La démo principale.** 1 UR5e + PandaHand, cube rouge, plateau bleu. C'est le fichier à ouvrir. |
| `pinn_vs_math.wbt` | `comparison_controller` + 2× `ur5_controller_pandahand` (args `pinn` / `math`) | Le duel côte à côte avec HUD chronométré. |
| `multi_ur5e_comparison.wbt` | 3× `multi_controller` | Comparatif à 3 technologies (DH / SE(3) / PINN). |
| `my_first_simulation.wbt` | `ur5_controller` | Version pince Robotiq 3F. |
| `my_first_simulation_datagen.wbt` | `data_collector` | Génération du dataset de vision. |
| `course.wbt` | — | Monde d'exercice, sans rapport avec le pipeline. |

**Comment lancer :** ouvrir Webots → `File ▸ Open World…` → sélectionner le `.wbt`
→ bouton **Play** ▶.

---

## 8. 📦 `archive_v1_custom_arm/` — le prototype v1

Première itération, sur un bras **anthropomorphe 3-DOF fait maison** (pas un UR5e) :

- `robot_model.py` — FK analytique du bras 3 segments, en numpy et en PyTorch ;
- `pinn_model.py` — réseau `PINNInverseKinematics` (3 → 128 → 3, SiLU) ;
- `interactive_gui.py` — visualisation matplotlib 3D avec sliders, comparant le PINN
  au solveur numérique DLS (Damped Least Squares) ;
- `index.html` — la même démo en page web Three.js (OrbitControls / TransformControls) ;
- `webots/` — mondes et contrôleurs de cette époque, dont un essai d'évitement
  d'obstacle (`pinn_ik_avoidance`).

Utile à conserver : ça montre la progression 3-DOF → 6-DOF industriel. À présenter
explicitement comme une **archive**, pas comme du code actif.

---

## 9. 🚀 Publier sur GitHub

### 9.1 ⚠️ Le point bloquant : la taille

Le projet fait **344 Mo**, dont :

| Élément | Taille | Problème |
| :-- | :-- | :-- |
| `reference_ur5_repo/computer_vision/vgg16.h5` | **160 Mo** | 🔴 GitHub **refuse** tout fichier > 100 Mo. |
| `reference_ur5_repo/dataset/images/` | plusieurs milliers de `.jpg` | 🟠 Alourdit le clone pour rien. |
| `reference_ur5_repo/.git/` | dépôt git imbriqué | 🔴 Casse le suivi de versions du dépôt parent. |

**Trois options pour `vgg16.h5` :**

1. **Git LFS** (le `.gitattributes` du repo de référence le prévoit déjà) :
   `git lfs install && git lfs track "*.h5"` — attention, quota LFS gratuit limité.
2. **L'exclure** et le publier dans une *GitHub Release* (limite 2 Go par fichier),
   avec un lien de téléchargement dans le README. ✅ **Recommandé.**
3. **Le retirer** et ne documenter que le pipeline d'entraînement du VGG16 —
   `ur5.py` gère déjà proprement son absence (le fallback superviseur prend le relais).

Les `.pth` du PINN (2 Mo chacun) passent sans problème et **doivent** être versionnés :
c'est le résultat du projet.

### 9.2 Fichiers ajoutés pour la publication

- **`.gitignore`** — exclut `__pycache__/`, `.venv/`, `.idea/`, `.claude/`, le dépôt
  git imbriqué, le dataset d'images et `vgg16.h5` ;
- **`LICENSE`** — MIT, annoncée dans le README mais jusque-là absente.

### 9.3 Avant de pousser — checklist

- [ ] Supprimer le dépôt git imbriqué : `rm -rf reference_ur5_repo/.git`
- [ ] Vérifier qu'aucun fichier > 100 Mo n'est indexé
- [ ] Corriger dans le README la ligne `git clone https://github.com/your-username/…`
- [ ] Aligner `requirements.txt` et le README (voir §10, point 4)

### 9.4 Commandes

```bash
git init && git add . && git commit -m "Initial commit: PINN inverse kinematics for UR5e in Webots"
```

```bash
git branch -M main && git remote add origin https://github.com/TON-PSEUDO/pinn-ur5e-ik.git && git push -u origin main
```

---

## 10. 🔧 Points fragiles repérés à la lecture

Rien de bloquant pour la démo principale, mais à connaître — surtout avant une
soutenance ou une mise en ligne publique.

1. **`pinn_vs_math.wbt` — noms d'objets incohérents.** Le contrôleur cherche
   `getFromDef("red_box_pinn")`, or dans le monde `red_box_pinn` est le champ `name`
   du nœud, dont le **DEF** est `bottle_pinn`. `getFromDef()` ne matche que les DEF →
   renvoie `None`. Le repli appelle alors `get_bottle_frame()`, qui déréférence
   `self.bottle` lui aussi `None` (ni `RED_BOX` ni `bottle` n'existent dans ce monde)
   → **`AttributeError` probable au lancement**.
   *Correctif :* renommer les DEF en `RED_BOX_PINN` / `RED_BOX_MATH` dans le `.wbt`,
   ou protéger `get_bottle_frame()` par un `if self.bottle is None: return None`.

2. **`train_supervised_ik.py` — chemin d'import erroné.** Le script fait
   `os.path.join(os.path.dirname(__file__), 'reference_ur5_repo')`, soit
   `training/reference_ur5_repo/` qui n'existe pas. Il faut remonter d'un cran,
   comme le fait correctement `train_true_pinn.py`.

3. **Code mort.** `measure_ik_and_move()` dans le contrôleur pandahand (défini,
   jamais appelé) et `set_robot_joint()` dans `comparison_controller.py` (corps
   `pass` + commentaire de réflexion). À supprimer avant publication.

4. **Dépendances incomplètes.** `requirements.txt` liste `flask` et `scikit-learn`
   (non utilisés dans le pipeline actuel) mais **oublie** `ikpy`, `scikit-image`
   (`skimage.transform.resize` dans `ur5.py`) et `tensorflow`/`keras` (chargement du
   VGG16). Le README, lui, cite `ikpy` mais pas les deux autres.

5. **Deux modèles de FK coexistent.** `ur5_pytorch_fk.py` (DH complet, 6 joints) et
   `ur5e_6dof_ik.py` (planaire simplifié). Ils ne donnent pas exactement les mêmes
   positions. Le modèle réellement déployé (`pinn_model_true_physics.pth`) est
   entraîné avec le **DH complet** — c'est le bon choix, mais mieux vaut le dire
   explicitement pour éviter la confusion.

6. **Les chiffres du benchmark** (`~0.44 ms` vs `~0.50–0.85 ms`) sortent de
   `time.perf_counter()` sur des appels uniques, non moyennés et sans warm-up
   PyTorch. L'ordre de grandeur est juste, la précision affichée à trois décimales ne
   l'est pas. Pour un rapport : moyenner sur 1000 itérations après warm-up.

---

## 11. 📋 Antisèche

| Je veux… | Fichier |
| :-- | :-- |
| Lancer la démo | `reference_ur5_repo/simulation/worlds/my_first_simulation_pandahand.wbt` |
| Voir le duel PINN vs Maths | `reference_ur5_repo/simulation/worlds/pinn_vs_math.wbt` |
| Comprendre le PINN | `training/train_true_pinn.py` + `robotics_utils/ur5_pytorch_fk.py` |
| Voir l'architecture du réseau | `training/train_pinn_6dof.py` (classe `PINN6DOF`) |
| Modifier le scénario de saisie | `.../controllers/ur5_controller_pandahand/ur5_controller_pandahand.py` |
| Changer le solveur (IA ↔ maths) | attribut `ur5.use_pinn` (dans `ur5.py`) |
| Réentraîner le modèle | `python training/train_true_pinn.py` |
| Comparer les maths pures | `robotics_utils/` (`ur5e_6dof_ik`, `ur5e_se3_ik`, `ikpy_ur5e_solver`) |

---

## 12. ✅ Audit de complétude du dossier

Vérification faite fichier par fichier : **rien d'essentiel ne manque**. Tout ce que
le code référence existe. Les points ci-dessous sont des incohérences de chemin ou
d'environnement, pas des fichiers absents.

### 12.1 Ce qui est complet ✔

| Vérification | Résultat |
| :-- | :-- |
| Les 5 contrôleurs référencés par les `.wbt` existent | ✔ 5/5 |
| Chaque contrôleur a bien son `.py` au nom du dossier (exigence Webots) | ✔ 5/5 |
| Fichiers `.wbproj` présents pour les 6 mondes | ✔ 6/6 |
| `models/pinn_model_true_physics.pth` chargé par `ur5.py` | ✔ présent, se charge (`hidden_dim=512`) |
| `robotics_utils/ur5e.urdf` pour IKPY | ✔ présent et valide |
| Dataset vision `dataset/images/` + `labels.csv` | ✔ 1000 images |
| `computer_vision/vgg16.h5` | ✔ présent (160 Mo) |
| `computer_vision/train_vgg16.ipynb` | ✔ notebook valide |
| Assets de documentation | ✔ 3 images |
| Fichiers vides / corrompus | ✔ aucun |

*Bonus : `computer_vision/network_architecture` (sans extension) est en fait un
diagramme **draw.io** (`<mxfile>`), ouvrable sur app.diagrams.net.*

### 12.2 Ce qui manque vraiment ❌

**1. L'installation de TensorFlow est corrompue.** Sur le Python utilisé par Webots
(`Python 3.11.9`, celui du Microsoft Store) :

| Paquet | État |
| :-- | :-- |
| `skimage`, `torch`, `numpy`, `scipy`, `ikpy`, `matplotlib` | ✔ installés et fonctionnels |
| `keras` 3.9.2 | ✔ installé, mais inutilisable (dépend de TF) |
| `tensorflow` | ⚠️ **présent mais cassé** |

L'import échoue avec :

```
ModuleNotFoundError: No module named 'tensorflow.python'
```

Trois indices concordants d'une installation interrompue :

1. le dossier `site-packages/tensorflow/` ne contient que **9 entrées** et il lui
   manque le sous-paquet `python/`, qui est le cœur de TensorFlow ;
2. **aucun `tensorflow-*.dist-info`** n'existe — pip n'a pas trace du paquet, alors
   que ses dépendances (`tensorflow_estimator` 2.15.0,
   `tensorflow_io_gcs_filesystem` 0.31.0) ont bien la leur ;
3. `keras` 3.9.2 est présent et sain, mais plante en cascade sur le même import.

Autrement dit : un `pip install tensorflow` (~600 Mo) coupé en cours de route.

**Conséquence dans le projet.** `ur5.py` enveloppe `from keras.models import load_model`
dans un `try/except`, donc l'erreur est absorbée silencieusement : `self.model` reste
à `None` et **le contrôleur ne passe jamais par la vision** — il lit directement la
position du cube via le superviseur Webots. La simulation tourne parfaitement, mais
la brique « Computer Vision » n'est pas exercée, et les 160 Mo de `vgg16.h5` sont
pour l'instant du poids mort.

**Réparation.** ⚠️ `pip uninstall tensorflow` est inopérant ici : sans dist-info, pip
ignore que le paquet existe. Il faut supprimer le dossier orphelin à la main, puis
réinstaller proprement :

```bash
rm -rf "$LOCALAPPDATA/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages/tensorflow"
pip uninstall -y tensorflow-estimator tensorflow-io-gcs-filesystem keras
pip install tensorflow
```

Point de vigilance : `numpy` est en 2.2.6 alors que les résidus pointent vers
TF 2.15, qui exige `numpy<2`. Une réinstallation propre récupère TF ≥ 2.16,
compatible numpy 2 — d'où l'intérêt de repartir de zéro plutôt que de rafistoler.

**2. `requirements.txt` est faux.** Il déclare `flask` et `scikit-learn` (non
utilisés) et **oublie** trois dépendances réellement importées :

```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
torch>=1.9.0
ikpy>=3.3                 # ← manquant (robotics_utils/ikpy_ur5e_solver.py)
scikit-image>=0.19        # ← manquant (ur5.py, import NON protégé ligne 16)
tensorflow>=2.10          # ← manquant (chargement du VGG16)
opencv-python             # ← manquant (data_collector)
```

⚠️ L'import `from skimage.transform import resize` en tête de `ur5.py` **n'est pas**
dans un `try/except`, contrairement à ceux de keras et torch. Si `scikit-image`
manque sur une autre machine, **tous les contrôleurs plantent au démarrage**. C'est
la dépendance la plus critique du projet, et la seule non protégée.

### 12.3 Chemins incohérents (fichiers présents, mais introuvables par le code) ⚠

**3. Les scripts d'entraînement sauvegardent au mauvais endroit.**
`train_true_pinn.py` fait `model.save_model("pinn_model_true_physics.pth")` — chemin
**relatif au répertoire courant**. Lancé depuis la racine, le `.pth` atterrit à la
racine, alors que `ur5.py` le cherche dans `models/`. Il faut donc le déplacer à la
main après chaque entraînement. Correctif d'une ligne :

```python
os.path.join(os.path.dirname(__file__), '..', 'models', 'pinn_model_true_physics.pth')
```

**4. `multi_controller` ne trouve jamais son modèle.** Il cherche
`pinn_model_ur5e_6dof.pth` à la racine du projet ou de `reference_ur5_repo/`, or le
fichier est dans `models/archive_models/`. Le `except` silencieux fait alors
basculer le **Robot C sur `inverse_kinematics()` analytique**. Autrement dit, dans
`multi_ur5e_comparison.wbt`, les trois robots tournent aujourd'hui sur des maths —
**le comparatif à 3 technologies ne démontre pas ce qu'il annonce**.

À savoir aussi : ce modèle-là a été entraîné contre la FK **planaire simplifiée**
(`ur5e_6dof_ik.py`), pas contre la FK DH complète. Mesuré contre le DH complet, son
erreur est de ~970 mm — les deux conventions ne sont pas interchangeables.

**5. `training/archive_training/` a des imports cassés.** `train_pinn.py` et
`pinn_simulator.py` font `from robot_model import UR5e3Axis` et
`from pinn_model import PINNInverseKinematics`, mais ces deux fichiers sont dans
`archive_v1_custom_arm/`, sans aucun `sys.path` pour les relier. Ces scripts ne
peuvent pas s'exécuter tels quels. Comme c'est de l'archive, le plus honnête est de
le noter dans un `README` local plutôt que de les réparer.

**6. Mélange de versions Webots.** `multi_ur5e_comparison.wbt` charge ses PROTO
depuis la branche **R2025a**, les 5 autres mondes depuis **R2023a**. Sur une
installation R2023a, ce monde peut refuser de se charger. Aligner tout sur la
version de Webots effectivement installée.

**7. `runtime.ini` absent pour 3 contrôleurs.** Seuls `ur5_controller` et
`ur5_controller_pandahand` en ont un (contenu : `[python]`). `multi_controller`,
`comparison_controller` et `data_collector` n'en ont pas. Sans conséquence tant que
le Python par défaut de Webots convient, mais c'est plus robuste d'en ajouter un
identique dans chaque dossier.

### 12.4 Mesure de performance des modèles

Benchmark exécuté sur **4814 cibles valides** (tirées dans la zone d'entraînement,
filtrées par le même critère que `train_true_pinn.py` — 80 % des tirages bruts sont
atteignables), erreur mesurée contre la FK DH complète :

| Modèle | Erreur moyenne | Médiane | P95 | Max |
| :-- | ---: | ---: | ---: | ---: |
| `pinn_model_true_physics.pth` (PINN hybride) | 0,187 mm | 0,164 mm | 0,406 mm | 1,31 mm |
| `pinn_model_webots.pth` (baseline supervisée) | **0,058 mm** | **0,052 mm** | **0,117 mm** | **0,31 mm** |

Les deux sont excellents (sous le dixième de millimètre pour la baseline), mais
**la baseline supervisée est ~3× plus précise en position que le PINN hybride**.

C'est logique une fois posé : le dataset est généré en **forçant une seule branche
IK** (`wrist='up', shoulder='left', elbow='up'`), donc la fonction à apprendre est
déjà univoque et propre — précisément le cas où l'apprentissage supervisé pur
excelle. Le terme physique (`w_data=0,1`) laisse au réseau une liberté dans l'espace
nul dont il n'avait pas besoin ici.

⚠️ **À ne pas ignorer avant une soutenance** : le README affirme qu'un réseau
supervisé « moyenne les 8 solutions et produit des configurations invalides ».
C'est vrai *en général*, mais **pas dans ce projet**, puisque le dataset ne contient
qu'une seule branche. En l'état, les chiffres contredisent l'argumentaire.

Trois façons de rendre la comparaison honnête et défendable :

1. **Entraîner le PINN sans étiquettes** (`w_data = 0`). C'est le vrai argument du
   PINN : il n'a pas besoin d'un solveur IK préexistant. Si le résultat reste sous
   le millimètre, la démonstration est bien plus forte.
2. **Générer le dataset sans forcer la branche** et montrer que le supervisé
   s'effondre là où le PINN tient — ce serait la reproduction de l'argument annoncé.
3. **Changer de métrique** : mesurer la continuité de trajectoire (sauts articulaires
   entre deux cibles voisines) plutôt que la seule précision de position. C'est là
   que le PINN a un avantage réel et mesurable.

---

## 13. 🔨 Corrections appliquées

Les points de l'audit §12 ont été corrigés. **Le monde qui fonctionnait
(`my_first_simulation_pandahand.wbt`) n'a pas été touché** — seuls les fichiers
défectueux ont été modifiés.

### 13.1 Ce qui débloque les mondes qui ne marchaient pas

| # | Fichier | Correction |
| :-: | :-- | :-- |
| 1 | `simulation/worlds/pinn_vs_math.wbt` | `DEF bottle_pinn` → **`DEF RED_BOX_PINN`**, `DEF bottle_math` → **`DEF RED_BOX_MATH`**. Le contrôleur cherchait des DEF qui n'existaient pas → `AttributeError` au lancement. |
| 2 | `controllers/ur5_controller_pandahand/` | `getFromDef("red_box_pinn")` → `getFromDef("RED_BOX_PINN")`, aligné sur les DEF du monde et sur la convention déjà utilisée pour `BLUE_TRAY_*`. |
| 3 | `ur5.py` — `get_bottle_frame()` | Ajout d'un garde `if self.bottle is None: return None`. La méthode déréférençait `self.bottle` sans vérification, ce qui transformait un objet introuvable en crash au lieu d'un repli propre. |
| 4 | `controllers/multi_controller/` | Le modèle était cherché à la racine du projet alors qu'il est dans `models/`. Remplacé par une **liste de chemins candidats**, avec `models/pinn_model_true_physics.pth` en priorité. Le Robot C exécute enfin réellement le réseau de neurones. |

> Sur le point 4 : l'ancien `pinn_model_ur5e_6dof.pth` a été entraîné contre la FK
> **planaire simplifiée**, alors que `multi_controller` mesure l'erreur avec la FK
> **DH complète** — deux conventions incompatibles (~970 mm d'écart). D'où la
> priorité donnée à `pinn_model_true_physics.pth`, seul modèle cohérent avec la
> métrique utilisée. ⚠️ Reste à vérifier que les cibles de ce monde tombent dans la
> zone d'entraînement (`x∈[0;0,4]`, `y∈[−0,9;−0,5]`, `z∈[0,05;0,45]`) : hors de
> cette zone, le réseau extrapole et l'erreur explose.

### 13.2 Robustesse et portabilité

| # | Fichier | Correction |
| :-: | :-- | :-- |
| 5 | `ur5.py` — ligne 16 | `from skimage.transform import resize` placé dans un `try/except ImportError`. C'était le **seul import non protégé** : sans `scikit-image`, les 5 contrôleurs plantaient au démarrage. `predict_bottle_position()` lève maintenant une erreur explicite si la vision est réellement demandée. |
| 6 | `requirements.txt` | Réécrit : ajout de `ikpy`, `scikit-image`, `tensorflow`, `opencv-python` ; retrait de `flask` et `scikit-learn` (inutilisés). Chaque ligne est commentée avec le fichier qui en dépend. |
| 7 | 3 contrôleurs | Ajout du `runtime.ini` manquant (`multi_controller`, `comparison_controller`, `data_collector`) — les 5 contrôleurs sont maintenant homogènes. |

### 13.3 Chemins de sauvegarde des modèles

| # | Fichier | Correction |
| :-: | :-- | :-- |
| 8 | `training/train_true_pinn.py`<br>`training/train_supervised_ik.py`<br>`training/train_pinn_6dof.py` | Les trois scripts écrivaient leur `.pth` **dans le répertoire courant**, alors que `ur5.py` le lit dans `models/`. Ajout d'un helper `model_path()` qui résout systématiquement vers `pinn_ik_project/models/`. Plus besoin de déplacer le fichier à la main après un entraînement. |
| 9 | `training/train_supervised_ik.py` | Chemin d'import corrigé : pointait vers `training/reference_ur5_repo/` (inexistant) au lieu de remonter d'un cran. Le script était inexécutable. |

### 13.4 Nettoyage

| # | Fichier | Correction |
| :-: | :-- | :-- |
| 10 | `controllers/ur5_controller_pandahand/` | Suppression de `measure_ik_and_move()` — définie, jamais appelée, avec un commentaire signalant qu'elle ne mesurait pas ce qu'elle prétendait. |
| 11 | `controllers/comparison_controller/` | Suppression du stub `set_robot_joint()` (corps `pass`), de `distance()` (inutilisée) et des imports morts. |
| 12 | `ur5.py`, les 2 contrôleurs HUD | Libellé `[IKPY Solver]` / `MATH (IKPY)` → **`[IK Analytique]`**. Le solveur « maths » est la fonction `inverse_kinematics()` en forme fermée de `ur5.py`, pas la bibliothèque IKPY. Le libellé faussait la lecture du benchmark. |
| 13 | `README.md` | `pip install -r requirements.txt`, note sur `vgg16.h5` distribué en Release, ajout du guide à l'arborescence. |
| 14 | racine | Ajout de `.gitignore` et `LICENSE` (MIT annoncée dans le README mais absente). |

### 13.5 Vérifications post-correction

- ✅ Les 20 fichiers `.py` actifs compilent (`py_compile`)
- ✅ `import ur5` fonctionne hors Webots (avec le mock `controller`)
- ✅ `multi_controller` résout bien `models/pinn_model_true_physics.pth`
- ✅ Les DEF cherchés par les contrôleurs correspondent aux DEF des mondes (`pinn_vs_math` et `multi_ur5e_comparison`)
- ✅ `pinn_vs_math.wbt` : accolades équilibrées (45/45)
- ✅ `my_first_simulation_pandahand.wbt` : **non modifié**

### 13.6 Ce qui reste à ta main

**a) Réparer l'installation TensorFlow** pour activer la brique vision (voir §12.2 :
le paquet est présent mais incomplet, `pip uninstall` ne suffit pas). Sans cela,
`vgg16.h5` reste inutilisé et le pipeline lit la position du cube via le superviseur.

**b) Versions de PROTO Webots.** Webots **R2025a** est installé sur cette machine.
`multi_ur5e_comparison.wbt` référence les PROTO R2025a ; les 5 autres mondes
référencent R2023a. Comme ils fonctionnent tels quels (Webots télécharge les PROTO
depuis GitHub), **je n'y ai pas touché** — les aligner serait un risque inutile.
À noter seulement : ces mondes **nécessitent une connexion Internet** au premier
chargement pour récupérer les PROTO distants.

**c) `training/archive_training/`** : imports cassés (`robot_model`, `pinn_model`
vivent dans `archive_v1_custom_arm/`). Laissé tel quel — c'est de l'archive
historique, la réparer donnerait l'illusion d'un code maintenu.

**d) Le benchmark du §12.4** reste le point le plus important à traiter avant une
soutenance : en l'état, la baseline supervisée est plus précise que le PINN, ce qui
contredit l'argumentaire du README.

---

## 14. 👁️ Réparation de la brique Vision

### 14.1 Le vrai diagnostic

Deux problèmes empilés, et non un seul comme le laissait croire le §12.2 :

**a) Deux environnements Python.** La machine en a deux, et Webots utilisait le mauvais :

| | Python système (Microsoft Store) | **Anaconda** |
| :-- | :-- | :-- |
| TensorFlow | ⚠️ installation tronquée (`tensorflow/python` absent, pas de dist-info) | ✅ 2.21.0 fonctionnel |
| torch, numpy, scipy, cv2 | ✅ | ✅ |
| skimage, ikpy | ✅ | ❌ (avant réparation) |

Le VGG16 avait été entraîné sous Anaconda — d'où l'impression légitime que « la
vision était déjà installée ». Elle l'était : simplement pas dans l'interpréteur
que Webots lançait.

**b) Rupture Keras 2 → Keras 3.** Même avec un TensorFlow sain, le chargement
échouait :

```
AttributeError: Exception encountered when calling Flatten.call().
'list' object has no attribute 'shape'
```

`vgg16.h5` a été sauvegardé au format **legacy Keras 2.x**. Keras 3 ne sait pas
relire ce type de modèle (un `Sequential` enveloppant une base VGG16 fonctionnelle).
Réparer TensorFlow seul n'aurait donc **pas suffi** : les deux environnements ont
Keras 3.

### 14.2 Ce qui a été fait

**1. Paquets installés dans Anaconda** (~30 Mo) :

```bash
/c/Users/<utilisateur>/anaconda3/python.exe -m pip install tf-keras scikit-image ikpy
```

`tf-keras` 2.21.0 est le paquet de compatibilité **Keras 2**, seul capable de relire
le `.h5` legacy. `scikit-image` et `ikpy` complétaient les dépendances manquantes.

**2. Les 5 `runtime.ini` pointent désormais vers Anaconda :**

```ini
[python]
COMMAND = C:\Users\<utilisateur>\anaconda3\python.exe
```

**3. `ur5.py` — import Keras compatible 2 et 3.** Nouvelle fonction
`import_load_model()` qui essaie `tf_keras` puis `keras`, et **affiche pourquoi** la
vision est désactivée le cas échéant. C'est ce silence qui masquait le problème
depuis le début.

**4. `ur5.py` — chargement paresseux du modèle.** Importer TensorFlow coûte ~28 s sur
cette machine. Le faire au niveau du module faisait attendre **chaque contrôleur**
avant le moindre mouvement, même sans vision. `UR5.model` est devenue une
**propriété** qui ne charge qu'au premier accès réel, avec cache :

| | Avant (import global) | Après (paresseux) |
| :-- | ---: | ---: |
| Démarrage d'un contrôleur | 27,8 s | **8,8 s** |
| Premier accès à `.model` | 0 s | 23,3 s |
| Accès suivants | 0 s | **0,000 s** |

Un setter est conservé : `ur5.model = None` force le repli superviseur sans toucher
au code.

### 14.3 Vérifications

- ✅ `tf_keras` charge `vgg16.h5` : entrée `(None, 256, 256, 3)` → sortie `(None, 2)`, 23 136 706 paramètres
- ✅ Inférence testée sur une image factice
- ✅ Anaconda (Python 3.13) importe le module Webots `controller` — les bindings R2025a sont en Python pur, donc compatibles
- ✅ `PINN6DOF` se charge sous Anaconda (0,228 mm sur une cible test)
- ✅ Les 5 `runtime.ini` pointent vers un exécutable existant
- ✅ Propriété `model` : chargement unique, cache effectif, setter fonctionnel

### 14.4 Changement de comportement à connaître ⚠️

La vision étant maintenant **active**, `ur5_controller_pandahand` ne prend plus le
repli superviseur : il appelle `predict_bottle_position()`, qui localise le cube
**par la caméra**. Deux conséquences :

1. La conversion pixels → coordonnées réelles repose sur des bornes calibrées en dur
   dans `ur5.py` (`xlim = [40, 465]`, `ylim = [81, 395]`, et les plages réelles
   associées). Si la caméra ou la table ont bougé depuis l'entraînement du VGG16, le
   robot visera à côté.
2. `predict_bottle_position(show_img=True)` ouvre une fenêtre matplotlib pendant 2 s
   à chaque détection.

**Pour revenir instantanément au comportement d'avant** (celui qui fonctionnait),
au choix :

- commenter la ligne `COMMAND` dans le `runtime.ini` concerné → retour au Python
  système, vision désactivée ;
- ou ajouter `ur5.model = None` juste après `ur5 = UR5()` dans le contrôleur.

C'est le seul changement susceptible d'affecter `my_first_simulation_pandahand.wbt`,
qui fonctionnait déjà. À tester en priorité.

---

## 15. 📷 Vision : recalibration complète (Option A)

### 15.1 Pourquoi l'ancienne vision ne pouvait pas marcher

Trois défauts empilés, mesurés et non supposés :

1. **La caméra ne regardait pas le cube.** Analyse des images générées (détection
   des pixels rouges, croisée avec la position de téléportation) :

   | Position monde Y | Images | Cube visible |
   | :-- | ---: | ---: |
   | −0,50 → −0,30 | 47 | **100 %** |
   | −0,30 → −0,20 | 22 | 59 % |
   | −0,20 → −0,10 | 16 | 6 % |
   | −0,10 → +0,30 | 76 | **0 %** |

   Le cube de la démo est à **y = +0,20** : jamais dans le champ. Le plateau bleu
   à y = −0,20 : à la limite. L'ancien dataset (y ∈ [−0,325 ; −0,075]) tombait
   déjà majoritairement dans la zone aveugle.

2. **Les étiquettes étaient calculées, pas mesurées.** `data_collector` déduisait
   la position en pixels par interpolation linéaire depuis la position monde,
   avec des constantes (`xlim`, `ylim`, `x_real_lim`, `y_real_lim`) héritées d'un
   autre monde. Une étiquette pouvait donc désigner un cube absent de l'image.

3. **Contrainte mécanique.** La pose de lecture est à **0,822 m** de la base pour
   une portée UR5e de ~0,85 m. Impossible d'éloigner ou de surélever la caméra
   pour voir toute la table : le seul levier est son **orientation**.

### 15.2 La nouvelle approche : le robot se calibre lui-même

`data_collector.py` a été entièrement réécrit. Il ne suppose plus rien, il mesure.

| Phase | Ce qu'elle fait |
| :-- | :-- |
| **0 — Choix de la pose** | Essaie 6 poses de lecture atteignables (positions et orientations variées). Pour chacune, balaye une grille 5×5 sur la table et compte les points où le cube est effectivement visible. Retient la meilleure couverture. |
| **1 — Calibration** | À la pose retenue, balaye une grille 9×9. Pour chaque position **connue** du cube, **détecte** ses pixels rouges dans l'image. Ajuste par moindres carrés la transformation affine `[x, y]_monde = A · [u, v, 1]`, et délimite la zone réellement visible. |
| **2 — Collecte** | Tire 1000 positions **uniquement dans la zone visible**. Étiquette chaque image par le **centroïde mesuré**. Rejette les images où le bras masque le cube. |

Sorties dans `reference_ur5_repo/dataset/` :

- `images/*.jpg` — 256×256, l'entrée du réseau
- `labels.csv` — `filename, x_pixel, y_pixel` (repère 256×256, plus 512)
- `calibration.json` — pose de lecture, matrice affine, zone visible

### 15.3 Ce qui change dans `ur5.py`

Les quatre constantes en dur ont disparu. `predict_bottle_position()` :

1. lit `calibration.json` (mis en cache par `vision_calibration()`) ;
2. redimensionne l'image à `image_size` et interroge le réseau → `(u, v)` ;
3. convertit en coordonnées monde par la matrice affine calibrée ;
4. **avertit si la prédiction sort de la zone calibrée** — c'est exactement le cas
   qui, auparavant, envoyait silencieusement le bras hors de portée ;
5. passe dans le repère robot en utilisant la vraie hauteur du cube (`cube_z`),
   là où l'ancienne version mettait `z = 0`.

Le contrôleur reprend désormais la **pose de lecture enregistrée dans la
calibration** au lieu d'une valeur en dur : une calibration n'est valable qu'à la
pose exacte où elle a été mesurée.

Enfin, `load_vision_model()` tente réellement le **chargement** avec `tf_keras`
puis `keras` — et non seulement l'import. L'ancien `.h5` (format Keras 2) et un
modèle réentraîné aujourd'hui (Keras 3) fonctionnent donc tous les deux.

### 15.4 Procédure

1. **Régénérer le dataset** — ouvrir `my_first_simulation_datagen.wbt` et lancer.
   Le script affiche la couverture de chaque pose candidate, la RMSE de la
   calibration et la zone visible retenue. Compter quelques minutes (6 × 25 +
   81 + 1000 placements).
2. **Réentraîner** — exécuter `computer_vision/train_vgg16.ipynb` sous Anaconda.
   Le notebook lit `labels.csv` sans modification.
3. **Activer** — passer `USE_VISION = True` en tête de
   `ur5_controller_pandahand.py`.
4. **Tester** — lancer `my_first_simulation_pandahand.wbt`. La console affiche
   l'erreur en millimètres entre position prédite et position réelle à chaque
   détection.

### 15.5 Le point à surveiller

La phase 0 peut conclure que **même la meilleure pose ne voit qu'une partie de la
table**. C'est une limite mécanique, pas un défaut de code : à 0,822 m sur 0,85 m
de portée, le bras ne peut pas prendre plus de recul.

Si la zone visible retenue ne contient pas la position du cube de la démo
(monde x −1,55 / y +0,20), deux issues :

- **déplacer le cube** dans la zone visible en éditant le `.wbt` — le plus simple,
  et parfaitement légitime : c'est un banc d'essai ;
- **ajouter des orientations** dans `CANDIDATE_POSES` (en tête de
  `data_collector.py`) pour incliner davantage la caméra.

Le script affiche la zone retenue : la comparaison est immédiate.

---

## 16. 🎯 Modèle de caméra exact (remplace la §15.2)

La §15 devinait où regardait la caméra en essayant 6 poses à la main. Ça ne
pouvait pas marcher : la caméra n'est pas au bout de l'outil mais **décalée sur
le poignet**, comme le montre le `toolSlot` du `.wbt` :

```
Transform { translation -2.9e-07 0.08 0.068   rotation 0 0 1 1.5708
  JetBotRaspberryPiCamera { translation 0 0 0.03  fieldOfView 1.5708 } }
```

8 cm de côté, 6,8 cm en avant, plus une rotation propre. On ne calcule pas ça de
tête. `data_collector.py` a donc été réécrit autour d'un **modèle de caméra
mesuré**.

### 16.1 Les quatre phases

| Phase | Ce qu'elle fait | Coût |
| :-- | :-- | :-- |
| **A — Modèle** | `Supervisor.getFromDevice(camera)` donne la pose **réelle** de la caméra dans le monde. Comparée à celle de l'outil, elle donne la transformation rigide outil→caméra, constante. On identifie aussi la convention d'axes Webots en comparant projection théorique et cube réellement détecté, sur 35 placements. | ~35 placements |
| **B — Orientation** | Grâce à A, prédire ce que voit la caméra pour une pose donnée n'est plus qu'un calcul : **ni mouvement, ni rendu**. On balaie 1863 combinaisons position/orientation atteignables et on garde celle qui voit le plus de table, en privilégiant celles qui voient le cube de la démo. | **~9 s** |
| **C — Vérification** | À la pose retenue, on place le cube en 36 points connus et on compare projection théorique et détection réelle (l'écart valide toute la chaîne). Puis on ajuste une **homographie** pixel→monde. | 36 placements |
| **D — Collecte** | 1000 images, cube tiré **uniquement dans l'empreinte visible**. Étiquette = centroïde mesuré. | 1000 placements |

### 16.2 Pourquoi une homographie et pas une transformation affine

Une caméra inclinée qui observe un plan produit une transformation **projective**
(homographie, 8 DDL). L'ajustement affine de la §15 (6 DDL) ne peut pas
représenter la perspective : il a un biais systématique, d'autant plus grand que
la caméra est inclinée. L'homographie est le modèle **exact** pour une scène
plane. `ur5.py` l'applique désormais, tout en restant compatible avec les
calibrations affines antérieures.

### 16.3 Validation hors simulation

La géométrie a été testée sans Webots, sur une caméra synthétique verticale
placée à 0,45 m au-dessus du centre de la table :

| Vérification | Résultat |
| :-- | :-- |
| Aller-retour `project` → `backproject` | **0,000000 mm** d'erreur |
| Empreinte prédite | 0,84 m de côté (théorie : 2 × 0,45 × tan 45° − marges) ✅ |
| Couverture de la table | 60 % |
| **Cube de la démo (y = +0,20) visible** | ✅ **oui** |

C'est le point important : **si le poignet peut approcher la verticale, le cube
est dans le champ.** Ce n'est donc pas perdu d'avance — toute la question est de
savoir quelle orientation le bras peut réellement atteindre, et c'est exactement
ce que la phase B détermine.

### 16.4 Ce qu'affiche la console

- phase A : décalage outil→caméra mesuré, et RMS de chaque convention d'axes
  testée (la bonne doit sortir nettement en dessous des autres) ;
- phase B : nombre d'orientations atteignables, et le top 5 avec couverture et
  visibilité du cube ;
- phase C : écart projection/détection en pixels, puis RMSE de l'homographie en
  millimètres ;
- phase D : progression et nombre d'images rejetées (cube masqué par le bras).

### 16.5 Si aucune pose ne voit le cube

Le script le dit explicitement et **calcule la position de repli** :

```
Le cube de la demo n'est PAS dans le champ de la camera.
Deplacer DEF bottle dans les .wbt vers le centre de la zone visible :
    translation -1.500 -0.150 0.955
```

Il suffit alors de reporter cette ligne dans `DEF bottle` des `.wbt` concernés.
C'est un banc d'essai : déplacer l'objet à saisir est parfaitement légitime, et
bien plus honnête que de prétendre détecter un cube hors champ.

---

## 17. 🎨 Perception : seuillage couleur plutôt que CNN

### 17.1 Les deux réseaux, à ne pas confondre

| Réseau | Rôle | Fichier | État |
| :-- | :-- | :-- | :-- |
| **PINN** (PyTorch) | `(x,y,z)` → 6 angles articulaires | `pinn_model_true_physics.pth` | ✅ 0,2 mm sur la zone de saisie |
| **VGG16** (Keras) | image → pixel du cube | `vgg16.h5` | ⚠️ ~49 mm |

Le PINN ne voit aucune image et ne reconnaît rien : c'est le solveur de cinématique
inverse, le cœur du projet. Seul le VGG16 fait de la perception.

### 17.2 Mesures sur les mêmes 200 images de validation

| Méthode | Erreur sur la table | Saisie réussie (cube 50 mm) |
| :-- | ---: | ---: |
| VGG16 (`block4_pool` + `Flatten`, 250 époques) | 48,6 mm | **17,5 %** |
| **Seuillage couleur** | **11,6 mm** | **100 %** |

Détail du seuillage : 0 échec de détection sur 200, 0,79 px d'erreur moyenne
(du bruit de compression JPEG — en simulation l'image n'est pas compressée, donc
c'est encore meilleur), soit 4,7 mm avant l'homographie.

### 17.3 Pourquoi le CNN perd contre une méthode triviale

Les étiquettes du dataset ont été produites **par ce seuillage même**. Le réseau ne
peut donc pas dépasser son professeur — c'est attendu.

Ce qui rend le CNN inutile ici n'est pas d'être moins précis, c'est que **le
professeur est disponible gratuitement à l'inférence** : même caméra, même image,
une milliseconde de calcul. On distille normalement une méthode dans un réseau
quand elle n'est plus disponible ensuite (annotation manuelle, capteur coûteux,
traitement hors ligne). Ce n'est pas le cas.

À quoi s'ajoute que VGG16 figé, pré-entraîné sur ImageNet, est conçu pour dire
« il y a un objet », pas « il est à ce pixel près ».

### 17.4 Ce qui a été mis en place

- `UR5.detect_cube_color()` — seuillage rouge, centroïde, puis conversion par
  l'homographie calibrée. Renvoie `None` si le cube est hors champ ou masqué.
- `UR5.pixel_to_robot()` — conversion pixel → repère robot, partagée par les deux
  détecteurs (plus de code dupliqué).
- `VISION_MODE` en tête de `ur5_controller_pandahand.py` :

| Valeur | Comportement |
| :-- | :-- |
| `"color"` | seuillage couleur — **défaut** |
| `"cnn"` | VGG16, pour la comparaison |
| `"supervisor"` | position exacte du moteur Webots (vérité terrain, pas de la perception) |

### 17.5 Le facteur limitant a changé

L'erreur totale de 11,6 mm se décompose ainsi :

```
détection couleur    4,7 mm
homographie         10,1 mm   ← désormais dominant
                   -------
total               11,6 mm
```

Ce n'est plus la perception qui limite, mais la **calibration**. Cause probable
identifiée : le cube mesure 10 cm de haut alors que l'homographie suppose une
scène plane. Le centroïde des pixels rouges est celui de la face visible, pas de
l'empreinte au sol, et avec une caméra inclinée de 0,3 rad la parallaxe vaut
≈ 5 cm × tan 17° ≈ 15 mm — le bon ordre de grandeur.

Piste si tu veux descendre sous 5 mm : ajuster l'homographie sur le **bas** du
cube (le bord inférieur du masque rouge) plutôt que sur son centroïde.

### 17.6 Un bug d'affichage corrigé au passage

La cellule 12 du notebook divisait les coordonnées par 2 — héritage de l'époque où
`data_collector` écrivait les étiquettes en repère 512. Les nouvelles sont en 256,
la même échelle que l'image affichée. Le `/2` décalait donc **aussi la croix verte
de la vérité terrain**, ce qui faisait croire à une erreur du modèle là où il n'y
avait qu'un défaut d'affichage. Corrigé, et la cellule montre maintenant trois
exemples avec l'erreur en pixels et en millimètres.

---

## 18. 🎥 Commander la camera, pas un repere invisible

### 18.1 Le probleme

`move_to_pose([x, y, z], ...)` place le **bout de la chaine DH** -- un repere
purement mathematique defini par `d6 = 0.0996 + 0.1237` dans la table DH de
`ur5.py`. Ce n'est ni le bout des doigts, ni la camera.

La camera est vissee sur le cote du poignet :

```
Transform { translation 0 0.08 0.068   rotation 0 0 1 1.5708
  JetBotRaspberryPiCamera { translation 0 0 0.03 } }
```

Mesure de la phase A : elle se trouve a **10,8 cm** du point commande, dans une
direction qui tourne avec le poignet. Commander une position ne place donc pas
l'objectif ou on croit.

### 18.2 La correction

Dans le repere robot, la camera se trouve en :

```
position_camera = R(rot) . decalage_local + position_outil
```

d'ou, en inversant :

```
position_outil = position_camera - R(rot) . decalage_local
```

C'est ce que fait `UR5.move_camera_to()`. Verification numerique de l'aller-retour :
**0,000000 mm** d'erreur.

### 18.3 Usage

```python
# Avant -- on commande un repere invisible
ur5.move_to_pose([-0.05, -0.65, 0.50], [PI, 0.45, -PI/2], wrist='up')
#   la camera finit ~10,8 cm ailleurs

# Apres -- on commande l'objectif
ur5.move_camera_to([-0.05, -0.65, 0.50], [PI, 0.45, -PI/2], wrist='up')
#   [Camera] cible objectif [-0.05 -0.65  0.5] -> outil commande [-0.031 -0.551  0.462]
```

Les 108 mm d'ecart entre les deux lignes sont exactement le decalage du montage.

`move_camera_to` accepte les memes arguments que `move_to_pose` (`wrist`,
`shoulder`, `duration`) et renvoie la position d'outil effectivement commandee.

### 18.4 Ce qui a change dans le code

| Fichier | Modification |
| :-- | :-- |
| `data_collector.py` | la phase A mesurait deja la transformation outil -> camera mais la **jetait**. Elle est desormais enregistree sous la cle `tool_to_camera` de `calibration.json`. |
| `ur5.py` | `camera_offset()` relit la matrice 4x4 ; `move_camera_to()` applique la correction. |

### 18.5 Prerequis

`tool_to_camera` n'existe pas dans les calibrations anterieures. Tant que le
monde `my_first_simulation_datagen.wbt` n'a pas ete relance, `move_camera_to()`
leve une erreur explicite plutot que de viser a cote :

```
KeyError: 'tool_to_camera' absent de calibration.json :
relancer le monde my_first_simulation_datagen.wbt pour le mesurer.
```

---

## 19. ✅ Etat final de la chaine de perception

Les chapitres 15 a 18 racontent des approches successives, dont plusieurs ont
ete abandonnees. Voici ce qui est reellement en place.

### 19.1 Mesures

| Grandeur | Debut | Final |
| :-- | ---: | ---: |
| Inclinaison de la camera | 17,20 deg | **0,07 deg** |
| Derive du bras apres deplacement | 63 mm | **0,00 mm** |
| Residu de l'homographie (median) | 38 mm (RMSE) | **1,1 mm** |
| Residu apres correction radiale | -- | **0,9 mm** |
| RMSE | 38,0 mm | **1,6 mm** |
| Erreur de localisation du cube | 148 mm | **~1 mm** |

La pince tolere 25 mm. La perception est donc **25 fois plus precise** que
necessaire.

### 19.2 Les cinq causes qui bloquaient

1. **Commande en boucle ouverte.** `move_to_config` pilotait les moteurs en
   vitesse puis s'arretait, sans correction : jusqu'a 114 mm d'ecart entre pose
   commandee et pose atteinte. La trajectoire quintique se termine desormais par
   un asservissement en position.

2. **Camera penchee de 17 deg.** Le deuxieme angle de `ROT_CAMERA` etait
   exactement son inclinaison ; le passer a 0 la redresse. Identifie en lancant
   un rayon depuis chacun des 6 axes locaux et en regardant lequel touche la
   table -- ce que deux ajustements numeriques successifs n'avaient pas su
   trouver.

3. **Pas de simulation trop grossier.** `basicTimeStep` au defaut de 32 ms dans
   5 mondes sur 6 : contacts mal resolus, bras qui vibre pendant les mesures.
   Porte a 8 ms partout.

4. **Doigts de la pince non pilotes.** `init_handles()` vide `finger_joints`
   puis parcourt cette liste vide : aucun doigt n'etait jamais commande. Ils
   ballottaient au gre des mouvements, multipliant les contacts.

5. **Cube trop haut.** 10 cm de haut sous une camera a 44 cm : le centroide des
   pixels rouges glissait vers l'exterieur du champ, jusqu'a 56 mm au bord.
   Ramene a 5 cm, ce qui divise le deport par deux ; l'homographie absorbe le
   reste.

### 19.3 Comment ca marche maintenant

```
image 256x256
     |
  seuillage rouge -> centroide en pixels
     |
  homographie (ajustee sur 91 mesures)
     |
  correction radiale (si elle ameliore d'au moins 10 %)
     |
  changement de repere monde -> robot
     |
  position (x, y) a ~1 mm pres
```

Le champ de vue n'est plus cherche mais **calcule** : sous une camera verticale
a hauteur h avec un champ FOV, c'est un disque de rayon `h*tan(FOV/2)`, soit
0,433 m ici. Le cube n'est tire que dans ce disque intersecte avec la table.

Pour verifier de ses yeux : `View > Optional Rendering > Show Camera Frustums`
dessine le cone de vision dans la scene.

### 19.4 Ce qui reste

La perception est reglee. Le maillon suivant est la **saisie** : la pince ne
s'etait pas encore refermee correctement sur le cube lors des derniers essais.
`ur5_controller_pandahand.py` contient de quoi diagnostiquer -- capteurs de
position des doigts, verification que le cube a bien ete souleve, et recherche
automatique de la hauteur de saisie (`CALIBRER_HAUTEUR`).
