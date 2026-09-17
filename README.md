# 🤖 Physics-Informed Neural Network (PINN) for 6-DOF UR5e Industrial Robot Inverse Kinematics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Webots R2023a+](https://img.shields.io/badge/webots-R2023a+-green.svg)](https://cyberbotics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end robotic automation and inverse kinematics (IK) solver for the **Universal Robots UR5e 6-DOF industrial manipulator**, leveraging a **True Physics-Informed Neural Network (PINN)** combined with a **Computer Vision target recognition pipeline** in a high-fidelity **Webots 3D physics simulation**.

![UR5e Single Robot Simulation](assets/webots_single_robot.png)

---

## 📌 Key Highlights

- **🧠 True Physics-Informed Loss**: the predicted angles are pushed through a differentiable PyTorch Denavit–Hartenberg forward kinematics, so the gradient travels through the robot's geometry and the quantity minimised is **millimetres of end-effector error** — not similarity to a reference answer. **0.185 mm** position, **0.009°** orientation on held-out targets.
- **🔬 The central claim is ablated, not asserted**: a 2 × 3 experiment (single IK branch vs all 8 mixed, $w_{phys} \in \{0, 1, 20\}$) measures what the physics term is actually worth. [Results below](#-does-the-physics-term-actually-help).
- **⚡ Sub-Millisecond Execution**: IK predictions in **0.25 ms** (median, warm), **182× faster than the iterative numerical solver** it replaces (IKPY, 45 ms) — and honestly reported as **11 % slower** than the closed-form analytic solution.
- **👁️ Perception rebuilt to 1 mm**: camera-to-world localisation of the target went from **148 mm** to **~1 mm** by replacing a mis-trained CNN with a calibrated colour detector plus homography — a documented negative result on the CNN.
- **🤖 Vision & Actuation Integration**: end-to-end pick-and-place with the Franka Emika PandaHand in Webots, closed-loop on joint position.

---

## 📐 System Architecture

The robot operational pipeline connects perception, spatial resolution, physics-informed IK, and actuation:

```
[ Webots Camera ] ──► [ Computer Vision Model ] ──► Cartesian Target Pos (X, Y, Z)
                                                                 │
                                                                 ▼
[ Webots Joint Motors ] ◄── [ 6-DOF Joint Angles ] ◄── [ True PINN Neural Net ]
```

### 1. Perception Layer (Computer Vision)
The robot positions its camera at a elevated reading pose `[-0.1, -0.68, 0.45]`. The vision model processes the camera stream to locate the target object (red box) and outputs its 3D Cartesian coordinates relative to the robot base.

### 2. Resolution Layer (Physics-Informed IK)
The 3D coordinate $(X, Y, Z)$ is fed into the trained `PINN6DOF` network, which
infers the 6 joint angles $(q_1, \ldots, q_6)$ of the arm.

> **Scope, stated precisely.** The manipulator has 6 degrees of freedom, but the
> network solves a **position-only IK problem at a fixed tool orientation**: it
> is a $\mathbb{R}^3 \rightarrow \mathbb{R}^6$ map, and the tool orientation is
> held at $[\pi, 0, -\pi/2]$ (gripper pointing straight down) throughout
> training. This is the natural restriction for top-down pick-and-place, and it
> is what makes a 530 k-parameter network sufficient. Full 6-DoF *pose* IK would
> require feeding the orientation in as an input and regenerating the dataset —
> it is **not** what this repository demonstrates.

### 3. Actuation Layer (Webots Simulation)
The joint angles are executed via smooth cubic polynomial trajectories (`ur5.move_to_config`), closing the PandaHand parallel gripper on the target and transferring it to the destination tray.

---

## 🧮 Mathematical Formulation of True PINN Loss

A purely supervised network is trained to *resemble* reference joint angles. It
has no notion of a robot, so a small error on $q_2$ counts the same as a small
error on $q_6$ — although the first moves the hand by centimetres and the second
by a hair.

The PINN is corrected on **where its hand actually lands**. The predicted angles
are pushed through a Denavit–Hartenberg forward kinematics rewritten in PyTorch
(`robotics_utils/ur5_pytorch_fk.py`), which makes it a differentiable link in the
computation graph: the gradient travels through the geometry of the robot, and
the quantity being minimised is **millimetres of end-effector error**.

$$\mathcal{L} = w_{data}\underbrace{\|q_{pred} - q_{ref}\|^2}_{\text{stay on the right branch}} + w_{phys}\Big(\underbrace{\|\mathbf{p}_{pred} - \mathbf{p}_{target}\|^2}_{\text{position}} + r_{tool}^2\underbrace{\|\mathbf{R}_{pred} - \mathbf{R}_{ref}\|_F^2}_{\text{orientation}}\Big)$$

where $\mathbf{p}_{pred}, \mathbf{R}_{pred} = \text{FK}_{PyTorch}(q_{pred})$, and
$w_{data} = 0.1$, $w_{phys} = 1.0$, $r_{tool} = 0.05\,\text{m}$.

Three implementation choices worth naming:

- **The orientation term is compared against $\text{FK}(q_{ref})$**, not against a
  matrix rebuilt from Euler angles — both sides pass through the *same* forward
  kinematics, so no angle convention can introduce a spurious gap. Without this
  term the loss is blind to orientation: rotating $q_6$ by 180° moves the point
  by zero microns but flips the gripper, and both poses score identically.
- **Frobenius norm rather than geodesic distance**, whose $\arccos$ has a
  gradient that diverges near zero — exactly where the network must converge.
- **$r_{tool}^2$ makes the two terms commensurable**: a position error is in
  m², a rotation error is dimensionless. Weighting by the square of a
  characteristic tool length converts a 1° rotation into the 0.9 mm of tip
  displacement it actually causes.

| Metric | Value |
| :-- | ---: |
| Position error, validation set (2 002 held-out targets) | **0.185 mm** |
| Orientation error, validation set | **0.009°** |
| Position error measured *in Webots* at the grasp point † | **0.30 mm** |

The last row is the only one that really counts: it is obtained by driving to the
same target with the closed-form solver and then with the PINN, and comparing the
poses **actually reached** (`mesurer_erreur_pinn()` in the controller) — so it
includes the DH model, the motor servo loop and the physics engine, not just the
network's internal consistency.

† Measured on the model *before* the orientation term was added. The current
model is better on validation, but has not been re-measured in Webots; the
0.30 mm figure is reported as-is rather than silently attributed to the newer
weights.

---

## 🔬 Does the physics term actually help?

An earlier version of this README asserted that *"a standard supervised network
averages the 8 IK solutions, resulting in invalid joint configurations, and our
PINN resolves this ambiguity."* That claim was **not tested by the experiment**:
both dataset generators filter to a single solution family before training, so
there was never anything to average. The claim is now replaced by a measurement.

`training/ablation_physics_loss.py` runs a 2 × 3 design — everything held
constant (same seed, same 20 201 targets, same architecture, same 100 epochs,
same LR schedule, same selection criterion) except the two variables:

| Training data | $w_{phys}=0$ (supervised) | $w_{phys}=1$ (this repo) | $w_{phys}=20$ |
| :-- | ---: | ---: | ---: |
| **One IK branch** (what this repo ships) | 0.238 mm / 0.012° | **0.187 mm / 0.011°** | 0.180 mm / 0.044° |
| **All 8 branches mixed** | 911 mm / 99.0° | 69.6 mm / 98.6° | 3.68 mm / 71.6° |

*Position / orientation error on 2 021 held-out targets. All branches were
verified by forward kinematics to reach the same pose to within 0.0003 mm and
0.04°, so the two rows differ only in joint configuration, not in task.*

Three conclusions, and the second one contradicts what this README used to say:

1. **On the data this project actually uses, the physics term is worth 21 %** —
   0.238 mm down to 0.187 mm. Real, reproducible, and far more modest than
   "invalid joint configurations". Note also that $w_{phys}=20$ buys a further
   0.007 mm of position at the cost of **4× worse orientation**, which is why
   $w_{phys}=1$ is the shipped setting.

2. **The premise was right, but it was never this repo's situation.** Trained on
   mixed branches, the supervised network does collapse to 911 mm — it predicts
   the mean configuration, which reaches nowhere near the target. That failure is
   real. It is simply not the failure this project avoids, because the single
   branch is forced in the *data generator*, not by the loss.

3. **The physics term mitigates the ambiguity; it does not resolve it.** At
   $w_{phys}=1$ the mixed-branch model never improves past its first epoch. At
   $w_{phys}=20$ it reaches 3.68 mm — a 250× improvement over supervised, but
   still 20× worse than the single-branch model and with 71.6° of orientation
   error, i.e. unusable for grasping.

**The engineering takeaway:** constraining the dataset to one solution family is
what makes this problem tractable; the physics-informed loss then buys a further
21 % and a well-behaved orientation. Both matter, and neither substitutes for the
other.

```bash
python training/ablation_physics_loss.py     # ~1 h on CPU, writes models/ablation_physics_loss.json
```

---

## 📂 Project Directory Structure

```
pinn_ik_project/
├── models/
│   └── pinn_model_true_physics.pth     # Trained PyTorch PINN model weights
├── robotics_utils/
│   ├── ur5_pytorch_fk.py               # Differentiable PyTorch Forward Kinematics module
│   └── ur5e_6dof_ik.py                 # Geometric UR5e kinematics utilities
├── training/
│   ├── train_true_pinn.py              # Main training script with hybrid loss (0.1 Data + 1.0 Physics)
│   ├── train_pinn_6dof.py              # Synthetic workspace dataset generator & model architecture
│   ├── train_supervised_ik.py          # Supervised baseline training script
│   └── ablation_physics_loss.py        # 2x3 ablation: is the physics term worth it?
├── reference_ur5_repo/
│   ├── ur5.py                          # Main robot controller wrapper & IK bridge
│   └── simulation/
│       ├── controllers/
│       │   ├── ur5_controller_pandahand/ # Primary Pick & Place controller script
│       │   └── comparison_controller/   # Supervisor HUD controller for benchmarks
│       └── worlds/
│           ├── my_first_simulation_pandahand.wbt # Single UR5e Pick & Place world
│           └── pinn_vs_math.wbt                  # Comparative benchmark world (PINN vs IKPY)
├── archive_v1_custom_arm/               # Prototype custom 3-DOF web arm project
├── GUIDE_DES_CODES.md                   # Detailed file-by-file walkthrough (FR)
├── DOCUMENTATION.md                     # Technical report (FR)
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/clementdumeril/pinn-ur5e-ik.git
cd pinn-ur5e-ik
pip install -r requirements.txt
```

> **Note on the vision model.** The CNN weights (`computer_vision/vgg16.h5`,
> 37 MB) are **not** tracked here, and you do not need them: the default
> perception mode is `VISION_MODE = "color"`, a colour-threshold detector that
> needs no model and is *more* accurate than the CNN on this task — 8 mm versus
> 49 mm on the same 200 validation images. The CNN is kept in the repository as
> a documented negative result (see below); `training/` contains everything
> needed to retrain it. Without TensorFlow installed, `ur5.py` simply stays in
> colour mode and the simulation runs end to end.

### 2. Training the True PINN

To train the PINN model from scratch using physics-informed loss:

```bash
python training/train_true_pinn.py
```

The trained model checkpoint will be saved to `models/pinn_model_true_physics.pth`.

### 3. Running Webots 3D Simulations

#### Option A: Single UR5e Pick & Place (True PINN)
Launch Webots and open the main simulation world:
```
reference_ur5_repo/simulation/worlds/my_first_simulation_pandahand.wbt
```
Press **Play** in Webots. The UR5e arm will perform top-down visual recognition, calculate the IK via the PINN network, grab the red box, and place it on the blue tray.

#### Option B: Real-Time Comparative Benchmark (PINN vs Analytic Math)
Launch Webots and open the benchmark world:
```
reference_ur5_repo/simulation/worlds/pinn_vs_math.wbt
```
Press **Play** in Webots to observe two UR5e robots operating simultaneously:
- **Red Robot (Left)**: Driven by **True PINN Neural Network**.
- **Green Robot (Right)**: Driven by **Analytic IKPY Matrix Math**.
- **HUD Ticker**: Displays live computation time in milliseconds (`ms`) for both algorithms on the upper-left corner of the 3D viewport.

---

## 📊 Performance Benchmarks

Measured over 1000 timed calls, after discarding 200 warm-up calls, single
thread, on 300 reachable targets drawn from the training workspace.

| Metric | IKPY (iterative) | Closed-form analytic | **True PINN** |
| :--- | ---: | ---: | ---: |
| **Median compute time** | 45.0 ms | **0.223 ms** | 0.248 ms |
| Mean compute time | 56.4 ms | 0.280 ms | 0.322 ms |
| Position error at grasp point | — | exact | **0.30 mm** |
| Differentiable | ❌ | ❌ | **✅** |
| Continuous in the target | ❌ (8 branch flips) | ❌ (8 branch flips) | **✅** |

**Read this table honestly.** The PINN is **not** faster than the closed-form
analytic solution — it is 11 % slower, and no neural network will beat a few
dozen trigonometric operations. The first call additionally costs ~4.5 ms while
PyTorch warms up.

What the PINN *does* buy, and what this project demonstrates:

- **182× faster than IKPY**, the iterative numerical solver it actually
  replaces — 0.25 ms against 45 ms;
- **differentiable**, so it can sit inside an end-to-end learning pipeline,
  which neither of the other two methods allows;
- **continuous** in the target position, where analytic solvers jump between
  their 8 solution branches.

An earlier version of this table claimed 0.35–0.45 ms for the PINN against
0.50–0.85 ms for the analytic solver. That was measured without warm-up on a
handful of calls, and was wrong in both directions.

---

## ⚠️ Limitations

Stated plainly, because a reviewer will find them anyway and they are cheap to
measure:

| Limitation | Evidence |
| :-- | :-- |
| **Extrapolates poorly outside the training box.** Above the drop-off tray ($x = -0.20$, outside the learned $[0, 0.4]$), error rises to 7–11 mm. | `mesurer_erreur_pinn()` output, four scenario waypoints |
| **One tool orientation only.** The gripper cannot approach from the side; that would need orientation as a network input and a regenerated dataset. | `train_true_pinn.py`, `ROT_DOWN` hard-coded |
| **One IK branch only.** The network cannot route around an obstacle via the elbow-down family. | dataset generator forces `wrist='up', shoulder='left', elbow='up'` |
| **Not faster than the closed-form solver** — 0.248 ms against 0.223 ms, 11 % slower. The speed claim is against IKPY, not against trigonometry. | benchmark table above |
| **The grasp height sits 20 mm below the trained $z$ range** ($z = 0.030$ versus a learned $[0.05, 0.45]$). Error stays at 0.30 mm, so it is not urgent, but retraining on $z \in [0.02, 0.45]$ would be cleaner. | `GRASP_Z` versus the generator bounds |

## 🔬 How the numbers were obtained

Several figures in this repository replaced earlier ones that were wrong. The
method that caught them is worth more than any single result:

- **Every timing discards 200 warm-up calls** and pins `torch.set_num_threads(1)`.
  The first published table compared a cold PyTorch against a warm NumPy and was
  wrong in both directions.
- **Accuracy is measured on the robot, not on the model.** Verifying a PINN by
  running its output back through the same forward kinematics only tests internal
  consistency. The reported error compares poses *actually reached* in the
  physics engine against a closed-form solver on identical targets.
- **The measurement campaign runs after the scenario, never before.** Placed
  first, it nudged the cube and the demo then used stale coordinates — a bug that
  looked exactly like "the PINN does not work".
- **Claims are tested by ablation, not asserted.** See the section above.

---

## 📜 License & Citation

Distributed under the MIT License. See `LICENSE` for details.

---

## Prerequisites

Beyond `pip install -r requirements.txt`, two things cannot ship in a Git repository:

- **Webots R2023a or later** (tested on R2025a).
- **An internet connection on first launch** — the worlds pull their PROTO
  definitions (`UR5e`, `PandaHand`, `Table`, ...) from GitHub via `EXTERNPROTO`.

The controllers use whichever `python` is on your PATH. If that interpreter
lacks the dependencies, add a `COMMAND` line to
`simulation/controllers/*/runtime.ini`:

```ini
[python]
COMMAND = C:/path/to/your/python.exe
```

Note that Webots does **not** support `#` comments in `runtime.ini` — it reads
them as unknown keys and warns about each one.

## What is *not* in this repository

| Excluded | Size | Why it does not matter |
| :-- | ---: | :-- |
| `computer_vision/vgg16.h5` | 37 MB | The default perception mode is `VISION_MODE = "color"`, a colour-threshold detector needing no model. Measured on the same 200 validation images: **8 mm** error versus **49 mm** for the CNN. Only `VISION_MODE = "cnn"` needs these weights. |
| `dataset/images/` | 20 MB | 1000 training images for the CNN only. Regenerate by running `my_first_simulation_datagen.wbt` with `COLLECTER_IMAGES = True`. |

`dataset/calibration.json` **is** included — it holds the camera pose and the
pixel-to-world homography, and the colour detector cannot work without it.
