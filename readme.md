# Deep Learning for Probabilistic Slope Stability Analysis

This repository contains a fully automated, end-to-end Machine Learning pipeline for probabilistic geotechnical slope stability analysis. It uses **Physics-Informed Neural Networks (PINNs)** to solve groundwater seepage, **Limit Analysis (Log-Spiral)** to evaluate structural collapse, and a **Multi-Modal Convolutional Neural Network (CNN)** to act as an ultra-fast Universal Surrogate for massive Monte Carlo simulations.

## 📂 Project Structure

```text
├── data/
│   ├── raw/               # Step 1: Raw generated spatial fields + scalars (.npz)
│   ├── processed/         # Step 2: Individual physics checkpoints (.npz)
│   ├── train/             # Step 3: Shuffled training master arrays (X, S, y)
│   ├── test/              # Step 3: Unseen testing master arrays (X, S, y)
│   ├── models/            # Step 4: Timestamped CNN weights (.pth)
│   └── simulated/         # Step 5: Final Monte Carlo FoS distributions & Plots
├── src/
│   ├── config.py              # Geotechnical and architectural hyperparameters
│   ├── random_fields.py       # Lognormal spatial variability generator
│   ├── pinn_seepage.py        # GPU-accelerated Seepage Solver
│   ├── limit_analysis.py      # Kinematic Limit Analysis (Upper Bound)
│   ├── generate_raw_data.py   # (Step 1) Universal batch field generation
│   ├── process_data.py        # (Step 2) Parallelized physics calculation & checkpointing
│   ├── compile_data.py        # (Step 3) Train/Test splitting & compilation
│   ├── cnn_surrogate.py       # (Step 4) Two-Headed Surrogate Model training 
│   ├── monte_carlo.py         # (Step 5) Producer-Consumer GPU reliability & parametric sweeps
│   └── plot_surface.py        # Visualization tool for critical/sub-critical surfaces
└── main.py                # Pipeline Orchestrator
```

## 🚀 The 5-Step Universal Pipeline

You can run the entire pipeline start to finish using the `--all` flag, or execute individual steps to distribute the computational load.

### Step 1: Raw Data Generation
Generates reproducible, spatially variable random soil fields. It randomly selects geometry (e.g., slope angle) and soil strengths from `config.py` to build a "Universal" dataset, bundling the spatial tensor and physical scalars together.
```bash
python main.py --raw --num_samples 400
```

### Step 2: Physics Processing (Parallelized Checkpointing)
Wakes up the GPU for PINN seepage and the CPU (Numba JIT) for limit analysis. Uses a capped **ProcessPoolExecutor** to run multiple Python workers in parallel, maximizing Numba CPU threading without overflowing the GPU VRAM. 
**Batching Feature:** You can process specific chunks of data in separate terminal windows to parallelize the workload further!
```bash
python main.py --process --process_range 0 100
python main.py --process --process_range 100 200
```

### Step 3: Dataset Compilation
Sweeps through your `.npz` checkpoints, securely randomizes/shuffles the data, and performs an 80/20 split. It generates three perfectly synced arrays: Images (`X`), Scalars (`S`), and Targets (`y`).
```bash
python main.py --compile
```

### Step 4: Multi-Modal CNN Surrogate Training
Trains the "Two-Headed" Deep Convolutional Neural Network. Head 1 (CNN) processes the spatial blotchiness, while Head 2 (MLP) explicitly processes the physics rules (Beta, Cohesion, Friction).
* **Early Stopping:** Monitors validation loss (patience=200) to prevent overfitting.
* **Timestamped Archives:** Automatically saves models with unique datetimes (e.g., `cnn_model_20260427_173834.pth`).
* **Unseen Evaluation:** Automatically evaluates against the `test/` dataset (calculating MSE & MAE) immediately after training.
```bash
python main.py --train --epochs 5000
```

### Step 5: Monte Carlo Parametric Sweep
Performs massive-scale probabilistic analysis. This step uses a high-speed **Producer-Consumer Architecture**: isolated CPU workers constantly generate random soil fields and push them into a multiprocessing queue, while the GPU instantly consumes and evaluates them through the CNN.
* **Auto-Discovery:** Automatically finds and loads the most recently trained CNN model.
* **Parametric Sweeps:** Automatically loops through multiple slope configurations (e.g., $25^\circ, 35^\circ, 45^\circ$) to analyze how design changes affect reliability.
* **Headless Plotting:** Automatically generates and saves an overlapping Probability Density KDE plot to `data/simulated/` upon completion.
```bash
python main.py --mc --mc_samples 100000
```

### Run Everything
```bash
python main.py --all --num_samples 400 --mc_samples 100000 --epochs 5000
```

---
## 🎛️ Command Line Arguments Reference

Below are the arguments supported by `main.py` to control the pipeline execution.

| Argument | Type | Default | Description / Notes |
| :--- | :---: | :---: | :--- |
| `--all` | Flag | `False` | Executes all 5 steps (Raw -> Process -> Compile -> Train -> MC) sequentially. |
| `--raw` | Flag | `False` | Executes Step 1: Generates raw random fields. |
| `--process` | Flag | `False` | Executes Step 2: Solves physics and saves checkpoints in parallel. |
| `--compile` | Flag | `False` | Executes Step 3: Splits and compiles train/test `.npy` arrays. |
| `--train` | Flag | `False` | Executes Step 4: Trains the CNN and evaluates on the test set. |
| `--mc` | Flag | `False` | Executes Step 5: Runs Monte Carlo parametric sweeps. |
| `--plot` | Flag | `False` | Utility: Launches the Matplotlib visualizer. Does NOT run with `--all`. |
| `--num_samples` | `int` | `400` | Number of soil fields to generate during Step 1. |
| `--epochs` | `int` | `5000` | Maximum training epochs for Step 4 (Early stopping may halt this sooner). |
| `--mc_samples` | `int` | `100000` | Total number of spatial fields evaluated *per slope angle* during Step 5. |
| `--process_range` | `int int` | `None` | Processes a targeted slice of data in Step 2 (e.g., `--process_range 0 50`). |
| `--model_path` | `str` | `Auto` | Optional: Points to a specific `.pth` file. If omitted, auto-finds the newest model. |
| `--plot_indices` | `int(s)` | `0` | Space-separated list of array indices to visualize (e.g., `--plot_indices 5 10`). |

---

## 📊 Visualization Tools

A built-in geotechnical analysis tool to visually inspect how the algorithm routes slip surfaces through heterogeneous soil. 

Instead of running a separate script, you can trigger the visualization directly from the main pipeline using the `--plot` utility flag. 

**Features:**
* Plots the absolute **Critical Slip Surface** (thick dashed blue line).
* Plots a gradient of **Sub-Critical Surfaces** to visualize the "search space" and competing failure mechanisms (Base, Toe, Face).
* Supports batch viewing: Pass a list of dataset indices, and closing one Matplotlib window automatically opens the next.

**Usage:**
```bash
# Inspect a single slope (e.g., Sample 42)
python main.py --plot --plot_indices 42

# Inspect multiple specific slopes in sequence
python main.py --plot --plot_indices 0 15 150 399
```

---

## ⚙️ Universal Model Configuration
To modify the range of slopes and soils your model learns, edit the **Universal Dataset Parameters** inside `src/config.py`:
* Add to `SLOPE_ANGLES` to train the CNN on different embankment geometries.
* Add to `COHESION_MEANS` or `FRICTION_MEANS` to expand the material behaviors (e.g., lower cohesion values simulate sand/gravel, higher values simulate clay).
* *Note: Changing geometry or spatial parameters (`L_H`, `L_V`) requires regenerating the data (Step 1) and reprocessing the physics (Step 2).*