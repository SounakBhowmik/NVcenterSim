# NV-QML Magnetometry  
**Quantum-state regression for NV-center–inspired magnetic field sensing**

---

## Table of Contents
- [Overview](#overview)
- [Scientific Context](#scientific-context)
  - [Why NV centers?](#why-nv-centers)
  - [Key idea of this project](#key-idea-of-this-project)
- [What this project does](#what-this-project-does)
- [Project Structure](#project-structure)
- [Physical Model](#physical-model)
- [Sensing Protocol](#sensing-protocol)
- [Learning Tasks](#learning-tasks)
- [Running the Experiment](#running-the-experiment)
- [Interpreting the Results](#interpreting-the-results)

---

## Overview

This project implements a **controlled benchmark for quantum machine learning (QML) in quantum sensing**, using an **NV-center–inspired magnetometry task** as the physical setting.

The core question studied is:

> **How does learning from quantum sensing states differ from learning from classical measurement records derived from those states?**

Rather than comparing *quantum models vs classical models*, this project compares **learning under different levels of data access**:

- **Classical data**: finite-shot, noisy measurement outcomes (fluorescence-style readout)
- **Quantum data**: pre-measurement quantum states produced by the sensing protocol

The sensing dynamics and noise are grounded in an **NV-center–specific noise and readout model**, while the learning task is formulated as a **regression problem for magnetic field estimation**.

---

## Scientific Context

### Why NV centers?

Nitrogen-vacancy (NV) centers in diamond are a canonical platform for **quantum sensing**, particularly magnetometry.

In Ramsey-type NV magnetometry:

- An external magnetic field \( B \) is encoded as a **relative quantum phase**
- This phase resides in **quantum coherence**
- Fluorescence readout converts coherence into a **classical signal**, inevitably losing information

This makes NV magnetometry an ideal testbed for studying **measurement-induced information loss** in learning pipelines.

---

### Key idea of this project

> **Quantum machine learning becomes meaningful when the data itself is quantum.**

This project demonstrates, in a minimal and controlled setting, that:

- Learning from quantum states enables lower estimation error than learning from fixed classical readouts
- The performance gap arises from **measurement bottlenecks**, not from model expressivity

---

## What this project does

We simulate a **single-qubit NV-center sensing protocol** (Ramsey interferometry) under realistic NV-like decoherence and readout noise.

An unknown magnetic field \( B \) is encoded as a phase during free evolution at several evolution times.  
We generate **paired datasets** consisting of:

1. **Quantum data** – the density matrix of the sensing qubit *before measurement*
2. **Classical data** – finite-shot measurement outcomes after a fixed readout

Both datasets are used to train simple regression models that estimate \( B \), and their performance is compared across different shot budgets.

---

## Project Structure
NVcentreMagnetometry/\
│\
├── nvquantum.py # NVQuantum simulator (external library)\
│\
├── nvqml/ # Core experiment modules\
│ ├── init.py\
│ ├── config.py # Experiment configuration\
│ ├── backend.py # NVBackend construction (noise + readout)\
│ ├── circuits.py # Ramsey sensing circuits\
│ ├── simulator.py # Circuit execution + density matrix extraction\
│ ├── features.py # Classical & quantum feature extraction\
│ ├── dataset.py # Dataset generation\
│ └── ridge.py # Ridge regression baseline\
│\
└── run_nv_magnetometry.py # Main experiment runner

Each file has a **single responsibility**, making the code easy to debug and extend.

---

## Physical Model

### Effective NV Hamiltonian

We work in the **effective two-level electronic spin subspace** relevant for Ramsey magnetometry:

\[
H = \gamma_e B S_z
\]

where:
- \( B \) is the external magnetic field
- \( \gamma_e \approx 28\,\text{GHz/T} \) is the electron gyromagnetic ratio

This term encodes the sensing signal.  
Decoherence and readout imperfections are handled separately by the noise model.

---

### Noise and readout

The simulation uses NV-specific imperfections provided by `nvquantum.py`:

- \( T_1 \) and \( T_2 \) relaxation
- Thermal relaxation noise
- Fluorescence-style readout error (5–15%)
- Finite-shot sampling

These effects define the **classical measurement bottleneck** studied in this work.

---

## Sensing Protocol

### Ramsey sequence (per evolution time)

For each evolution time \( t_k \):

1. Prepare \(  |0⟩)
2. Apply Hadamard \( H \)
3. Apply phase rotation  
   \[
   R_z(\phi_k), \quad \phi_k = 2\pi \gamma_e B t_k
   \]
4. Apply Hadamard \( H \)
5. **Snapshot the quantum state**
6. Measure in the computational basis

---

### Why multiple evolution times?

We use **3–5 evolution times** (default: 5):
t = [5, 10, 20, 40, 60] µs


This:
- Breaks phase ambiguity
- Improves robustness to noise
- Mirrors practical NV sensing protocols

---

## Learning Tasks

### Target

Estimate the magnetic field \( B \).

---

### Classical learning (ML-C)

**Input features**

Estimated probabilities from measurement outcomes:

\[
[p_0(t_1), p_0(t_2), \dots, p_0(t_M)]
\]

**Model**
- Ridge regression (linear, interpretable baseline)

---

### Quantum-data learning (QML-Q)

**Input features**

Bloch-vector components extracted from the density matrix:

\[
[x_1, y_1, z_1, \dots, x_M, y_M, z_M]
\]

where:
\[
x = \mathrm{Tr}(\rho \sigma_x), \quad
y = \mathrm{Tr}(\rho \sigma_y), \quad
z = \mathrm{Tr}(\rho \sigma_z)
\]

**Model**
- Ridge regression (same model class as ML-C)

This isolates the effect of **data access**, not model complexity.

---

## Running the Experiment

### Quick sanity check

Edit `run_nv_magnetometry.py`:

```python
cfg = ExperimentConfig(
    n_train=200,
    n_test=80,
    shots_list=(64,),
)

# Full experiment (paper-ready)
shots_list = (16, 32, 64, 128, 256, 512, 1024)
n_train = 2000
n_test = 500
```

```
Run:
python run_nv_magnetometry.py

Example output:
=== Summary ===\
shots=  64 | mse_c=1.45e-14 | mse_q=8.99e-23\
shots= 128 | mse_c=1.21e-14 | mse_q=6.78e-23\
```


## Interpreting the Results

#### Classical learning is limited by:
* Fixed measurement basis
* Shot noise
* Readout error
#### Quantum-state learning retains:

* Phase information
* Coherence across evolution times

The gap shrinks or disappears when classical measurements are enriched (e.g., X/Y/Z bases), confirming that the advantage arises from measurement-induced information loss.
