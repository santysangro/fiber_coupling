# Automated Fiber Coupling

## Overview

This project implements an automated fiber-coupling system using five servo motors (two per mirror and one z-translation stage) and a PicoScope feedback signal. The system is capable of:

* Recovering from fiber-coupling misalignments caused by drift.
* Performing blind fiber coupling from an unknown starting position.
* Optimizing optical power coupled into a single-mode fiber.
* Operating through Bayesian Optimization combined with local iterative correction.

The optimization process uses Latin Hypercube Sampling (LHS) for initial exploration, a Gaussian Process (GP) model to learn the coupling landscape, and an Upper Confidence Bound (UCB) acquisition function to identify promising regions of the search space.

---

## Hardware

* SCServo motors with URT-1 controller board (serial communication)
* PicoScope used as the optimization feedback signal
* Two-mirror beam steering system
* Fiber coupling setup

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```text
controller/
│
├── Picoscope.py
├── ServoMotors.py
└── FiberCoupling.py
```

Contains the hardware interfaces and optimization loop:

* PicoScope communication
* Servo motor control
* Fiber-coupling optimization routines

```text
model/
│
├── Data acquisition
├── Latin Hypercube Sampling (LHS)
└── Gaussian Process implementation
```

Contains the sampling strategies and machine-learning models used during optimization.

Alternatively, optimization can be performed using M-LOOP:

```text
m_loop/
└── run_experiment_fiber_coupling.py
```

```text
view/
```

Graphical user interface providing:

* Read/write servo positions
* Real-time PicoScope voltage monitoring
* Blind fiber coupling
* Drift correction

---

## Usage

### Drift Correction Workflow

```text
Initial Sampling
        ↓
Train Gaussian Process (GP)
        ↓
Bayesian Optimization (UCB)
        ↓
Identify High-Coupling Region
        ↓
Iterative Local Correction
        ↓
Final Validation
```

The drift-correction routine performs a complete optimization cycle.

Initially, a set of measurements is collected using Latin Hypercube Sampling and used to train a Gaussian Process model of the coupling landscape.

The Gaussian Process then performs a user-defined number of Bayesian Optimization iterations using an Upper Confidence Bound acquisition function to identify regions likely to contain improved coupling.

Once a promising region has been identified, a local iterative correction routine is executed. The algorithm first performs a sweep of the z-position and moves to the value producing the highest signal. It then performs a plus/minus search on the mirror actuators until no further improvement is observed.

### Running Drift Correction

```bash
python run_fine_tune.py
```

Important parameters:

| Parameter                 | Description                                                |
| ------------------------- | ---------------------------------------------------------- |
| `global_samples`          | Number of samples collected using Latin Hypercube Sampling |
| `bo_iterations`           | Number of Bayesian Optimization iterations                 |
| `local_step`              | Initial mirror step size during local correction           |
| `local_z_step`            | Initial z-axis step size during local correction           |
| `local_rounds`            | Number of local correction rounds                          |
| `validation_measurements` | Number of measurements used to validate the final result   |

These values can be modified directly in:

```text
run_fine_tune.py
```
### Running Blind Fiber Coupling

Blind fiber coupling is intended for situations where no prior alignment information is available. Unlike drift correction, which starts from a previously aligned state, the algorithm must first locate promising coupling regions within the full actuator search space.

The procedure consists of three stages:

```text
Broad Search
     ↓
Cluster Selection
     ↓
Medium Optimization
     ↓
Fine Optimization
```

#### 1. Broad Search

A large Latin Hypercube Sampling (LHS) scan is performed over the entire servo workspace. The objective of this stage is not to find the optimum directly, but rather to identify regions that exhibit measurable coupling.

```python
RANGE_BROAD = np.asarray([2000, 2000, 2000, 2000, 2000])
BROAD_GLOBAL_SAMPLES = 3000
```

#### 2. Cluster Selection and Medium Optimization

The highest-voltage points obtained during the broad search are grouped into distinct clusters. This prevents the optimization from focusing on multiple points belonging to the same coupling region.

```python
BROAD_TOP_N_FOR_CLUSTERING = 30
N_CLUSTERS = 5
CLUSTER_DISTANCE_STEPS = 1000
```

Each cluster is then independently optimized using the same Bayesian Optimization and local correction strategy employed during drift correction, but within a larger local search region.

```python
MEDIUM_RANGE = np.asarray([500, 500, 500, 500, 500])
```

#### 3. Fine Optimization

The best-performing medium-search result is selected and refined further within a smaller search region to maximize the final coupling efficiency.

```python
FINE_RANGE = np.asarray([200, 200, 200, 200, 200])
```

The final output of this stage is the estimated optimal actuator configuration together with the corresponding measured coupling signal.


# Broad range: CENTER_POS +- RANGE_BROAD.
# With CENTER_POS=2048 and RANGE_BROAD=2000, this searches approx 48..4048.
RANGE_BROAD = np.asarray([2000, 2000, 2000, 2000, 2000], dtype=float)

BROAD_GLOBAL_SAMPLES = 3000

# How many measured points from broad scan to consider for clustering.
BROAD_TOP_N_FOR_CLUSTERING = 30

# How many distinct high-voltage clusters to keep.
N_CLUSTERS = 5

# Minimum distance between cluster representatives in full 5D servo space.
# Increase this if top clusters are still essentially the same point.
CLUSTER_DISTANCE_STEPS = 1000


# Medium search box around each cluster center.
MEDIUM_RANGE = np.asarray([500, 500, 500, 500, 500], dtype=float)

MEDIUM_OPT_CONFIG = {
    "global_samples": 250,
    "bo_iterations": 30,
    "local_step": 40,
    "local_z_step": 30,
    "local_rounds": 5,
    "validation_measurements": 5,
}


FINE_RANGE = np.asarray([200, 200, 200, 200, 200], dtype=float)

FINE_OPT_CONFIG = {
    "global_samples": 100,
    "bo_iterations": 30,
    "local_step": 20,
    "local_z_step": 10,
    "local_rounds": 6,
    "validation_measurements": 10,
}

DATASETS_FOLDER = "datasets"

### Configuration Parameters

The following parameters can be modified in `configuration.py`:

| Parameter           | Description                                |
| ------------------- | ------------------------------------------ |
| `STS_IDS`           | IDs of the servo motors used by the system |
| `PICOSCOPE_RANGE`   | Voltage range configured for the PicoScope |
| `M_LOOP_ITERATIONS` | Number of M-LOOP optimization iterations   |

Example:

```python
STS_IDS = [1, 2, 3, 4, 5]
M_LOOP_ITERATIONS = 3000
PICOSCOPE_RANGE = "PS2000_10V"
```

---

## Results

The system has been successfully used for automated fiber coupling and drift correction in a laboratory optical setup. Performance depends on the selected optimization parameters and initial alignment conditions.

---

## Author

Santiago Sangro Cid

---

## License
