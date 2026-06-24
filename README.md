# Automated Fiber Coupling

## Overview

This code connects to 5 servo motors (two per mirror and one for the z-translation) and is able to correct drift missaligments in fiber coupling or completely couple light to the fiber.

---

## Hardware

* SCServo motors + URT1 board (COM connection)
* PicoScope as input signal
* Optical setup

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

Contains classes for:

* PicoScope communication
* Servo motor control
* Fiber coupling optimization

```text
model/
│
├── Data acquisition
├── Latin Hypercube Sampling (LHS)
└── Gaussian Process implementation
```

Contains classes for data acquisition (currently implemented: LHS) and the Gaussian Process used.

Alternatively, one could run M-LOOP:

```text
m_loop/
run_experiment_fiber_coupling.py
```

```text
view/
```

Displays a UI which is able to:

* Read/Write servo positions
* Read input signal (PicoScope voltage)
* Perform blind fiber coupling
* Correct drift

---

## Usage

### Drift Correction Workflow

```
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

The drift correction will perform one full optimization loop with the parameters input.

Initially, a certain number of samples will be collected, which will be used to train the Gaussian Process (GP).

After this, the GP will run a certain number of specified iterations with an Upper Confidence Bound Bayesian Optimization in order to find best coupling regions.

Finally, iterative correction is performed until no improvement is observed.

The iterative correction will initially do a sweep of the z value and move to whichever value produces the largest outcome, and then do a plus minus iterative strategy in each mirror.

### Running Drift Correction

Run:

```bash
python run_fine_tune.py
```

Important parameters:

| Parameter                 | Description                                                             |
| ------------------------- | ----------------------------------------------------------------------- |
| `global_samples`          | Number of samples taken with Latin Hypercube sampling                   |
| `bo_iterations`           | Number of GP + Bayesian Optimization iterations                         |
| `local_step`              | Initial step size for iterative correction after GP                     |
| `local_z_step`            | Initial step size for iterative correction after GP                     |
| `local_rounds`            | Number of rounds the iterative correction will perform per cycle        |
| `validation_measurements` | Number of measurements to take at the end to validate the final voltage |

All of these values can be updated in:

```text
run_fine_tune.py
```

### Measurement Settings

```python
SETTLE_AFTER_INITIAL_MOVE_S = 1.0
INITIAL_MEASUREMENTS = 10

SERVO_MIN = 0
SERVO_MAX = 4095
```

---

## Results

Brief summary of achieved performance.

---

## Authors

Your name

---

## License
