# Project Title
Automated Fiber Coupling
## Overview
This code connects to 5 servo motors (two per mirror and one for the z-translation) and is able to correct drift missaligments in fiber coupling or completely couple light to the fiber. 

## Hardware
- SCServo motors + URT1 board (COM connection)
- PicoScope as input signal
- Optical setup

## Installation
pip install -r requirements.txt

## Repository Structure
controller/
contains class for Picoscope, ServoMotors, and FiberCoupling loop
model/ 
contains class for data acquisition (currently implemented: LHS), and the Gaussian Process used.
Alternatively, one could run M-loop: m_loop and run_exeperiment_fiber coupling files.
view/
Displays a UI wich is able to:
    - Read/Write Servo Positions.
    - Read input signal (Picoscope) voltage.
    - Perform blind fiber coupling.
    - Correct drift.
## Usage
The drift correction will perform one full optimzation loop with the parameters input. Initally, a certain number of samples will be collected, which will be used to train the Gaussian Process (GP). After this, the GP will run a certain number of specificed iterations with an Upper Confidence Bound Bayesian Optimzation in order to find best coupling regions. Finally, iterative correction is performed until no improvement is observed. The iterative correction will intially do a sweep of the z value and move to whichever value produces the largest outcome, and then do a plus minus iterative strategy in each mirror.
In order to run a drift correction, the run_fine_tune.py file should be ran. It is important to keep in mind the following parameters:
    - "global_samples" -> number of samples taken with Latin Hypercube sampling
    "bo_iterations" -> Number of GP + Bayesian Optimization iterations
    "local_step" -> Initial step size for iterative correction after GP.
    "local_z_step" -> Initial step size for iterative correction after GP.
    "local_rounds" -> Number of rounds the iterative correction will perform per cycle.
    "validation_measurements" -> Number of measurements to take at the end to validate your final voltage.
All of these values can be updated in the file run_fine_tune.py
# Measurement settings
SETTLE_AFTER_INITIAL_MOVE_S = 1.0
INITIAL_MEASUREMENTS = 10

SERVO_MIN = 0
SERVO_MAX = 4095

## Results
Brief summary of achieved performance.

## Authors
Your name

## License