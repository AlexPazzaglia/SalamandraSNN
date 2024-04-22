# SalamandraSNN

Closed loop simulations of the salamander's motor circuits using spiking neural networks

## Code organization

- experimental_data <br />
Scripts to analyze data from biological experiments.

- network_parameters <br />
Configuration files listing the parameters of the simulations.

- network modules <br />
Scripts to generate spiking neural network and mechanical simulations.

- network_implementation <br />
Files to define the connectivity of the spiking neural networks and
the network behavior during the simulation (e.g. the step functions to be executed).

- run_anl_files <br />
Scripts to run simulations in parallel (multi-processing) and analyze the effect of specific parameters via grid searches.

- run_sal_files <br />
Scripts to run simulations in parallel (multi-processing) and perform sensitivity analyses to understand the effect of parameters on the performance.

- run_opt_files <br />
Scripts to run simulations in parallel (multi-processing) and use optimization to minimize one or more cost functions of the model.

- run_sim_files <br />
Scripts to run single or few simulations with some desired behaviors.

## Controller types

- Position control <br />
Simulation of the mechanical body with a pre-determined joints trajectory (position control)

- Signal-driven <br />
Simulation of the mechanical body with a pre-determined input for the muscle models (torque control)

- Open-loop <br />
Simulation of the spiking neural network mdoel without controlling of a body

- Closed-loop <br />
Simulation of the spiking neural network model coupled with a simulated mechanical body

- Hybrid <br />
Simulation of the spiking neural network model coupled with a position-controlled simulated mechanical body

## Installation instructions

- Download and install Python 3.9+

- Create a folder for your project (e.g. ProjectSNN)

- Create and activate a virtual environment within the ProjectSNN folder:

    `python -m venv snnenv` \
    `snnenv\Scripts\activate` (Windows) \
    `source snnenv\bin\activate` (Linux)

- Clone SalamandraSNN repository within the ProjectSNN folder:

   `git clone git@github.com:AlexPazzaglia/SalamandraSNN.git`

- Enter SalamandraSNN folder and install requirements:

   `pip install -r requirements.txt`

- Run test scripts:

  `TEST_sim_signal.py`	 Runs a mechanical model with an arbitrary motor output signal. \
  `TEST_sim_openloop.py`   Runs a SNN model without the mechanical model. \
  `TEST_sim_farms.py`      Runs the mechanical model with the SNN model.

