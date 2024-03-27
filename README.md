# SalamandraSNN

Closed loop simulations of the salamander's motor circuits using spiking neural networks

## Instructions

- Download and install Python 3.10

- Create a folder for your project (e.g. ProjectSNN)

- Create and activate a virtual environment within the ProjectSNN folder:

    `python -m venv snnenv` \
    `snnenv\Scripts\activate` (Windows) \
    `source snnenv\bin\activate` (Linux)

- Clone SalamandraSNN repository within the ProjectSNN folder:

   `git clone git@gitlab.com:alessandro.pazzaglia/salamandrasnn.git`

- Enter SalamandraSNN folder and install requirements:

   `pip install -r requirements.txt`

<!-- - Run test scripts:

  `TEST_sim_signal.py`	 Runs a mechanical model with an arbitrary motor output signal. \
  `TEST_sim_openloop.py`   Runs a SNN model without the mechanical model. \
  `TEST_sim_farms.py`      Runs the mechanical model with the SNN model. -->
