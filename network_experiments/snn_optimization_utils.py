'''
Functions to Run Optimizations with Neuro-Mechanical Models

This module contains functions to launch network simulations for optimization purposes using multiprocessing techniques.

Functions:
    - network_sub_process_runner
    - save_optimization_parameters

'''

import os
import logging

from typing import Callable
from multiprocessing import Pipe

import dill

from network_modules.experiment.network_experiment import SnnExperiment

from network_experiments import (
    snn_logging,
    snn_simulation,
)

def network_sub_process_runner(
    control_type            : str,
    snn_sim                 : SnnExperiment,
    sub_process_ind         : int,
    connection              : Pipe,
    params_runs             : list[dict],
    mech_sim_options        : dict,
    verbose_logging         : bool,
    motor_output_signal_func: Callable = None,
):
    '''
    Launches network simulations and sends results via multiprocessing pipe.

    Parameters:
    - control_type (str): Type of control for the simulation.
    - snn_sim (SnnExperiment): The SnnExperiment object containing simulation parameters.
    - sub_process_ind (int): Index of the sub-process.
    - connection (Pipe): Multiprocessing Pipe for communication.
    - params_runs (list[dict]): List of dictionaries containing simulation parameters.
    - mech_sim_options (dict): Options for mechanics simulation.
    - verbose_logging (bool): Flag for verbose logging.
    '''

    # Change logging lile (sub process)
    snn_logging.define_logging(
        modname         = snn_sim.params.simulation.netname,
        tag_folder      = snn_sim.params.simulation.tag_folder,
        tag_process     = snn_sim.params.simulation.tag_process,
        results_path    = snn_sim.params.simulation.results_path,
        data_file_tag   = snn_sim.params.simulation.data_file_tag,
        tag_sub_process = str(sub_process_ind),
        verbose         = verbose_logging,
    )

    # Update tag_sub_process field
    snn_sim.params.simulation.tag_sub_process = str(sub_process_ind)

    # Save network state in the sub_process folder
    snn_sim.save_network_state()

    # Launch simulation
    metrics_sub_process = snn_simulation._simulate_single_net_multi_run(
        control_type             = control_type,
        snn_sim                  = snn_sim,
        params_runs              = params_runs,
        save_data                = False,
        plot_figures             = False,
        delete_files             = True,
        delete_connectivity      = False,
        mech_sim_options         = mech_sim_options,
        motor_output_signal_func = motor_output_signal_func,
    )

    # Send results via the Pipe
    connection.send(metrics_sub_process)
    return

def save_optimization_parameters(
    opt_pars_folder_name: str,
    pars_optimization   : list[tuple[str, float, float]],
    obj_optimization    : list[tuple[str, str]],
    results_path        : str,
    constr_optimization : dict[str, tuple[float]] = None,
):
    '''
    Save parameters of the optimization to a file.

    Parameters:
    - opt_pars_folder_name (str): Folder name for saving optimization parameters.
    - pars_optimization (list[tuple[str, float, float]]): List of variable optimization parameters.
    - obj_optimization (list[tuple[str, str]]): List of objective optimization parameters.
    - results_path (str): Path for saving results.
    - constr_optimization (dict[str, tuple[float]]): Dictionary of constraint optimization parameters (default: None).
    '''

    # Save
    folder = f'{results_path}/data/{opt_pars_folder_name}'
    os.makedirs(folder, exist_ok=True)

    filename = f'{folder}/parameters_optimization.dill'
    logging.info('Saving parameters_optimization data to %s', filename)
    with open(filename, 'wb') as outfile:
        dill.dump(pars_optimization,   outfile)
        dill.dump(obj_optimization,    outfile)
        dill.dump(constr_optimization, outfile)
