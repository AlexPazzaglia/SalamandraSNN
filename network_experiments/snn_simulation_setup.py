'''
Setup functions for neuromechanical simulations

This module provides setup functions for initializing and configuring neuromechanical simulations with a neural network model. It includes functions for setting up network initialization, managing simulation parameters, and defining mechanical simulation options.

Functions:
- setup_neural_network_simulation(control_type, modname, parsname, params_process, params_runs, tag_folder, tag_process, save_data, results_path, define_topology=None, **kwargs): Set up before running multiple simulations with one network.
- _set_processes_seeds(params_processes, np_random_seed=100): Set a different seed for different processes.
- _set_processes_indices(params_processes): Set different indices for different processes.
- get_params_processes(params_processes_shared, params_processes_variable=None, n_processes_copies=1, np_random_seed=100, set_seed=True): Get simulation parameters for different processes.
- get_mech_sim_options(**kwargs): Get parameters for a mechanical simulation.
'''

import os
import logging
import importlib

import numpy as np

from queue import Queue
from importlib.machinery import SourceFileLoader

from network_experiments import (
    snn_logging,
    snn_simulation_data,
)
from network_experiments.snn_utils import (
    divide_params_in_batches
)

from network_modules.experiment.network_experiment import SnnExperiment

# ------------ [ NETWORK INITIALIZATION ] ------------
def import_snn_experiment_module(modname: str):
    ''' Import network module from file path.'''

    # Check if modname is a python file path
    if os.path.isfile(modname) and modname.endswith('.py'):
        net_module_name = os.path.basename(modname[:-3])
        net_module      = SourceFileLoader(
            os.path.basename(modname).split('.')[0],
            modname,
        ).load_module()

    # Import from default network implementations
    else:
        net_module_name = modname
        net_module = importlib.import_module('network_implementations.' + modname)

    return net_module, net_module_name


def setup_neural_network_simulation(
    control_type   : str,
    modname        : str,
    parsname       : str,
    params_process : dict,
    params_runs    : list[dict],
    tag_folder     : str,
    tag_process    : str,
    save_data      : bool,
    results_path   : str,
    define_topology: bool = None,
    **kwargs,
) -> SnnExperiment:
    '''
    Set up before running multiple simulations with one network.

    Args:
        control_type (str): The type of control for the simulation, e.g., 'open_loop', 'closed_loop', 'signal_driven',
        'position_control', 'hybrid_position_control'.
        modname (str): Name of the neural network model.
        parsname (str): Name of the network parameters.
        params_process (dict): Parameters specific to the simulation process.
        params_runs (list[dict]): List of parameters for individual simulation runs.
        tag_folder (str): Tag for the simulation folder.
        tag_process (str): Tag for the simulation process.
        save_data (bool): Whether to save simulation data.
        results_path (str): Path to the results folder.
        define_topology (bool, optional): Whether to define the network topology. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        SnnExperiment: An instance of the SnnExperiment class representing the neuromechanical simulation.
    '''

    # Get network module implementation and name
    net_module, net_module_name = import_snn_experiment_module(modname)

    # Define logging functionality
    snn_logging.define_logging(
        modname         = net_module_name,
        tag_folder      = tag_folder,
        tag_process     = tag_process,
        results_path    = results_path,
        data_file_tag   = params_process.get('simulation_data_file_tag'),
        tag_sub_process = kwargs.get('tag_sub_process'),
    )

    # Initialize neural network
    thread_communication = ( control_type in ['closed_loop', 'signal_driven', 'hybrid_position_control'] )
    define_topology    = (
        define_topology
        if define_topology is not None
        else
        control_type in ['open_loop', 'closed_loop', 'hybrid_position_control' ]
    )
    q_in  = Queue() if thread_communication else None
    q_out = Queue() if thread_communication else None

    # TODO: Avoid importlib for the network implementations
    logging.info(
        f'Imported network module {net_module_name}'
        f'from {os.path.abspath(net_module.__file__)}'
    )

    snn_sim: SnnExperiment = net_module.SalamandraSimulation(
        network_name = net_module_name,
        params_name  = parsname,
        results_path = results_path,
        control_type = control_type,
        q_in         = q_in,
        q_out        = q_out,
        new_pars     = params_process,
        tag_folder   = tag_folder,
        tag_process  = tag_process,
        **kwargs,
    )

    if define_topology:
        snn_sim.define_network_topology()

    # Save parameters
    if save_data:
        snn_simulation_data.save_parameters_process(
            snn_sim           = snn_sim,
            params_process    = params_process,
            params_runs       = params_runs,
        )

    return snn_sim

# \----------- [ NETWORK INITIALIZATION ] ------------

# ------------ [ PARAMETERS ] ------------
def _set_processes_seeds(
        params_processes: list[dict],
        np_random_seed  : int = 100,
) -> list[dict]:
    '''
    Set a different seed for the different processes.

    Args:
        params_processes (list[dict]): List of parameters for individual simulation processes.
        np_random_seed (int, optional): Seed value for random number generation. Defaults to 100.

    Returns:
        list[dict]: Updated list of parameters for individual simulation processes.
    '''

    n_processes = len(params_processes)
    randstate      = np.random.RandomState(
        np.random.MT19937(np.random.SeedSequence(np_random_seed))
    )
    seed_values = randstate.randint(0, 10000000, n_processes)

    for params_process, seed_value in zip( params_processes, seed_values ):
        params_process['set_seed']   = True
        params_process['seed_value'] = seed_value
        params_process['simulation_data_file_tag'] += f'_{np_random_seed}'

    return params_processes

def _set_processes_indices(
    params_processes: list[dict],
) -> list[dict]:
    '''
    Set different indices for the different processes.

    Args:
        params_processes (list[dict]): List of parameters for individual simulation processes.

    Returns:
        list[dict]: Updated list of parameters for individual simulation processes.
    '''

    for process, params_process in enumerate(params_processes):
        params_process['tag_process'] = process

    return params_processes

def get_params_processes(
    params_processes_shared   : dict,
    params_processes_variable: list[dict] = None,
    n_processes_copies        : int        = 1,
    np_random_seed            : int        = 100,
    set_seed                  : bool       = True,
    start_index               : int        = None,
    finish_index              : int        = None,
    n_processes_batch         : int        = None,
) -> list[dict]:
    '''
    Get simulation parameters for different processes.

    Args:
        params_processes_shared (dict): Parameters shared among all processes.
        params_processes_variable (list[dict], optional): List of dictionaries containing variable parameters for individual processes. Defaults to None.
        n_processes_copies (int, optional): Number of copies of variable parameters for each process. Defaults to 1.
        np_random_seed (int, optional): Seed value for random number generation. Defaults to 100.
        set_seed (bool, optional): Whether to set seeds for randomness. Defaults to True.

    Returns:
        list[dict]: List of parameters for individual simulation processes.
    '''

    if params_processes_variable in [None, []]:
        params_processes_variable = [{}]

    params_processes = [
        params_process | params_processes_shared
        for params_process in params_processes_variable
        for _process in range(n_processes_copies)
    ]

    if set_seed:
        params_processes = _set_processes_seeds(params_processes, np_random_seed)
    params_processes = _set_processes_indices(params_processes)

    # If specified, select process range
    start_index  = start_index if start_index is not None else 0
    finish_index = finish_index if finish_index is not None else len(params_processes)

    params_processes = params_processes[start_index:finish_index]

    # If specified, divide in batches for multiprocessing
    if n_processes_batch is not None:
        params_processes_batches = divide_params_in_batches(
            params_processes  = params_processes,
            n_processes_batch = n_processes_batch,
        )
    else:
        params_processes_batches = [params_processes]

    return params_processes, params_processes_batches

def get_mech_sim_options(**kwargs) -> dict:
    '''
    Get parameters for a mechanical simulation.

    Keyword Args:
        timestep (float, optional): Time step of the simulation. Defaults to 0.001.
        n_iterations (int, optional): Number of simulation iterations. Defaults to 10000.
        play (bool, optional): Whether to play the simulation. Defaults to True.
        fast (bool, optional): Whether to run the simulation in fast mode. Defaults to True.
        headless (bool, optional): Whether to run the simulation in headless mode. Defaults to True.
        show_progress (bool, optional): Whether to show simulation progress. Defaults to False.
        record (bool, optional): Whether to record simulation data. Defaults to False.

    Returns:
        dict: Dictionary of mechanical simulation options.

    Note:
        Default is as fast and non-verbose as possible
    '''

    mech_sim_options = {
        'timestep'                 : kwargs.pop('timestep', 0.001),
        'n_iterations'             : kwargs.pop('n_iterations', 10000),
        'play'                     : kwargs.pop('play', True),
        'fast'                     : kwargs.pop('fast', True),
        'headless'                 : kwargs.pop('headless', True),
        'show_progress'            : kwargs.pop('show_progress', False),
        'video'                    : kwargs.pop('video', False),
        'video_fps'                : kwargs.pop('video_fps', 30),
        'video_speed'              : kwargs.pop('video_speed', 1.0),
        'muscle_parameters_options': kwargs.pop('muscle_parameters_options', []),
        'drag_coefficients_options': kwargs.pop('drag_coefficients_options', []),
    }
    assert kwargs == {}, f'Could not assign all parameters: {kwargs}'
    return mech_sim_options

# \----------- [ PARAMETERS ] ------------