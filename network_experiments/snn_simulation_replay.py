'''
Spinal Cord Model with Mechanical Simulator Integration

This module provides functions for replicating simulations of a spinal cord model
integrated with a mechanical simulator. The simulations can be recreated using saved data,
enabling the study of neural and mechanical interactions under different control scenarios.

It offers functionality for setting up the neural network, running simulations, and performing
post-processing on the simulation results. The integration allows for exploring closed-loop,
open-loop, and signal-driven control strategies.
'''

import dill
import logging

import numpy as np

from network_experiments import (
    snn_simulation_data,
    snn_simulation_setup,
    snn_simulation,
)
from network_modules.plotting.mechanics_plotting import MechPlotting

# ------------ [ SETUP ] ------------
def replicate_network_setup(
    control_type: str,
    modname     : str,
    parsname    : str,
    folder_name : str,
    tag_folder  : str,
    tag_process : str,
    results_path: str,
    run_id      : int  = None,
    run_params  : dict = None,
    load_conn   : bool = True,
    new_pars    : dict = None,
):
    '''
    Get a replica of the network and its parameters from a saved simulation

    Args:
        control_type (str): Type of control for the simulation.
        modname (str): Name of the neural network module.
        parsname (str): Name of the parameter set.
        folder_name (str): Name of the folder containing the saved data.
        tag_folder (str): Tag for the folder.
        tag_process (str): Tag for the process.
        results_path (str): Path to the results directory.
        run_id (int, optional): ID of the run. Defaults to None.
        run_params (dict, optional): Parameters for the run. Defaults to None.
        load_conn (bool, optional): Whether to load connectivity indices. Defaults to True.
        new_pars (dict, optional): New parameters to update the process parameters. Defaults to None.

    Returns:
        tuple: A tuple containing the replicated neural network simulation and mechanical simulation options.
    '''
    # Parameters
    params_process, params_runs = snn_simulation_data.load_parameters_process(
        folder_name  = folder_name,
        tag_process  = tag_process,
        results_path = results_path,
    )

    if new_pars is not None:
        params_process = params_process | new_pars

    # Run params
    if run_id is not None and run_params is None:
        run_params = params_runs[run_id]

    # Adjust parameters
    data_folder_path = f'{results_path}/{folder_name}/process_{tag_process}'
    mech_sim_options : dict = params_process.pop(
        'mech_sim_options',
        snn_simulation_setup.get_mech_sim_options(video= True)
    )
    motor_output_signal_func = params_process.pop(
        'motor_output_signal_func',
        None
    )
    params_process['connectivity_indices_file'] = f'{data_folder_path}/snn_connectivity_indices.dill'
    params_process['load_connectivity_indices'] = load_conn

    tag_process = f'{tag_process}_REPLAY'

    # Setup neural network
    snn_sim = snn_simulation_setup.setup_neural_network_simulation(
        control_type   = control_type,
        modname        = modname,
        parsname       = parsname,
        params_process = params_process,
        params_runs    = params_runs,
        tag_folder     = tag_folder,
        tag_process    = tag_process,
        save_data      = False,
        results_path   = results_path,

        pars_path      = data_folder_path,
    )

    return snn_sim, mech_sim_options, motor_output_signal_func

# \----------- [ SETUP ] ------------

# ------------ [ SIMULATION ] ------------
def replicate_network_simulation(
    control_type  : str,
    modname       : str,
    parsname      : str,
    folder_name   : str,
    tag_folder    : str,
    tag_process   : str,
    results_path  : str,
    run_id        : int = None,
    run_params    : dict = None,
    plot_figures  : bool = True,
) -> dict[str, float]:
    '''
    Replicate a simulation from saved data

    Args:
        control_type (str): Type of control for the simulation.
        modname (str): Name of the neural network module.
        parsname (str): Name of the parameter set.
        folder_name (str): Name of the folder containing the saved data.
        tag_folder (str): Tag for the folder.
        tag_process (str): Tag for the process.
        results_path (str): Path to the results directory.
        run_id (int, optional): ID of the run. Defaults to None.
        run_params (dict, optional): Parameters for the run. Defaults to None.
        plot_figures (bool, optional): Whether to plot figures. Defaults to True.

    Returns:
        dict: A dictionary containing simulation metrics.
    '''

    if run_params is None and run_id is None:
        raise ValueError('Either run_id or run_params should be provided')

    # Setup neural network
    (
        snn_sim,
        mech_sim_options,
        motor_output_signal_func
    ) = replicate_network_setup(
        control_type = control_type,
        modname      = modname,
        parsname     = parsname,
        folder_name  = folder_name,
        tag_folder   = tag_folder,
        tag_process  = tag_process,
        results_path = results_path,
        run_id       = run_id,
        run_params   = run_params,
    )

    # Simulation parameters
    if control_type == 'closed_loop':
        simulation       = snn_simulation.simulate_single_net_single_run_closed_loop
        mech_sim_options = mech_sim_options
    elif control_type == 'open_loop':
        simulation       = snn_simulation.simulate_single_net_single_run_open_loop
        mech_sim_options = {}
    elif control_type == 'signal_driven':
        simulation       = snn_simulation.simulate_single_net_single_run_signal_driven
        mech_sim_options = mech_sim_options
    else:
        raise ValueError(f'Control type {control_type} not recognized')

    # Simulation
    logging.info('Launching simulation')
    _success, metrics_run = simulation(
        snn_sim                  = snn_sim,
        run_params               = run_params,
        plot_figures             = plot_figures,
        mech_sim_options         = mech_sim_options,
        motor_output_signal_func = motor_output_signal_func,
    )

    return metrics_run

# \----------- [ SIMULATION ] ------------

# ------------ [ POST PROCESSING ] ------------
def replicate_network_post_processing(
    control_type: str,
    modname     : str,
    parsname    : str,
    folder_name : str,
    tag_folder  : str,
    tag_process : str,
    run_id      : int,
    plot_figures: bool,
    results_path: str,
) -> dict[str, np.ndarray]:
    '''
    Run simulation post processing from saved data

    Args:
        control_type (str): Type of control for the simulation.
        modname (str): Name of the neural network module.
        parsname (str): Name of the parameter set.
        folder_name (str): Name of the folder containing the saved data.
        tag_folder (str): Tag for the folder.
        tag_process (str): Tag for the process.
        run_id (int): ID of the run.
        plot_figures (bool): Whether to plot figures.
        results_path (str): Path to the results directory.

    Returns:
        dict: A dictionary containing post-processed simulation metrics.
    '''

    # Parameters
    data_folder_path = f'{results_path}/data/{folder_name}/process_{tag_process}'

    data_file = f'{data_folder_path}/snn_parameters_process.dill'
    logging.info('Loading snn_parameters_process data from %s', data_file)

    with open(data_file, "rb") as infile:
        params_process: dict       = dill.load(infile)
        params_runs   : list[dict] = dill.load(infile)

    run_params = [ params_runs[run_id] ] if len(params_runs) > 0 else {}

    # Adjust parameters
    params_process['load_connectivity_indices'] = True
    params_process = params_process | run_params

    # Setup neural network
    snn_sim = snn_simulation_setup.setup_neural_network_simulation(
        control_type   = control_type,
        modname        = modname,
        parsname       = parsname,
        params_process = params_process,
        params_runs    = params_runs,
        tag_folder     = tag_folder,
        tag_process    = tag_process,
        save_data      = False,
        results_path   = results_path,

        pars_path      = data_folder_path,
    )

    # Run post processing
    if control_type == 'closed_loop':
        mech_sim = MechPlotting( snn_sim, params_process.get('mech_sim_options'))
        mech_sim.setup_empty_simulation()
        metrics_run = snn_simulation.post_processing_single_net_single_run_closed_loop(
            snn_sim      = snn_sim,
            mech_sim     = mech_sim,
            plot_figures = plot_figures,
        )
    elif control_type == 'open_loop':
        metrics_run = snn_simulation.post_processing_single_net_single_run_open_loop(
            snn_sim      = snn_sim,
            plot_figures = plot_figures,
        )
    elif control_type == 'signal_driven':
        mech_sim = MechPlotting( snn_sim, params_process.get('mech_sim_options'))
        mech_sim.setup_empty_simulation()
        metrics_run = snn_simulation.post_processing_single_net_single_run_signal_driven(
            snn_sim      = snn_sim,
            mech_sim     = mech_sim,
            plot_figures = plot_figures,
        )
    else:
        raise ValueError(f'Control type {control_type} not recognized')

    return metrics_run

# \----------- [ POST PROCESSING ] ------------
