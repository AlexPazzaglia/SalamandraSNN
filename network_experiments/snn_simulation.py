'''
Simulation Framework for Spinal Cord Network Models and Mechanical Simulators

This module provides a comprehensive simulation framework for running and analyzing spinal cord network models together with associated mechanical simulators. The framework supports various control modes, such as open loop, closed loop, and signal-driven control, and allows for conducting simulations with different parameter combinations.

The module includes functions for running both single-process and multi-process simulations, allowing efficient exploration of the model's behavior under diverse conditions. Additionally, post-processing functions are provided to analyze and visualize simulation results.

Note:
    This module relies on external dependencies, such as `dm_control`, `matplotlib`, and custom network modules. Ensure that these dependencies are properly installed before using the framework.

Classes:
    SnnExperiment: A class representing a network experiment for spinal cord models.
    MechPlotting: A class for simulating mechanical behaviors and plotting results.

Functions:
    simulate_single_net_single_run_open_loop: Run a single network model in open loop mode.
    simulate_single_net_single_run_closed_loop: Run a single network model in closed loop mode.
    simulate_single_net_single_run_signal_driven: Run a single network model with signal-driven control.

    simulate_single_net_multi_run_open_loop_build: Build and simulate multiple network models in open loop mode.
    simulate_single_net_multi_run_closed_loop_build: Build and simulate multiple network models in closed loop mode.
    simulate_single_net_multi_run_signal_driven_build: Build and simulate multiple network models with signal-driven control.

    simulate_multi_net_multi_run_open_loop: Simulate multiple network models in open loop mode.
    simulate_multi_net_multi_run_closed_loop: Simulate multiple network models in closed loop mode.
    simulate_multi_net_multi_run_signal_driven: Simulate multiple network models with signal-driven control.

Examples:
    # Initialize a network experiment
    snn_sim = SnnExperiment()

    # Define run parameters
    run_params = {
        'param1': value1,
        'param2': value2,
        # ...
    }

    # Run a single network simulation in open loop mode
    success, metrics = simulate_single_net_single_run_open_loop(
        snn_sim,
        run_params,
        plot_figures=True,
    )

    # Simulate multiple network models with open loop control
    simulate_multi_net_multi_run_open_loop(
        modname,
        parsname,
        params_processes,
        params_runs,
        tag_folder,
        results_path,
        delete_connectivity=True,
    )

For detailed information on each function, consult the respective function's docstring.

Note:
    This module is intended for research and simulation purposes. Use and adapt the provided functions according to your specific use case.
'''

import threading
import logging
import time
import copy

import numpy as np

from typing import Union, Callable
from multiprocessing import Process
from dm_control.rl.control import PhysicsError

from network_modules.plotting.mechanics_plotting import MechPlotting
from network_modules.experiment.network_experiment import SnnExperiment
from network_modules.performance.network_performance import SNN_METRICS
from network_modules.performance.mechanics_performance import (
    MECH_METRICS,
    MECH_METRICS_VECT_JOINTS,
    MECH_METRICS_VECT_LINKS,
)

from network_experiments import (
    snn_simulation_setup,
    snn_simulation_data,
    snn_simulation_post_processing,
    snn_logging,
)

## Multi-process simulations
def _simulate_multi_net_multi_run(
    control_type       : str,
    modname            : str,
    parsname           : str,
    params_processes   : list[dict],
    params_runs        : list[dict],
    tag_folder         : str,
    results_path       : str,
    delete_connectivity: bool,
    **kwargs,
) -> None:
    '''
    Run multiple network models with multiple parameter combinations.
    '''

    # Create processes
    if control_type == 'closed_loop':
        simulation_multi_run = simulate_single_net_multi_run_closed_loop_build
    elif control_type == 'open_loop':
        simulation_multi_run = simulate_single_net_multi_run_open_loop_build
    elif control_type == 'signal_driven':
        simulation_multi_run = simulate_single_net_multi_run_signal_driven_build
    elif control_type == 'position_control':
        simulation_multi_run = simulate_single_net_multi_run_position_control_build
    elif control_type == 'hybrid_position_control':
        simulation_multi_run = simulate_single_net_multi_run_hybrid_position_control_build
    else:
        raise ValueError(f'Control type {control_type} not recognized')

    processes = [
        Process(
            target= simulation_multi_run,
            kwargs= kwargs | {
                'modname'            : modname,
                'parsname'           : parsname,
                'params_process'     : params_process,
                'params_runs'        : params_runs,
                'tag_folder'         : params_process.get('tag_folder', tag_folder),
                'tag_process'        : params_process.get('tag_process', process_ind),
                'save_data'          : True,
                'plot_figures'       : False,
                'results_path'       : results_path,
                'delete_files'       : True,
                'delete_connectivity': delete_connectivity,
            }
        )
        for process_ind, params_process in enumerate(params_processes)
    ]

    # Simulate
    n_processes = len(params_processes)

    for i in range(n_processes):
        processes[i].start()
        time.sleep(0.1)

    # Terminate
    for i in range(n_processes):
        processes[i].join()

def simulate_multi_net_multi_run_open_loop(
    modname            : str,
    parsname           : str,
    params_processes   : list[dict],
    params_runs        : list[dict],
    tag_folder         : str,
    results_path       : str,
    delete_connectivity: bool,
) -> None:
    '''
    Run multiple network models in open loop with multiple parameter combinations.

    Args:
        modname (str): Network model name.
        parsname (str): Network parameters name.
        params_processes (list[dict]): List of dictionaries containing process parameters.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        tag_folder (str): Tag for the folder.
        results_path (str): Path to save results.
        delete_connectivity (bool): Whether to delete connectivity.
    '''

    _simulate_multi_net_multi_run(
        control_type        = 'open_loop',
        modname             = modname,
        parsname            = parsname,
        params_processes    = params_processes,
        params_runs         = params_runs,
        tag_folder          = tag_folder,
        results_path        = results_path,
        delete_connectivity = delete_connectivity,
    )

def simulate_multi_net_multi_run_closed_loop(
    modname            : str,
    parsname           : str,
    params_processes   : list[dict],
    params_runs        : list[dict],
    tag_folder         : str,
    results_path       : str,
    delete_connectivity: bool,
) -> None:
    '''
    Run multiple network models in closed loop with multiple parameters combinations.

    Args:
        modname (str): Network model name.
        parsname (str): Network parameters name.
        params_processes (list[dict]): List of dictionaries containing process parameters.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        tag_folder (str): Tag for the folder.
        results_path (str): Path to save results.
        delete_connectivity (bool): Whether to delete connectivity.
    '''

    _simulate_multi_net_multi_run(
        control_type        = 'closed_loop',
        modname             = modname,
        parsname            = parsname,
        params_processes    = params_processes,
        params_runs         = params_runs,
        tag_folder          = tag_folder,
        results_path        = results_path,
        delete_connectivity = delete_connectivity,
    )

def simulate_multi_net_multi_run_signal_driven(
    modname                 : str,
    parsname                : str,
    params_processes        : list[dict],
    params_runs             : list[dict],
    tag_folder              : str,
    results_path            : str,
    delete_connectivity     : bool,
) -> None:
    '''
    Run multiple network models in open loop with the specified motor output function and
    multiple parameters combinations

    Args:
        modname (str): Network model name.
        parsname (str): Network parameters name.
        params_processes (list[dict]): List of dictionaries containing process parameters.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        tag_folder (str): Tag for the folder.
        results_path (str): Path to save results.
        delete_connectivity (bool): Whether to delete connectivity.
    '''

    _simulate_multi_net_multi_run(
        control_type             = 'signal_driven',
        modname                  = modname,
        parsname                 = parsname,
        params_processes         = params_processes,
        params_runs              = params_runs,
        tag_folder               = tag_folder,
        results_path             = results_path,
        delete_connectivity      = delete_connectivity,
    )

def simulate_multi_net_multi_run_position_control(
    modname                 : str,
    parsname                : str,
    params_processes        : list[dict],
    params_runs             : list[dict],
    tag_folder              : str,
    results_path            : str,
    delete_connectivity     : bool,
) -> None:
    '''
    Run multiple network models in open loop with the specified position control function and
    multiple parameters combinations

    Args:
        modname (str): Network model name.
        parsname (str): Network parameters name.
        params_processes (list[dict]): List of dictionaries containing process parameters.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        tag_folder (str): Tag for the folder.
        results_path (str): Path to save results.
        delete_connectivity (bool): Whether to delete connectivity.
    '''

    _simulate_multi_net_multi_run(
        control_type             = 'position_control',
        modname                  = modname,
        parsname                 = parsname,
        params_processes         = params_processes,
        params_runs              = params_runs,
        tag_folder               = tag_folder,
        results_path             = results_path,
        delete_connectivity      = delete_connectivity,
    )

def simulate_multi_net_multi_run_hybrid_position_control(
    modname            : str,
    parsname           : str,
    params_processes   : list[dict],
    params_runs        : list[dict],
    tag_folder         : str,
    results_path       : str,
    delete_connectivity: bool,
) -> None:
    '''
    Run multiple network models in hybrid position control with multiple parameters combinations.

    Args:
        modname (str): Network model name.
        parsname (str): Network parameters name.
        params_processes (list[dict]): List of dictionaries containing process parameters.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        tag_folder (str): Tag for the folder.
        results_path (str): Path to save results.
        delete_connectivity (bool): Whether to delete connectivity.
    '''

    _simulate_multi_net_multi_run(
        control_type        = 'hybrid_position_control',
        modname             = modname,
        parsname            = parsname,
        params_processes    = params_processes,
        params_runs         = params_runs,
        tag_folder          = tag_folder,
        results_path        = results_path,
        delete_connectivity = delete_connectivity,
    )

## Single-process simulations

# Built network
def _simulate_single_net_multi_run(
    control_type       : str,
    snn_sim            : SnnExperiment,
    params_runs        : list[dict],
    save_data          : bool,
    plot_figures       : bool,
    delete_files       : bool,
    delete_connectivity: bool,
    **kwargs,
) -> dict[str, np.ndarray[float]]:
    '''
    Run a network model with multiple parameter combinations.
    '''

    # Parameters
    if control_type == 'closed_loop':
        simulation    = simulate_single_net_single_run_closed_loop
        metrics_names = SNN_METRICS + MECH_METRICS
    elif control_type == 'open_loop':
        simulation    = simulate_single_net_single_run_open_loop
        metrics_names = SNN_METRICS
    elif control_type == 'signal_driven':
        simulation    = simulate_single_net_single_run_signal_driven
        metrics_names = MECH_METRICS
    elif control_type == 'position_control':
        simulation    = simulate_single_net_single_run_position_control
        metrics_names = MECH_METRICS
    elif control_type == 'hybrid_position_control':
        simulation    = simulate_single_net_single_run_hybrid_position_control
        metrics_names = SNN_METRICS + MECH_METRICS
    else:
        raise ValueError(f'Unknown control type {control_type}')

    # Run multiple simulations
    metrics_runs_all : dict[str, list[Union[float, np.ndarray[float]]] ] = {
        metric_name : []
        for metric_name in metrics_names
    }

    for run_index, run_params in enumerate(params_runs):

        # Simulation
        logging.info('Launching simulation %i / %i', run_index+1, len(params_runs))
        success, metrics_run = simulation(
            snn_sim          = snn_sim,
            run_params       = run_params,
            plot_figures     = plot_figures,
            **kwargs,
        )

        for metric_name in metrics_run.keys():
            metrics_runs_all[metric_name].append( metrics_run[metric_name] )

        if not success:
            snn_sim._redefine_network_objects()

    # Convert to arrays
    metrics_runs_all : dict[str, np.ndarray[float]] = metrics_runs_all
    for metric_name in metrics_runs_all.keys():
        metrics_runs_all[metric_name] = np.array(metrics_runs_all[metric_name])

    # Save results
    if save_data:
        snn_simulation_data.save_data_process(
            snn_sim      = snn_sim,
            metrics_runs = metrics_runs_all,
        )

    if delete_files:
        snn_simulation_data.delete_state_files(
            snn_sim,
            delete_connectivity,
        )

    return metrics_runs_all

def simulate_single_net_multi_run_open_loop(
    snn_sim            : SnnExperiment,
    params_runs        : list[dict],
    save_data          : bool,
    plot_figures       : bool,
    delete_files       : bool,
    delete_connectivity: bool,
    **kwargs,
    ) -> dict[str, np.ndarray[float]]:
    '''
    Run a network model in open loop with multiple parameter combinations.

    Args:
        snn_sim (SnnExperiment): Instance of the SnnExperiment class.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        save_data (bool): Whether to save data.
        plot_figures (bool): Whether to plot figures.
        delete_files (bool): Whether to delete files.
        delete_connectivity (bool): Whether to delete connectivity.
        **kwargs: Additional keyword arguments.

    Returns:
        dict[str, np.ndarray[float]]: Dictionary containing computed metrics for each run.
    '''
    return _simulate_single_net_multi_run(
        control_type        = 'open_loop',
        snn_sim             = snn_sim,
        params_runs         = params_runs,
        save_data           = save_data,
        plot_figures        = plot_figures,
        delete_files        = delete_files,
        delete_connectivity = delete_connectivity,
        **kwargs,
    )

def simulate_single_net_multi_run_closed_loop(
    snn_sim            : SnnExperiment,
    params_runs        : list[dict],
    save_data          : bool,
    plot_figures       : bool,
    delete_files       : bool,
    delete_connectivity: bool,
    **kwargs,
    ) -> dict[str, np.ndarray[float]]:
    '''
    Run a network model in closed loop with multiple parameter combinations.

    Args:
        snn_sim (SnnExperiment): Instance of the SnnExperiment class.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        save_data (bool): Whether to save data.
        plot_figures (bool): Whether to plot figures.
        delete_files (bool): Whether to delete files.
        delete_connectivity (bool): Whether to delete connectivity.
        **kwargs: Additional keyword arguments.

    Returns:
        dict[str, np.ndarray[float]]: Dictionary containing computed metrics for each run.
    '''
    mech_sim_options = kwargs.pop(
        'mech_sim_options',
        snn_simulation_setup.get_mech_sim_options()
    )

    return _simulate_single_net_multi_run(
        control_type        = 'closed_loop',
        snn_sim             = snn_sim,
        params_runs         = params_runs,
        save_data           = save_data,
        plot_figures        = plot_figures,
        mech_sim_options    = mech_sim_options,
        delete_files        = delete_files,
        delete_connectivity = delete_connectivity,
        **kwargs,
    )

def simulate_single_net_multi_run_signal_driven(
    snn_sim                 : SnnExperiment,
    params_runs             : list[dict],
    save_data               : bool,
    plot_figures            : bool,
    delete_files            : bool,
    delete_connectivity     : bool,
    motor_output_signal_func: Callable,
    **kwargs,
    ) -> dict[str, np.ndarray[float]]:
    '''
    Run a network model in open loop with the specified motor output function and multiple parameter combinations.

    Args:
        snn_sim (SnnExperiment): Instance of the SnnExperiment class.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        save_data (bool): Whether to save data.
        plot_figures (bool): Whether to plot figures.
        delete_files (bool): Whether to delete files.
        delete_connectivity (bool): Whether to delete connectivity.
        motor_output_signal_func (Callable): Motor output signal function.
        **kwargs: Additional keyword arguments.

    Returns:
        dict[str, np.ndarray[float]]: Dictionary containing computed metrics for each run.
    '''
    mech_sim_options = kwargs.pop(
        'mech_sim_options',
        snn_simulation_setup.get_mech_sim_options()
    )

    return _simulate_single_net_multi_run(
        control_type             = 'signal_driven',
        snn_sim                  = snn_sim,
        params_runs              = params_runs,
        save_data                = save_data,
        plot_figures             = plot_figures,
        mech_sim_options         = mech_sim_options,
        delete_files             = delete_files,
        delete_connectivity      = delete_connectivity,
        motor_output_signal_func = motor_output_signal_func,
        **kwargs,
    )

def simulate_single_net_multi_run_position_control(
    snn_sim                 : SnnExperiment,
    params_runs             : list[dict],
    save_data               : bool,
    plot_figures            : bool,
    delete_files            : bool,
    delete_connectivity     : bool,
    **kwargs,
    ) -> dict[str, np.ndarray[float]]:
    '''
    Run a network model in open loop with the specified position control function and
    multiple parameter combinations.

    Args:
        snn_sim (SnnExperiment): Instance of the SnnExperiment class.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        save_data (bool): Whether to save data.
        plot_figures (bool): Whether to plot figures.
        delete_files (bool): Whether to delete files.
        delete_connectivity (bool): Whether to delete connectivity.
        **kwargs: Additional keyword arguments.

    Returns:
        dict[str, np.ndarray[float]]: Dictionary containing computed metrics for each run.
    '''
    mech_sim_options = kwargs.pop(
        'mech_sim_options',
        snn_simulation_setup.get_mech_sim_options()
    )

    return _simulate_single_net_multi_run(
        control_type             = 'position_control',
        snn_sim                  = snn_sim,
        params_runs              = params_runs,
        save_data                = save_data,
        plot_figures             = plot_figures,
        mech_sim_options         = mech_sim_options,
        delete_files             = delete_files,
        delete_connectivity      = delete_connectivity,
        **kwargs,
    )

def simulate_single_net_multi_run_hybrid_position_control(
    snn_sim            : SnnExperiment,
    params_runs        : list[dict],
    save_data          : bool,
    plot_figures       : bool,
    delete_files       : bool,
    delete_connectivity: bool,
    **kwargs,
    ) -> dict[str, np.ndarray[float]]:
    '''
    Run a network model in hybrid position control with multiple parameter combinations.

    Args:
        snn_sim (SnnExperiment): Instance of the SnnExperiment class.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        save_data (bool): Whether to save data.
        plot_figures (bool): Whether to plot figures.
        delete_files (bool): Whether to delete files.
        delete_connectivity (bool): Whether to delete connectivity.
        **kwargs: Additional keyword arguments.

    Returns:
        dict[str, np.ndarray[float]]: Dictionary containing computed metrics for each run.
    '''
    mech_sim_options = kwargs.pop(
        'mech_sim_options',
        snn_simulation_setup.get_mech_sim_options()
    )

    return _simulate_single_net_multi_run(
        control_type        = 'hybrid_position_control',
        snn_sim             = snn_sim,
        params_runs         = params_runs,
        save_data           = save_data,
        plot_figures        = plot_figures,
        mech_sim_options    = mech_sim_options,
        delete_files        = delete_files,
        delete_connectivity = delete_connectivity,
        **kwargs,
    )

# Unbuilt network
def _simulate_single_net_multi_run_build(
    control_type       : str,
    modname            : str,
    parsname           : str,
    params_process     : dict,
    params_runs        : list[dict],
    tag_folder         : str,
    tag_process        : str,
    save_data          : bool,
    plot_figures       : bool,
    results_path       : str,
    delete_files       : bool,
    delete_connectivity: bool,
    **kwargs,
) -> dict[str, np.ndarray[float]]:
    '''
    Build and run a network model with multiple parameter combinations.
    '''

    mech_sim_options         = params_process.pop('mech_sim_options', {})
    motor_output_signal_func = params_process.pop('motor_output_signal_func', None)
    save_prompt              = kwargs.pop('save_prompt', True)

    # Setup neural network
    snn_sim = snn_simulation_setup.setup_neural_network_simulation(
        control_type        = control_type,
        modname             = modname,
        parsname            = parsname,
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = tag_folder,
        tag_process         = tag_process,
        save_data           = save_data,
        results_path        = results_path,
        **kwargs,
    )

    metrics_runs_all =_simulate_single_net_multi_run(
        control_type             = control_type,
        snn_sim                  = snn_sim,
        params_runs              = params_runs,
        save_data                = save_data,
        plot_figures             = plot_figures,
        mech_sim_options         = mech_sim_options,
        delete_files             = delete_files,
        delete_connectivity      = delete_connectivity,
        motor_output_signal_func = motor_output_signal_func,
        save_prompt              = save_prompt,
        **kwargs
    )

    return metrics_runs_all

def simulate_single_net_multi_run_open_loop_build(
    modname            : str,
    parsname           : str,
    params_process     : dict,
    params_runs        : list[dict],
    tag_folder         : str,
    tag_process        : str,
    save_data          : bool,
    plot_figures       : bool,
    results_path       : str,
    delete_files       : bool,
    delete_connectivity: bool,
    **kwargs,
    ) -> dict[str, np.ndarray[float]]:
    '''
    Build and run a network model in open loop with multiple parameter combinations.

    Args:
        modname (str): Name of the network model.
        parsname (str): Name of the network parameters.
        params_process (dict): Dictionary of process parameters.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        tag_folder (str): Folder tag.
        tag_process (str): Process tag.
        save_data (bool): Whether to save data.
        plot_figures (bool): Whether to plot figures.
        results_path (str): Path to save results.
        delete_files (bool): Whether to delete files.
        delete_connectivity (bool): Whether to delete connectivity.
        **kwargs: Additional keyword arguments.

    Returns:
        dict[str, np.ndarray[float]]: Dictionary containing computed metrics for each run in
        open loop using the built network.
    '''

    return _simulate_single_net_multi_run_build(
        control_type        = 'open_loop',
        modname             = modname,
        parsname            = parsname,
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = tag_folder,
        tag_process         = tag_process,
        save_data           = save_data,
        plot_figures        = plot_figures,
        results_path        = results_path,
        delete_files        = delete_files,
        delete_connectivity = delete_connectivity,
        **kwargs
    )

def simulate_single_net_multi_run_closed_loop_build(
    modname            : str,
    parsname           : str,
    params_process     : dict,
    params_runs        : list[dict],
    tag_folder         : str,
    tag_process        : str,
    save_data          : bool,
    plot_figures       : bool,
    results_path       : str,
    delete_files       : bool,
    delete_connectivity: bool,
    **kwargs,
    ) -> dict[str, np.ndarray[float]]:
    '''
    Build and run a network model in closed loop with multiple parameter combinations.

    Args:
        modname (str): Name of the network model.
        parsname (str): Name of the network parameters.
        params_process (dict): Dictionary of process parameters.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        tag_folder (str): Folder tag.
        tag_process (str): Process tag.
        save_data (bool): Whether to save data.
        plot_figures (bool): Whether to plot figures.
        results_path (str): Path to save results.
        delete_files (bool): Whether to delete files.
        delete_connectivity (bool): Whether to delete connectivity.
        **kwargs: Additional keyword arguments.

    Returns:
        dict[str, np.ndarray[float]]: Dictionary containing computed metrics for each run
        in open loop using the built network.
    '''
    return _simulate_single_net_multi_run_build(
        control_type        = 'closed_loop',
        modname             = modname,
        parsname            = parsname,
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = tag_folder,
        tag_process         = tag_process,
        save_data           = save_data,
        plot_figures        = plot_figures,
        results_path        = results_path,
        delete_files        = delete_files,
        delete_connectivity = delete_connectivity,
        **kwargs,
    )

def simulate_single_net_multi_run_signal_driven_build(
    modname            : str,
    parsname           : str,
    params_process     : dict,
    params_runs        : list[dict],
    tag_folder         : str,
    tag_process        : str,
    save_data          : bool,
    plot_figures       : bool,
    results_path       : str,
    delete_files       : bool,
    delete_connectivity: bool,
    **kwargs,
    ) -> dict[str, np.ndarray[float]]:
    '''
    Run a network model with a user-defined motor output function and
    multiple parameters combinations.

    Args:
        modname (str): Name of the network model.
        parsname (str): Name of the network parameters.
        params_process (dict): Dictionary of process parameters.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        tag_folder (str): Folder tag.
        tag_process (str): Process tag.
        save_data (bool): Whether to save data.
        plot_figures (bool): Whether to plot figures.
        results_path (str): Path to save results.
        delete_files (bool): Whether to delete files.
        delete_connectivity (bool): Whether to delete connectivity.
        **kwargs: Additional keyword arguments.

    Returns:
        dict[str, np.ndarray[float]]: Dictionary containing computed metrics for each run
        in open loop using the built network.

    '''
    return _simulate_single_net_multi_run_build(
        control_type        = 'signal_driven',
        modname             = modname,
        parsname            = parsname,
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = tag_folder,
        tag_process         = tag_process,
        save_data           = save_data,
        plot_figures        = plot_figures,
        results_path        = results_path,
        delete_files        = delete_files,
        delete_connectivity = delete_connectivity,
        **kwargs,
    )

def simulate_single_net_multi_run_position_control_build(
    modname            : str,
    parsname           : str,
    params_process     : dict,
    params_runs        : list[dict],
    tag_folder         : str,
    tag_process        : str,
    save_data          : bool,
    plot_figures       : bool,
    results_path       : str,
    delete_files       : bool,
    delete_connectivity: bool,
    **kwargs,
    ) -> dict[str, np.ndarray[float]]:
    '''
    Run a network model with a user-defined position control function and
    multiple parameters combinations.

    Args:
        modname (str): Name of the network model.
        parsname (str): Name of the network parameters.
        params_process (dict): Dictionary of process parameters.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        tag_folder (str): Folder tag.
        tag_process (str): Process tag.
        save_data (bool): Whether to save data.
        plot_figures (bool): Whether to plot figures.
        results_path (str): Path to save results.
        delete_files (bool): Whether to delete files.
        delete_connectivity (bool): Whether to delete connectivity.
        **kwargs: Additional keyword arguments.

    Returns:
        dict[str, np.ndarray[float]]: Dictionary containing computed metrics for each run
        in open loop using the built network.

    '''
    return _simulate_single_net_multi_run_build(
        control_type        = 'position_control',
        modname             = modname,
        parsname            = parsname,
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = tag_folder,
        tag_process         = tag_process,
        save_data           = save_data,
        plot_figures        = plot_figures,
        results_path        = results_path,
        delete_files        = delete_files,
        delete_connectivity = delete_connectivity,
        **kwargs,
    )

def simulate_single_net_multi_run_hybrid_position_control_build(
    modname            : str,
    parsname           : str,
    params_process     : dict,
    params_runs        : list[dict],
    tag_folder         : str,
    tag_process        : str,
    save_data          : bool,
    plot_figures       : bool,
    results_path       : str,
    delete_files       : bool,
    delete_connectivity: bool,
    **kwargs,
    ) -> dict[str, np.ndarray[float]]:
    '''
    Build and run a network model in hybdrid control with multiple parameter combinations.

    Args:
        modname (str): Name of the network model.
        parsname (str): Name of the network parameters.
        params_process (dict): Dictionary of process parameters.
        params_runs (list[dict]): List of dictionaries containing run parameters.
        tag_folder (str): Folder tag.
        tag_process (str): Process tag.
        save_data (bool): Whether to save data.
        plot_figures (bool): Whether to plot figures.
        results_path (str): Path to save results.
        delete_files (bool): Whether to delete files.
        delete_connectivity (bool): Whether to delete connectivity.
        **kwargs: Additional keyword arguments.

    Returns:
        dict[str, np.ndarray[float]]: Dictionary containing computed metrics for each run
        in open loop using the built network.
    '''
    return _simulate_single_net_multi_run_build(
        control_type        = 'hybrid_position_control',
        modname             = modname,
        parsname            = parsname,
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = tag_folder,
        tag_process         = tag_process,
        save_data           = save_data,
        plot_figures        = plot_figures,
        results_path        = results_path,
        delete_files        = delete_files,
        delete_connectivity = delete_connectivity,
        **kwargs,
    )

## Single simulation
def simulate_single_net_single_run(
    control_type       : str,
    snn_sim            : SnnExperiment,
    params_run         : dict,
    plot_figures       : bool,
    save_prompt        : bool = True,
    **kwargs,
) -> tuple[ bool, dict[str, Union[float, np.ndarray[float]]] ]:
    '''
    Run a network model with multiple parameter combinations.
    '''

    # Get simulation function
    if control_type == 'closed_loop':
        simulation    = simulate_single_net_single_run_closed_loop
    elif control_type == 'open_loop':
        simulation    = simulate_single_net_single_run_open_loop
    elif control_type == 'signal_driven':
        simulation    = simulate_single_net_single_run_signal_driven
    elif control_type == 'position_control':
        simulation    = simulate_single_net_single_run_position_control
    elif control_type == 'hybrid_position_control':
        simulation    = simulate_single_net_single_run_hybrid_position_control
    else:
        raise ValueError(f'Unknown control type {control_type}')

    # Simulation
    logging.info('Launching simulation')
    return simulation(
        snn_sim          = snn_sim,
        run_params       = params_run,
        plot_figures     = plot_figures,
        save_prompt      = save_prompt,
        **kwargs,
    )

def simulate_single_net_single_run_open_loop(
    snn_sim     : SnnExperiment,
    run_params  : dict,
    plot_figures: bool,
    save_prompt : bool = True,
    **kwargs,
    ) -> tuple[ bool, dict[str, Union[float, np.ndarray[float]]] ]:
    '''
    Run a network model in open loop.

    Args:
        snn_sim (SnnExperiment): The SnnExperiment instance representing the network model.
        run_params (dict): Dictionary of run parameters.
        plot_figures (bool): Whether to plot figures.
        save_prompt (bool, optional): Whether to save prompts. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[bool, dict[str, Union[float, np.ndarray[float]]] ]: A tuple containing a boolean indicating the success of the simulation and a dictionary of computed metrics.
    '''

    # Initialize network state
    snn_sim.load_network_state()

    # Update model parameters
    parameters_scalings = run_params.pop('scalings', None)
    snn_sim.update_network_parameters_from_dict(run_params)

    # Run simulation
    snn_sim.simulation_run(parameters_scalings)
    success = True

    # Post-processing
    metrics = snn_simulation_post_processing.post_processing_single_net_single_run_open_loop(
        snn_sim      = snn_sim,
        plot_figures = plot_figures,
        save_prompt  = save_prompt,
    )

    return success, metrics

def simulate_single_net_single_run_closed_loop(
    snn_sim         : SnnExperiment,
    run_params      : dict,
    plot_figures    : bool,
    mech_sim_options: dict,
    save_prompt     : bool = True,
    **kwargs
) -> tuple[ bool, dict[str, Union[float, np.ndarray[float]]] ]:
    '''
    Run a network model in closed loop.

    Args:
        snn_sim (SnnExperiment): The SnnExperiment instance representing the network model.
        run_params (dict): Dictionary of run parameters.
        plot_figures (bool): Whether to plot figures.
        save_prompt (bool, optional): Whether to save prompts. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[bool, dict[str, Union[float, np.ndarray[float]]] ]: A tuple containing a boolean indicating the success of the simulation and a dictionary of computed metrics.

    '''

    # Initialize network state
    snn_sim.load_network_state()

    # Update snn_sim parameters
    run_params_snn_sim = {
        key : value
        for key, value in run_params.items()
        if key not in [
            'mech_sim_options',
            'motor_output_signal_pars'
        ]
    }
    parameters_scalings = run_params_snn_sim.pop('scalings', None)
    snn_sim.update_network_parameters_from_dict(run_params_snn_sim)

    # Mechanical simulation thread
    mech_sim = MechPlotting(
        snn_network           = snn_sim,
        mech_sim_options_dict = copy.deepcopy(mech_sim_options),
        run_params            = run_params,
    )

    mech_sim_thread = threading.Thread(
        target=mech_sim.simulation_run,
        args=(),
    )

    try:
        # Run simulation
        mech_sim_thread.start()
        snn_sim.simulation_run(parameters_scalings)

        # Wait for thread termination
        mech_sim_thread.join()

        # Post processing
        metrics = snn_simulation_post_processing.post_processing_single_net_single_run_closed_loop(
            snn_sim      = snn_sim,
            mech_sim     = mech_sim,
            plot_figures = plot_figures,
            save_prompt  = save_prompt,
        )
        success = True

    except PhysicsError:
        metrics = _get_error_metrics(snn_sim)
        success = False

    return success, metrics

def simulate_single_net_single_run_signal_driven(
    snn_sim                 : SnnExperiment,
    run_params              : dict,
    plot_figures            : bool,
    mech_sim_options        : dict,
    motor_output_signal_func: Callable,
    save_prompt             : bool = True,
    **kwargs
) -> tuple[ bool, dict[str, Union[float, np.ndarray[float]]] ]:
    '''
    Run a network model with a user-defined motor output function.

    Args:
        snn_sim (SnnExperiment): The SnnExperiment instance representing the network model.
        run_params (dict): Dictionary of run parameters.
        plot_figures (bool): Whether to plot figures.
        save_prompt (bool, optional): Whether to save prompts. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[bool, dict[str, Union[float, np.ndarray[float]]] ]: A tuple containing a boolean indicating the success of the simulation and a dictionary of computed metrics.
    '''

    # Update snn_sim parameters
    run_params_snn_sim = {
        key : value
        for key, value in run_params.items()
        if key not in [
            'mech_sim_options',
            'motor_output_signal_pars'
        ]
    }
    snn_sim.update_network_parameters_from_dict(run_params_snn_sim)

    # Compute motor output signal
    logging.info(
        'Updating motor output signal parameters to: \n {}'.format(
            snn_logging.pretty_string(run_params["motor_output_signal_pars"])
        )
    )

    motor_output_signal : np.ndarray = motor_output_signal_func(
        motor_output_params = run_params['motor_output_signal_pars'],
        snn_network         = snn_sim,
    )

    # Mechanical simulation thread
    mech_sim = MechPlotting(
        snn_network             = snn_sim,
        mech_sim_options_dict   = copy.deepcopy(mech_sim_options),
        run_params              = run_params,
        motor_output_signal     = motor_output_signal,
    )

    try:
        # Run simulation
        mech_sim.simulation_run(main_thread = True)

        # Post processing
        metrics = snn_simulation_post_processing.post_processing_single_net_single_run_signal_driven(
            snn_sim      = snn_sim,
            mech_sim     = mech_sim,
            plot_figures = plot_figures,
            save_prompt  = save_prompt,
        )
        success = True

    except PhysicsError:
        metrics = _get_error_metrics(snn_sim)
        success = False

    return success, metrics

def simulate_single_net_single_run_position_control(
    snn_sim                 : SnnExperiment,
    run_params              : dict,
    plot_figures            : bool,
    mech_sim_options        : dict,
    save_prompt             : bool = True,
    **kwargs
) -> tuple[ bool, dict[str, Union[float, np.ndarray[float]]] ]:
    '''
    Run a network model with a user-defined position control function.

    Args:
        snn_sim (SnnExperiment): The SnnExperiment instance representing the network model.
        run_params (dict): Dictionary of run parameters.
        plot_figures (bool): Whether to plot figures.
        mech_sim_options (dict): Dictionary of mechanical simulation options.
        save_prompt (bool, optional): Whether to save prompts. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[bool, dict[str, Union[float, np.ndarray[float]]] ]: A tuple containing a
        boolean indicating the success of the simulation and a dictionary of computed metrics.
    '''

    # Update snn_sim parameters
    run_params_snn_sim = {
        key : value
        for key, value in run_params.items()
        if key not in [
            'mech_sim_options',
        ]
    }
    snn_sim.update_network_parameters_from_dict(run_params_snn_sim)

    # Mechanical simulation thread
    mech_sim = MechPlotting(
        snn_network             = snn_sim,
        mech_sim_options_dict   = copy.deepcopy(mech_sim_options),
        run_params              = run_params,
    )

    try:
        # Run simulation
        mech_sim.simulation_run(main_thread = True)

        # Post processing
        metrics = snn_simulation_post_processing.post_processing_single_net_single_run_position_control(
            snn_sim      = snn_sim,
            mech_sim     = mech_sim,
            plot_figures = plot_figures,
            save_prompt  = save_prompt,
        )
        success = True

    except PhysicsError:
        metrics = _get_error_metrics(snn_sim)
        success = False

    return success, metrics

def simulate_single_net_single_run_hybrid_position_control(
    snn_sim         : SnnExperiment,
    run_params      : dict,
    plot_figures    : bool,
    mech_sim_options: dict,
    save_prompt     : bool = True,
    **kwargs
) -> tuple[ bool, dict[str, Union[float, np.ndarray[float]]] ]:
    '''
    Run a network model in hybrid position control.

    Args:
        snn_sim (SnnExperiment): The SnnExperiment instance representing the network model.
        run_params (dict): Dictionary of run parameters.
        plot_figures (bool): Whether to plot figures.
        save_prompt (bool, optional): Whether to save prompts. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[bool, dict[str, Union[float, np.ndarray[float]]] ]: A tuple containing a boolean
        indicating the success of the simulation and a dictionary of computed metrics.

    '''

    # Initialize network state
    snn_sim.load_network_state()

    # Update snn_sim parameters
    run_params_snn_sim = {
        key : value
        for key, value in run_params.items()
        if key not in [
            'mech_sim_options',
            'motor_output_signal_pars'
        ]
    }
    parameters_scalings = run_params_snn_sim.pop('scalings', None)
    snn_sim.update_network_parameters_from_dict(run_params_snn_sim)

    # Mechanical simulation thread
    mech_sim = MechPlotting(
        snn_network           = snn_sim,
        mech_sim_options_dict = copy.deepcopy(mech_sim_options),
        run_params            = run_params,
    )

    mech_sim_thread = threading.Thread(
        target=mech_sim.simulation_run,
        args=(),
    )

    try:
        # Run simulation
        mech_sim_thread.start()
        snn_sim.simulation_run(parameters_scalings)

        # Wait for thread termination
        mech_sim_thread.join()

        # Post processing
        metrics = snn_simulation_post_processing.post_processing_single_net_single_run_hybrid_position_control(
            snn_sim      = snn_sim,
            mech_sim     = mech_sim,
            plot_figures = plot_figures,
            save_prompt  = save_prompt,
        )
        success = True

    except PhysicsError:
        metrics = _get_error_metrics(snn_sim)
        success = False

    return success, metrics

# Error handling
def _get_error_metrics(snn_sim: SnnExperiment):
    ''' Get NAN metrics in case of error '''

    if snn_sim.control_type == 'open_loop':
        all_metrics = SNN_METRICS
    if snn_sim.control_type == 'closed_loop':
        all_metrics = SNN_METRICS + MECH_METRICS
    if snn_sim.control_type == 'signal_driven':
        all_metrics = MECH_METRICS
    if snn_sim.control_type == 'position_control':
        all_metrics = MECH_METRICS
    if snn_sim.control_type == 'hybrid_position_control':
        all_metrics = SNN_METRICS + MECH_METRICS

    # In case of error, return NAN metrics
    n_joints_ax = snn_sim.params.mechanics.mech_axial_joints
    n_links_ax  = n_joints_ax + 2

    metrics = {
        key : np.nan
        if key not in MECH_METRICS_VECT_JOINTS + MECH_METRICS_VECT_LINKS
        else
        (
            list( np.nan * np.ones(n_joints_ax) )
            if key in MECH_METRICS_VECT_JOINTS
            else
            list( np.nan * np.ones(n_links_ax) )
        )
        for key in all_metrics
    }

    logging.info(f'METRICS: {snn_logging.pretty_string(metrics)}')

    return metrics
