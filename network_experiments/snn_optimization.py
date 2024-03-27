'''
Optimization Functions for Neuro-Mechanical Models

This module provides functions to run optimizations with neuro-mechanical models using multi-processing techniques.

Functions:
    - _run_optimization_single_process
    - _run_optimization
    - run_optimization_open_loop
    - run_optimization_closed_loop
    - run_optimization_signal_driven

'''

import time
import logging

from multiprocessing import Process
from typing import Callable

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from network_experiments import (
    snn_simulation_setup,
    snn_optimization_utils,
    snn_optimization_problem,
)

# ------------ [ SINGLE PROCESS ] ------------

def _run_optimization_single_process(
    control_type       : str,
    modname            : str,
    parsname           : str,
    params_process     : dict,
    tag_folder         : str,
    tag_process        : str,
    pars_optimization  : list[list[str, float, float]],
    obj_optimization   : list[str, str],
    results_path       : str,
    constr_optimization: dict[str, tuple[float]] = None,
    n_sub_processes    : int = 1,
    pop_inputs_process : np.ndarray = None,
    problem_class      : object = snn_optimization_problem.OptimizationPropblem,
    **kwargs,
):
    '''
    Process to run a single process of the optimization pipeline.

    Parameters:
    - control_type (str): Type of control for the simulation.
    - modname (str): Name of the module.
    - parsname (str): Name of the parameters.
    - params_process (dict): Process parameters.
    - tag_folder (str): Folder tag for naming.
    - tag_process (str): Process tag for naming.
    - pars_optimization (list[list[str, float, float]]): List of variable optimization parameters.
    - obj_optimization (list[list[str, str]]): List of objective optimization parameters.
    - results_path (str): Path for saving results.
    - constr_optimization (dict[str, tuple[float]]): Dictionary of constraint optimization parameters (default: None).
    - n_sub_processes (int): Number of sub-processes (default: 1).
    - pop_inputs_process (np.ndarray): Population inputs for optimization (default: None).
    - problem_class  (snn_optimization_problem.OptimizationPropblem, optional): Optimization problem class.

    - gen_index_start (int): Starting generation index (default: 0).
    - pop_size (int): Population size for optimization (default: 50).
    - n_gen (int): Number of generations for optimization (default: 300).
    - verbose (bool): Flag for verbose logging (default: True).
    - save_history (bool): Flag for saving optimization history (default: False).
    '''

    # Parameters of the optimization
    pop_size        = kwargs.pop('pop_size', snn_optimization_problem.DEFAULT_PARAMS['pop_size'])
    n_gen           = kwargs.pop(   'n_gen', snn_optimization_problem.DEFAULT_PARAMS['n_gen'])
    gen_index_start = kwargs.pop('gen_index_start', 0)
    save_history    = kwargs.pop('save_history', False)
    verbose         = kwargs.pop('verbose', False)

    mech_sim_options = params_process.pop(
        'mech_sim_options',
        snn_simulation_setup.get_mech_sim_options()
    )

    motor_output_signal_func = params_process.pop(
        'motor_output_signal_func',
        None
    )

    # Create network
    snn_sim = snn_simulation_setup.setup_neural_network_simulation(
        control_type   = control_type,
        modname        = modname,
        parsname       = parsname,
        params_process = params_process,
        params_runs    = [],
        tag_folder     = tag_folder,
        tag_process    = tag_process,
        save_data      = True,
        results_path   = results_path,
    )

    # Define the sampling
    if pop_inputs_process is None:
        sampling = FloatRandomSampling()
    else:
        sampling = pop_inputs_process
        pop_size = pop_inputs_process.shape[0]

    # Define the optimization problem
    problem = problem_class(
        control_type             = control_type,
        n_sub_processes          = n_sub_processes,
        net                      = snn_sim,
        vars_optimization        = pars_optimization,
        obj_optimization         = obj_optimization,
        constr_optimization      = constr_optimization,
        mech_sim_options         = mech_sim_options,
        pop_size                 = pop_size,
        n_gen                    = n_gen,
        gen_index_start          = gen_index_start,
        motor_output_signal_func = motor_output_signal_func,
    )

    # Define the optimization algorithm
    algorithm = NSGA2(
        pop_size = pop_size,
        sampling = sampling,
    )

    # Define the termination condition
    termination = get_termination("n_gen", n_gen)

    # Run the optimization
    res = minimize(
        problem      = problem,
        algorithm    = algorithm,
        termination  = termination,
        seed         = params_process['seed_value'],
        save_history = save_history,
        verbose      = verbose,
    )

    # Clean saved files
    problem.clean_saved_files()

    logging.info('Optimization Completed')
    return

# \----------- [ SINGLE PROCESS ] ------------

# ------------ [ MULTIPLE PROCESSES ] ------------

def _run_optimization(
    control_type        : str,
    modname             : str,
    parsname            : str,
    params_processes    : dict,
    tag_folder          : str,
    pars_optimization   : list[list[str, float, float]],
    obj_optimization    : list[str, str],
    results_path        : str,
    constr_optimization : dict[str, tuple[float]] = None,
    n_sub_processes     : int = 1,
    pop_inputs_processes: list[np.ndarray] = None,
    problem_class       : object = snn_optimization_problem.OptimizationPropblem,
    **kwargs
):
    '''
    Run multiple instances of an optimization pipeline.

    Args:
        control_type (str): Type of control for the simulation.
        modname (str): Name of the module.
        parsname (str): Name of the parameters.
        params_processes (dict): Process parameters.
        tag_folder (str): Folder tag for naming.
        pars_optimization (list[list[str, float, float]]): List of variable optimization parameters.
        obj_optimization (list[str, str]): List of objective optimization parameters.
        results_path (str): Path for saving results.
        constr_optimization (dict[str, tuple[float]], optional): Dictionary of constraint optimization parameters. Defaults to None.
        n_sub_processes (int, optional): Number of sub-processes. Defaults to 1.
        pop_inputs_processes (list[np.ndarray], optional): List of population inputs for optimization. Defaults to None.
        problem_class  (snn_optimization_problem.OptimizationPropblem, optional): Optimization problem class.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    '''

    data_file_tag_list = list(
        set(
            [ pars_proc['simulation_data_file_tag'] for pars_proc in params_processes ]
        )
    )
    assert len(data_file_tag_list) == 1, 'Wrong number of data_file_tag provided'

    opt_pars_folder_name = f'{modname}_{data_file_tag_list[0]}_{tag_folder}'
    snn_optimization_utils.save_optimization_parameters(
        opt_pars_folder_name = opt_pars_folder_name,
        pars_optimization    = pars_optimization,
        obj_optimization     = obj_optimization,
        results_path         = results_path,
        constr_optimization  = constr_optimization,
    )

    # Check for assigned population inputs
    n_processes = len(params_processes)
    if pop_inputs_processes is None:
        pop_inputs_processes = [ None for i in range(n_processes)]

    # Create processes (one optimization per process)
    processes = [
        Process(
            target= _run_optimization_single_process,
            kwargs= kwargs | {
                'control_type'       : control_type,
                'modname'            : modname,
                'parsname'           : parsname,
                'params_process'     : params_process,
                'tag_folder'         : tag_folder,
                'tag_process'        : process_ind,
                'pars_optimization'  : pars_optimization,
                'obj_optimization'   : obj_optimization,
                'results_path'       : results_path,
                'constr_optimization': constr_optimization,
                'n_sub_processes'    : n_sub_processes,
                'pop_inputs_process' : pop_inputs_processes[process_ind],
                'problem_class'      : problem_class,
            }
        )
        for process_ind, params_process in enumerate(params_processes)
    ]

    # Run optimization
    for i in range(n_processes):
        processes[i].start()
        time.sleep(0.1)

    # Terminate
    for i in range(n_processes):
        processes[i].join()

def run_optimization_open_loop(
    modname             : str,
    parsname            : str,
    params_processes    : dict,
    tag_folder          : str,
    pars_optimization   : list[list[str, float, float]],
    obj_optimization    : list[str, str],
    results_path        : str,
    constr_optimization : dict[str, tuple[float]] = None,
    n_sub_processes     : int = 1,
    pop_inputs_processes: list[np.ndarray] = None,
    problem_class       : object = snn_optimization_problem.OptimizationPropblem,
    **kwargs
):
    '''
    Run multiple instances of an optimization pipeline with an open-loop network.

    Args:
        modname (str): Name of the module.
        parsname (str): Name of the parameters.
        params_processes (dict): Process parameters.
        tag_folder (str): Folder tag for naming.
        pars_optimization (list[list[str, float, float]]): List of variable optimization parameters.
        obj_optimization (list[str, str]): List of objective optimization parameters.
        results_path (str): Path for saving results.
        constr_optimization (dict[str, tuple[float]], optional): Dictionary of constraint optimization parameters. Defaults to None.
        n_sub_processes (int, optional): Number of sub-processes. Defaults to 1.
        pop_inputs_processes (list[np.ndarray], optional): List of population inputs for optimization. Defaults to None.
        problem_class  (snn_optimization_problem.OptimizationPropblem, optional): Optimization problem class.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    '''
    _run_optimization(
        control_type         = 'open_loop',
        modname              = modname,
        parsname             = parsname,
        params_processes     = params_processes,
        tag_folder           = tag_folder,
        pars_optimization    = pars_optimization,
        obj_optimization     = obj_optimization,
        results_path         = results_path,
        constr_optimization  = constr_optimization,
        n_sub_processes      = n_sub_processes,
        pop_inputs_processes = pop_inputs_processes,
        problem_class        = problem_class ,
        **kwargs
    )

def run_optimization_closed_loop(
    modname             : str,
    parsname            : str,
    params_processes    : dict,
    tag_folder          : str,
    pars_optimization   : list[list[str, float, float]],
    obj_optimization    : list[str, str],
    results_path        : str,
    constr_optimization : dict[str, tuple[float]] = None,
    n_sub_processes     : int = 1,
    pop_inputs_processes: list[np.ndarray] = None,
    problem_class       : object = snn_optimization_problem.OptimizationPropblem,
    **kwargs
):
    '''
    Run multiple instances of an optimization pipeline with a closed-loop network.

    Args:
        modname (str): Name of the module.
        parsname (str): Name of the parameters.
        params_processes (dict): Process parameters.
        tag_folder (str): Folder tag for naming.
        pars_optimization (list[list[str, float, float]]): List of variable optimization parameters.
        obj_optimization (list[str, str]): List of objective optimization parameters.
        results_path (str): Path for saving results.
        constr_optimization (dict[str, tuple[float]], optional): Dictionary of constraint optimization parameters. Defaults to None.
        n_sub_processes (int, optional): Number of sub-processes. Defaults to 1.
        pop_inputs_processes (list[np.ndarray], optional): List of population inputs for optimization. Defaults to None.
        problem_class  (snn_optimization_problem.OptimizationPropblem, optional): Optimization problem class.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    '''
    _run_optimization(
        control_type         = 'closed_loop',
        modname              = modname,
        parsname             = parsname,
        params_processes     = params_processes,
        tag_folder           = tag_folder,
        pars_optimization    = pars_optimization,
        obj_optimization     = obj_optimization,
        results_path         = results_path,
        constr_optimization  = constr_optimization,
        n_sub_processes      = n_sub_processes,
        pop_inputs_processes = pop_inputs_processes,
        problem_class        = problem_class ,
        **kwargs
    )

def run_optimization_signal_driven(
    modname                 : str,
    parsname                : str,
    params_processes        : dict,
    tag_folder              : str,
    pars_optimization       : list[list[str, float, float]],
    obj_optimization        : list[str, str],
    results_path            : str,
    constr_optimization     : dict[str, tuple[float]] = None,
    n_sub_processes         : int = 1,
    pop_inputs_processes    : list[np.ndarray] = None,
    problem_class           : object = snn_optimization_problem.OptimizationPropblem,
    **kwargs
):
    '''
    Run multiple instances of an optimization pipeline with a signal-driven network.

    Args:
        modname (str): Name of the module.
        parsname (str): Name of the parameters.
        params_processes (dict): Process parameters.
        tag_folder (str): Folder tag for naming.
        pars_optimization (list[list[str, float, float]]): List of variable optimization parameters.
        obj_optimization (list[str, str]): List of objective optimization parameters.
        results_path (str): Path for saving results.
        constr_optimization (dict[str, tuple[float]], optional): Dictionary of constraint optimization parameters. Defaults to None.
        n_sub_processes (int, optional): Number of sub-processes. Defaults to 1.
        pop_inputs_processes (list[np.ndarray], optional): List of population inputs for optimization. Defaults to None.
        problem_class  (snn_optimization_problem.OptimizationPropblem, optional): Optimization problem class.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    '''

    _run_optimization(
        control_type         = 'signal_driven',
        modname              = modname,
        parsname             = parsname,
        params_processes     = params_processes,
        tag_folder           = tag_folder,
        pars_optimization    = pars_optimization,
        obj_optimization     = obj_optimization,
        results_path         = results_path,
        constr_optimization  = constr_optimization,
        n_sub_processes      = n_sub_processes,
        pop_inputs_processes = pop_inputs_processes,
        problem_class        = problem_class ,
        **kwargs
    )

def run_optimization_position_control(
    modname                 : str,
    parsname                : str,
    params_processes        : dict,
    tag_folder              : str,
    pars_optimization       : list[list[str, float, float]],
    obj_optimization        : list[str, str],
    results_path            : str,
    constr_optimization     : dict[str, tuple[float]] = None,
    n_sub_processes         : int = 1,
    pop_inputs_processes    : list[np.ndarray] = None,
    problem_class           : object = snn_optimization_problem.OptimizationPropblem,
    **kwargs
):
    '''
    Run multiple instances of an optimization pipeline with a position-controlled network.

    Args:
        modname (str): Name of the module.
        parsname (str): Name of the parameters.
        params_processes (dict): Process parameters.
        tag_folder (str): Folder tag for naming.
        pars_optimization (list[list[str, float, float]]): List of variable optimization parameters.
        obj_optimization (list[str, str]): List of objective optimization parameters.
        results_path (str): Path for saving results.
        constr_optimization (dict[str, tuple[float]], optional): Dictionary of constraint optimization parameters. Defaults to None.
        n_sub_processes (int, optional): Number of sub-processes. Defaults to 1.
        pop_inputs_processes (list[np.ndarray], optional): List of population inputs for optimization. Defaults to None.
        problem_class  (snn_optimization_problem.OptimizationPropblem, optional): Optimization problem class.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    '''

    _run_optimization(
        control_type         = 'position_control',
        modname              = modname,
        parsname             = parsname,
        params_processes     = params_processes,
        tag_folder           = tag_folder,
        pars_optimization    = pars_optimization,
        obj_optimization     = obj_optimization,
        results_path         = results_path,
        constr_optimization  = constr_optimization,
        n_sub_processes      = n_sub_processes,
        pop_inputs_processes = pop_inputs_processes,
        problem_class        = problem_class ,
        **kwargs
    )

def run_optimization_hybrid_position_control(
    modname             : str,
    parsname            : str,
    params_processes    : dict,
    tag_folder          : str,
    pars_optimization   : list[list[str, float, float]],
    obj_optimization    : list[str, str],
    results_path        : str,
    constr_optimization : dict[str, tuple[float]] = None,
    n_sub_processes     : int = 1,
    pop_inputs_processes: list[np.ndarray] = None,
    problem_class       : object = snn_optimization_problem.OptimizationPropblem,
    **kwargs
):
    '''
    Run multiple instances of an optimization pipeline with a hybrid position control network.

    Args:
        modname (str): Name of the module.
        parsname (str): Name of the parameters.
        params_processes (dict): Process parameters.
        tag_folder (str): Folder tag for naming.
        pars_optimization (list[list[str, float, float]]): List of variable optimization parameters.
        obj_optimization (list[str, str]): List of objective optimization parameters.
        results_path (str): Path for saving results.
        constr_optimization (dict[str, tuple[float]], optional): Dictionary of constraint optimization parameters. Defaults to None.
        n_sub_processes (int, optional): Number of sub-processes. Defaults to 1.
        pop_inputs_processes (list[np.ndarray], optional): List of population inputs for optimization. Defaults to None.
        problem_class  (snn_optimization_problem.OptimizationPropblem, optional): Optimization problem class.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    '''
    _run_optimization(
        control_type         = 'hybrid_position_control',
        modname              = modname,
        parsname             = parsname,
        params_processes     = params_processes,
        tag_folder           = tag_folder,
        pars_optimization    = pars_optimization,
        obj_optimization     = obj_optimization,
        results_path         = results_path,
        constr_optimization  = constr_optimization,
        n_sub_processes      = n_sub_processes,
        pop_inputs_processes = pop_inputs_processes,
        problem_class        = problem_class ,
        **kwargs
    )
# \----------- [ MULTIPLE PROCESSES ] ------------
