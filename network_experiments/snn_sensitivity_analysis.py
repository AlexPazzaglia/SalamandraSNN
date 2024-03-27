''' Module to perform sensitivity analysis '''

import os
import copy
import logging
import dill

import numpy as np
from typing import Callable

from SALib.analyze import sobol
from SALib.sample import saltelli

from network_modules.parameters.network_parameters import SnnParameters

from network_experiments import snn_simulation

## UTILITY FUNCTIONS
def _get_sensitivity_analysis_parameters(
    results_path    : str,
    parsname        : str,
    n_saltelli      : int,
    sal_problem     : dict,
    params_processes: list[dict],
    run_params_func : Callable,
) -> tuple[dict, list[dict]]:
    '''
    Function to sample the parameters for the sensitivity analysis (Saltelli)
    '''

    # Check consistency of process parameters
    params_processes_keys = set(
        [key for params_process in params_processes for key in params_process]
    )
    params_processes_all = {
        key : [ params_process[key] for params_process in params_processes ]
        for key in params_processes_keys
        if key not in ['seed_value', 'tag_process', 'mech_sim_options', 'motor_output_signal_func']
    }

    assert np.all(
        [
            len(params_processes_all[key]) == 1 or
            (
                # Scalar
                np.unique(params_processes_all[key]).size == 1
                if isinstance(params_processes_all[key][0], (int, float, str))
                else False
            ) or
            (
                # Array
                len( np.unique(params_processes_all[key]).shape )== 1
                if isinstance(params_processes_all[key][0], (np.ndarray, list, tuple, set))
                else False
            )
            for key in params_processes_all
        ]
    ), 'All parameters should have the same value for all processes'

    params_processes_shared = {
        key : params_processes_all[key][0]
        for key in params_processes_all
    }

    # Get default parameters
    default_parameters = SnnParameters(
        results_path = results_path,
        parsname     = parsname,
        new_pars     = params_processes_shared
    )

    # N*(D+2) iterations
    sampled_values = saltelli.sample(sal_problem, n_saltelli, False)
    params_runs    = run_params_func(
        default_parameters = default_parameters,
        sal_problem        = sal_problem,
        sampled_values     = sampled_values,
    )

    return params_runs

def _get_sensitivity_analysis_path(
        modname         : str,
        tag_folder      : str,
        params_processes: list[dict],
        results_path    : str,
        params_runs     : list[dict] = None,
) -> list[str]:
    ''' Get the path for saving the results of the analysis '''

    if params_runs is None:
        params_runs = []

    # Check if modname is a path
    if os.path.isfile(modname) and modname.endswith('.py'):
        modname = os.path.basename(modname)[: -3]

    # Check for additional file tag
    file_tag_list_processes = [
        par_proc.get('simulation_data_file_tag')
        for par_proc in params_processes
        if par_proc.get('simulation_data_file_tag') not in ['', None]
    ]

    file_tag_list_runs = [
        par_run.get('simulation_data_file_tag')
        for par_run in params_runs
        if par_run.get('simulation_data_file_tag') not in ['', None]
    ]

    file_tag_list_processes = list(set(file_tag_list_processes))
    file_tag_list_runs      = list(set(file_tag_list_runs))

    assert len(file_tag_list_processes) == 1, \
            'simulation_data_file_tag not provided or provided multiple times'
    assert len(file_tag_list_runs) == 0, \
            'simulation_data_file_tag should be provided in params_processes'

    data_file_tag = file_tag_list_processes[0] if file_tag_list_processes else ''

    folder_name = f'{modname}_{data_file_tag}_{tag_folder}'
    folder_path = f'{results_path}/data/{folder_name}'

    return folder_path, folder_name

def _sensitivity_analysis_save_info(
    control_type    : bool,
    modname         : str,
    parsname        : str,
    n_saltelli      : int,
    params_processes: list[dict],
    params_runs     : list[dict],
    tag_folder      : str,
    problem         : dict,
    results_path    : str,
):
    ''' Write to files containing information about the sensitivity analysis '''

    analysis_info = copy.deepcopy(problem)
    analysis_info['control_type']     = control_type
    analysis_info['modname']          = modname
    analysis_info['parsname']         = parsname
    analysis_info['n_saltelli']       = n_saltelli
    analysis_info['params_processes'] = params_processes
    analysis_info['params_runs']      = params_runs
    analysis_info['tag_folder']       = tag_folder

    # Saving folder
    folder_path, _folder_name = _get_sensitivity_analysis_path(
        modname          = modname,
        tag_folder       = tag_folder,
        params_processes = params_processes,
        results_path     = results_path,
        params_runs      = params_runs,
    )
    os.makedirs(folder_path, exist_ok= True)

    data_file_dll = f'{folder_path}/sensitivity_analysis_parameters.dill'
    data_file_txt = f'{folder_path}/sensitivity_analysis_parameters.txt'

    # Save information
    with open(data_file_dll, 'wb') as outfile:
        dill.dump(analysis_info, outfile)

    with open(data_file_txt, 'w', encoding='UTF-8') as outfile:
        text = [
            f'{key} : {analysis_info[key]}'
            for key in sorted(analysis_info)
            if  key not in ['params_runs']
        ]
        outfile.write('\n'.join(text))

## RUN ANALYSIS
def _sensitivity_analysis_run(
    control_type    : str,
    modname         : str,
    parsname        : str,
    n_saltelli      : int,
    sal_problem     : dict,
    params_processes: list[dict],
    tag_folder      : str,
    results_path    : str,
    run_params_func : Callable,
) -> None:
    '''
    Function to run a sensitivity analysis with the selected network
    Specify N for the Saltelli sample, note that Iterations = N*(D+2)
    Where D is the number of parameters we are modifying in the network.
    Specify also the variation range for the parameters.\n
    Ex: N=8, var_range= [0.98, 1.02]
    '''

    # Define sensitivity analysis problem and parameter combinations
    param_runs = _get_sensitivity_analysis_parameters(
        results_path     = results_path,
        parsname         = parsname,
        n_saltelli       = n_saltelli,
        sal_problem      = sal_problem,
        params_processes = params_processes,
        run_params_func  = run_params_func,
    )

    # Save information on the analysis
    _sensitivity_analysis_save_info(
        control_type     = control_type,
        modname          = modname,
        parsname         = parsname,
        n_saltelli       = n_saltelli,
        params_processes = params_processes,
        params_runs      = param_runs,
        tag_folder       = tag_folder,
        problem          = sal_problem,
        results_path     = results_path,
    )

    # Select open or closed loop function
    if control_type == 'closed_loop':
        sal_function = snn_simulation.simulate_multi_net_multi_run_closed_loop
    elif control_type == 'open_loop':
        sal_function = snn_simulation.simulate_multi_net_multi_run_open_loop
    elif control_type == 'signal_driven':
        sal_function = snn_simulation.simulate_multi_net_multi_run_signal_driven
    elif control_type == 'position_control':
        sal_function = snn_simulation.simulate_multi_net_multi_run_position_control
    elif control_type == 'hybrid_position_control':
        sal_function = snn_simulation.simulate_multi_net_multi_run_hybrid_position_control
    else:
        raise ValueError(f'control_type {control_type} not recognized')

    # Run analysis
    sal_function(
        modname             = modname,
        parsname            = parsname,
        params_processes    = params_processes,
        params_runs         = param_runs,
        tag_folder          = tag_folder,
        results_path        = results_path,
        delete_connectivity = False,
    )

def sensitivity_analysis_open_loop_run(
    modname         : str,
    parsname        : str,
    n_saltelli      : int,
    sal_problem     : dict,
    params_processes: list[dict],
    tag_folder      : str,
    results_path    : str,
    run_params_func : Callable,
) -> None:
    '''
    Function to run a sensitivity analysis in open loop with the selected network
    Specify N for the Saltelli sample, note that Iterations = N*(D+2)
    Where D is the number of parameters we are modifying in the network.
    Specify also the variation range for the parameters.\n
    Ex: N=8, var_range= [0.98, 1.02]
    '''

    _sensitivity_analysis_run(
        control_type     = 'open_loop',
        modname          = modname,
        parsname         = parsname,
        n_saltelli       = n_saltelli,
        sal_problem      = sal_problem,
        params_processes = params_processes,
        tag_folder       = tag_folder,
        results_path     = results_path,
        run_params_func  = run_params_func,
    )

def sensitivity_analysis_closed_loop_run(
    modname         : str,
    parsname        : str,
    n_saltelli      : int,
    sal_problem     : dict,
    params_processes: list[dict],
    tag_folder      : str,
    results_path    : str,
    run_params_func : Callable,
) -> None:
    '''
    Function to run a sensitivity analysis in closed loop with the selected network
    Specify N for the Saltelli sample, note that Iterations = N*(D+2)
    Where D is the number of parameters we are modifying in the network.
    Specify also the variation range for the parameters.\n
    Ex: N=8, var_range= [0.98, 1.02]
    '''

    _sensitivity_analysis_run(
        control_type     = 'closed_loop',
        modname          = modname,
        parsname         = parsname,
        n_saltelli       = n_saltelli,
        sal_problem      = sal_problem,
        params_processes = params_processes,
        tag_folder       = tag_folder,
        results_path     = results_path,
        run_params_func  = run_params_func,
    )

def sensitivity_analysis_signal_driven_run(
    modname         : str,
    parsname        : str,
    n_saltelli      : int,
    sal_problem     : dict,
    params_processes: list[dict],
    tag_folder      : str,
    results_path    : str,
    run_params_func : Callable,
) -> None:
    '''
    Function to run a sensitivity analysis in a signal-driven network
    Specify N for the Saltelli sample, note that Iterations = N*(D+2)
    Where D is the number of parameters we are modifying in the network.
    Specify also the variation range for the parameters.\n
    Ex: N=8, var_range= [0.98, 1.02]
    '''

    _sensitivity_analysis_run(
        control_type     = 'signal_driven',
        modname          = modname,
        parsname         = parsname,
        n_saltelli       = n_saltelli,
        sal_problem      = sal_problem,
        params_processes = params_processes,
        tag_folder       = tag_folder,
        results_path     = results_path,
        run_params_func  = run_params_func,
    )

def sensitivity_analysis_position_control_run(
    modname         : str,
    parsname        : str,
    n_saltelli      : int,
    sal_problem     : dict,
    params_processes: list[dict],
    tag_folder      : str,
    results_path    : str,
    run_params_func : Callable,
) -> None:
    '''
    Function to run a sensitivity analysis in a position-controlled network
    Specify N for the Saltelli sample, note that Iterations = N*(D+2)
    Where D is the number of parameters we are modifying in the network.
    Specify also the variation range for the parameters.\n
    Ex: N=8, var_range= [0.98, 1.02]
    '''

    _sensitivity_analysis_run(
        control_type     = 'position_control',
        modname          = modname,
        parsname         = parsname,
        n_saltelli       = n_saltelli,
        sal_problem      = sal_problem,
        params_processes = params_processes,
        tag_folder       = tag_folder,
        results_path     = results_path,
        run_params_func  = run_params_func,
    )

def sensitivity_analysis_hybrid_position_control_run(
    modname         : str,
    parsname        : str,
    n_saltelli      : int,
    sal_problem     : dict,
    params_processes: list[dict],
    tag_folder      : str,
    results_path    : str,
    run_params_func : Callable,
) -> None:
    '''
    Function to run a sensitivity analysis in hybrid position control with the selected network
    Specify N for the Saltelli sample, note that Iterations = N*(D+2)
    Where D is the number of parameters we are modifying in the network.
    Specify also the variation range for the parameters.\n
    Ex: N=8, var_range= [0.98, 1.02]
    '''

    _sensitivity_analysis_run(
        control_type     = 'hybrid_position_control',
        modname          = modname,
        parsname         = parsname,
        n_saltelli       = n_saltelli,
        sal_problem      = sal_problem,
        params_processes = params_processes,
        tag_folder       = tag_folder,
        results_path     = results_path,
        run_params_func  = run_params_func,
    )

## POST-PROCESSING
def sensitivity_analysis_post_processing(
    modname         : str,
    tag_folder      : str,
    params_processes: list[dict],
    sal_problem     : dict,
    results_path    : str,
):
    ''' Compute resulting sensitivity indices '''

    logging.info('Computing sensitivity indices')

    folder_path, folder_name = _get_sensitivity_analysis_path(
        modname          = modname,
        tag_folder       = tag_folder,
        params_processes = params_processes,
        results_path     = results_path,
    )

    for process_ind in range(len(params_processes)):
        data_file = f'{folder_path}/{folder_name}_{process_ind}/snn_performance_process.dill'

        with open(data_file, "rb") as infile:
            metrics_runs : dict = dill.load(infile)

        ptcc_ax_runs = metrics_runs['ptcc_ax']
        freq_ax_runs = metrics_runs['freq_ax']
        ipls_ax_runs = metrics_runs['ipls_ax']

        # Compute sensitivity indices
        sensitivity_indices = {}
        sensitivity_indices['ptcc_ax'] = sobol.analyze(sal_problem, ptcc_ax_runs, False)
        sensitivity_indices['freq_ax'] = sobol.analyze(sal_problem, freq_ax_runs, False)
        sensitivity_indices['ipls_ax'] = sobol.analyze(sal_problem, ipls_ax_runs, False)

        # Save results
        results_file =  f'{folder_path}/{folder_name}_{process_ind}/snn_sensitivity_indices.dill'
        with open(results_file, 'wb') as outfile:
            dill.dump(sensitivity_indices, outfile)
