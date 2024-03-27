''' Analyze results of an optimization '''
import os
import logging

from typing import Union, Any

import dill
import numpy as np

from network_experiments import (
    snn_utils,
    snn_optimization_problem,
    snn_simulation_data,
    snn_simulation_replay,
    snn_simulation_results,
)

# TYPING
RESULT_GEN = dict[str, list[np.ndarray]]    # GENERATION
RESULT_PRC = list[RESULT_GEN]               # PROCESS
RESULT_TOT = list[RESULT_PRC]               # TOTAL

# DATA LOADING
def load_optimization_parameters(
    opt_pars_folder_name: str,
    results_path        : str,
    full_path           : str = None,
):
    ''' Save parameters of the optimization '''

    filename = (
        f'{results_path}/{opt_pars_folder_name}/parameters_optimization.dill'
        if full_path is None
        else full_path
    )
    logging.info('Loading parameters_optimization data from %s', filename)
    with open(filename, 'rb') as infile:
        pars_optimization  : list[tuple[str, float, float]] = dill.load(infile)
        obj_optimization   : list[tuple[str, str]]          = dill.load(infile)
        constr_optimization: dict[str, tuple[float]]        = dill.load(infile)

    return pars_optimization, obj_optimization, constr_optimization

def load_optimization_data_process(
    process_results_folder_name : str,
) -> tuple[RESULT_PRC, list[str], list[str]]:
    ''' Load optimization results from a process '''

    # Internal parameters
    filename = f'{process_results_folder_name}/parameters_optimization_process.dill'
    logging.info('Loading parameters_optimization_process data from %s', filename)
    with open(filename, 'rb') as infile:
        pars_names_list    : list[str] = dill.load(infile)
        metrics_names_list : list[str] = dill.load(infile)

    # Process parameters
    filename = f'{process_results_folder_name}/snn_parameters_process.dill'
    logging.info('Loading snn_parameters_process data from %s', filename)
    with open(filename, 'rb') as infile:
        params_process = dill.load(infile)

    # Results actoss generations
    gen = 0
    result_process : RESULT_PRC = []
    while True:
        filename = (
            f'{process_results_folder_name}/'
            f'optimization_results/generation_{gen}.dill'
        )
        if not os.path.isfile(filename):
            break
        logging.info('Loading generation %i data from %s', gen, filename)
        with open(filename, 'rb') as infile:
            result_process.append( dill.load(infile) )

        gen+= 1

    return result_process, pars_names_list, metrics_names_list

def load_optimization_results(
    opt_results_folder_name: str,
    results_path           : str,
    full_path              : str = None,
    processess_inds        : list[int] = None,
) -> tuple[ RESULT_TOT, list[str], list[str] ]:
    ''' Load optimization results from all processes '''

    results_folder_path = (
        f'{results_path}/{opt_results_folder_name}'
        if full_path is None
        else full_path
    )

    if processess_inds is None:
        processess_inds  = snn_simulation_results.get_inds_processes(results_folder_path)

    results_list  : RESULT_TOT       = []
    params_names  : list[list[str]]  = []
    metrics_names : list[list[str]]  = []

    for process_ind in processess_inds:
        result_folder_process  = f'{results_folder_path}/process_{process_ind}'

        (
            result_p,
            pars_names_p,
            metrics_names_p,
        )= load_optimization_data_process(result_folder_process)

        results_list.append(result_p)
        params_names.append(pars_names_p)
        metrics_names.append(metrics_names_p)

    params_names  = [list(x) for x in set(tuple(x) for x in params_names)]
    metrics_names = [list(x) for x in set(tuple(x) for x in metrics_names)]
    assert  len(params_names) == 1, 'Params names not uniquely defined'
    assert len(metrics_names) == 1, 'Metrics names not uniquely defined'

    return results_list, params_names[0], metrics_names[0]

def load_generation_inputs(
    opt_folder_name: str,
    results_path   : str,
    generation     : int,
    processess_inds: list[int] = None,
):
    '''
    Loads the parameters of a given generation of the optimization
    Needed to restart the optimization from a desired point.
    '''

    (
        results_list,
        _params_names,
        _metrics_names
    ) = load_optimization_results(
        opt_results_folder_name = opt_folder_name,
        results_path            = results_path,
        processess_inds         = processess_inds,
    )

    generation_inputs = [
        results_process['inputs'][generation]
        for results_process in results_list
    ]

    return generation_inputs

def load_all_optimization_data(
    results_path: str,
    folder_name : str,
    model_name  : str = None,
) -> dict[str, Any]:
    ''' Loads all data and parameters from the optimization '''

    # Get model name and parameters name
    modname, parsname, tag_folder = snn_utils.get_configuration_files_from_folder_name(
        folder_name = f'{results_path}/{folder_name}',
        model_name  = model_name,
        open_loop   = False,
    )

    # Get optimization parameters
    (
        pars_optimization,
        obj_optimization,
        constr_optimization
    ) = load_optimization_parameters(
        folder_name,
        results_path,
    )

    # Expand optimization objectives
    obj_optimization_dict = get_expanded_optimization_objectives(obj_optimization)

    # Get optimization results
    (
        results_all,
        params_names,
        metrics_names
    ) = load_optimization_results(
        folder_name,
        results_path = results_path,
    )

    optimization_data = {
        'folder_name'          : folder_name,
        'results_path'         : results_path,
        'modname'              : modname,
        'parsname'             : parsname,
        'tag_folder'           : tag_folder,
        'pars_optimization'    : pars_optimization,
        'obj_optimization'     : obj_optimization,
        'obj_optimization_dict': obj_optimization_dict,
        'constr_optimization'  : constr_optimization,
        'results_all'          : results_all,
        'params_names'         : params_names,
        'metrics_names'        : metrics_names,
    }

    return optimization_data

# RESULTS ANALYSIS
def get_expanded_optimization_objectives(
    obj_optimization: dict[str, list[str, str]],
) -> dict[str, dict[str, Union[str, int]]]:
    ''' Get expanded optimization objectives '''
    obj_optimization_parameters = {}
    for obj_pars in obj_optimization:
        obj_optimization_parameters[obj_pars[0]] = {
            'name'  : obj_pars[0],
            'label' : obj_pars[0].upper(),
            'type'  : obj_pars[1],
            'sign'  : -1 if obj_pars[1] == 'max' else +1,
            'target': obj_pars[2] if len(obj_pars) > 2 else None,
        }

    return obj_optimization_parameters

def get_individuals_satisfying_constraints(
    results_gen        : RESULT_GEN,
    constr_optimization: dict[str, list[float, float]],
    check_constraints  : bool,
    constr_additional  : dict[str, list[float, float]] = None,
) -> list[int]:
    ''' Get individuals satisfying constraints '''

    # NOTE: Additional constraints are always checked
    if not check_constraints and constr_additional is None:
        constr_optimization = {}
    else:
        constr_optimization = {} if constr_optimization is None else constr_optimization
        constr_additional   = {} if constr_additional is None else constr_additional
        constr_optimization = constr_optimization | constr_additional

    individuals_list = []
    for individual in range(len(results_gen['outputs'])):

        # Skip individuals with NaN values
        if np.isnan( results_gen['outputs'][individual] ).any():
            continue

        # Check constraints
        satisfied_constr = True

        for constr_key, constr_vals in  constr_optimization.items():
            metric_constr_value = results_gen[constr_key][individual]
            if constr_vals[0] is not None and metric_constr_value < constr_vals[0]:
                satisfied_constr = False
                break
            if constr_vals[1] is not None and metric_constr_value > constr_vals[1]:
                satisfied_constr = False
                break

        if satisfied_constr:
            individuals_list.append(individual)

    return np.array( individuals_list )

def get_best_from_generation(
    results_gen        : RESULT_GEN,
    metric_key         : str,
    obj_optimization   : dict[str, dict[str, Union[str, float]]],
    constr_optimization: dict[str, list[float, float]],
    check_constraints  : bool = True,
    constr_additional  : dict[str, list[float, float]] = None,
) -> tuple[float, float, int] :
    ''' Return best performance for a given generation and metric, satisfying constraints'''

    # Get individuals satisfying constraints
    metrics_results_inds = get_individuals_satisfying_constraints(
        results_gen         = results_gen,
        constr_optimization = constr_optimization,
        check_constraints   = check_constraints,
        constr_additional   = constr_additional,
    )

    if len(metrics_results_inds) == 0:
        return np.nan, np.nan, np.nan

    metrics_results = np.array(results_gen[metric_key])[metrics_results_inds]

    # Get best individual
    target = None
    if obj_optimization[metric_key]['type'] == 'min':
        best_pos_pruned = np.argmin(metrics_results)
    if obj_optimization[metric_key]['type'] == 'max':
        best_pos_pruned = np.argmax(metrics_results)
    if obj_optimization[metric_key]['type'] == 'trg':
        target          = obj_optimization[metric_key]['target']
        best_pos_pruned = np.argmin( np.abs(metrics_results - target) )

    best_pos     = metrics_results_inds[best_pos_pruned]
    best_met_val = metrics_results[best_pos_pruned]
    best_obj_val = (
        np.abs( metrics_results[best_pos_pruned] - target )
        if target is not None
        else
        best_obj_val
    )

    return best_obj_val, best_met_val, best_pos

def get_ranking_from_generation(
    results_gen        : RESULT_GEN,
    metric_key         : str,
    obj_optimization   : dict[str, dict[str, Union[str, float]]],
    constr_optimization: dict[str, list[float, float]],
    check_constraints  : bool = True,
    constr_additional  : dict[str, list[float, float]] = None,
) -> tuple[float, float, int] :
    ''' Return ranked performance for a given generation and metric, satisfying constraints'''

    # Get individuals satisfying constraints
    metrics_results_inds = get_individuals_satisfying_constraints(
        results_gen         = results_gen,
        constr_optimization = constr_optimization,
        check_constraints   = check_constraints,
        constr_additional   = constr_additional,
    )

    if len(metrics_results_inds) == 0:
        return np.nan, np.nan, np.nan

    metrics_results = np.array(results_gen[metric_key])[metrics_results_inds]

    # Get best individual
    target = None
    if obj_optimization[metric_key]['type'] == 'min':
        ranking_pos = np.argsort(metrics_results)

    if obj_optimization[metric_key]['type'] == 'max':
        ranking_pos = np.argsort(metrics_results)[::-1]

    if obj_optimization[metric_key]['type'] == 'trg':
        target          = obj_optimization[metric_key]['target']
        ranking_pos = np.argsort(np.abs(metrics_results - target))

    ranking_pos_gen = metrics_results_inds[ranking_pos]
    ranking_val_gen = metrics_results[ranking_pos]
    ranking_obj_gen = (
        np.abs( ranking_val_gen - target )
        if target is not None
        else
        ranking_val_gen
    )

    return ranking_obj_gen, ranking_val_gen, ranking_pos_gen

def get_best_evolution_across_generations(
    results_all        : RESULT_TOT,
    metric_key         : str,
    obj_optimization   : dict[str, dict[str, str]],
    constr_optimization: dict[str, list[float, float]] = None,
    check_constraints  : bool = True,
    constr_additional  : dict[str, list[float, float]] = None,
    inds_processes     : list[int] = None,
    inds_generations   : list[int] = None,
) -> tuple[ np.ndarray, np.ndarray, np.ndarray ]:
    ''' Evolution over the generations for a given metric '''

    # Parameters
    if inds_processes is None:
        inds_processes = range(len(results_all))

    if inds_generations is None:
        inds_generations = range(len(results_all[0]))

    n_processes_considered   = len(inds_processes)
    n_generations_considered = len(inds_generations)

    best_met = np.zeros( (n_processes_considered, n_generations_considered) )
    best_obj = np.zeros_like(best_met)
    best_pos = np.zeros_like(best_met)

    for process_ind in inds_processes:
        for generation in inds_generations:
            (
                best_obj[process_ind, generation],
                best_met[process_ind, generation],
                best_pos[process_ind, generation],
            ) = get_best_from_generation(
                results_gen         = results_all[process_ind][generation],
                metric_key          = metric_key,
                obj_optimization    = obj_optimization,
                constr_optimization = constr_optimization,
                check_constraints   = check_constraints,
                constr_additional   = constr_additional,
            )

    return best_obj, best_met, best_pos

def get_ranking_across_generations(
    results_all        : RESULT_TOT,
    metric_key         : str,
    obj_optimization   : dict[str, dict[str, str]],
    constr_optimization: dict[str, list[float, float]] = None,
    check_constraints  : bool = True,
    constr_additional  : dict[str, list[float, float]] = None,
    inds_processes     : list[int] = None,
    inds_generations   : list[int] = None,
) -> tuple[ np.ndarray, np.ndarray, np.ndarray ]:
    ''' Ranking over the generations for a given metric '''

    # Parameters
    if inds_processes is None:
        inds_processes = range(len(results_all))

    if inds_generations is None:
        inds_generations = range(len(results_all[0]))

    ranking_obj = []
    ranking_val = []
    ranking_pos = []

    for process_ind in inds_processes:

        ranking_obj_proc = []
        ranking_val_proc = []
        ranking_pos_proc = []

        # Get ranking for each generation
        for generation in inds_generations:
            (
                ranking_obj_gen,
                ranking_val_gen,
                ranking_pos_gen,
            ) = get_ranking_from_generation(
                results_gen         = results_all[process_ind][generation],
                metric_key          = metric_key,
                obj_optimization    = obj_optimization,
                constr_optimization = constr_optimization,
                check_constraints   = check_constraints,
                constr_additional   = constr_additional,
            )

            ranking_obj_proc.append(ranking_obj_gen)
            ranking_val_proc.append(ranking_val_gen)
            ranking_pos_proc.append(ranking_pos_gen)

        # Flatten process results across generations
        gen_sizes = [
            len(ranking_obj_gen)
            if not np.isnan(ranking_obj_gen).all()
            else 0
            for ranking_obj_gen in ranking_obj_proc
        ]

        ranking_obj_flat = np.array(
            [
                rank
                for gen, ranking_obj_gen in enumerate(ranking_obj_proc)
                if gen_sizes[gen] > 0
                for rank in ranking_obj_gen
            ]
        )

        ranking_val_flat = np.array(
            [
                rank
                for gen, ranking_val_gen in enumerate(ranking_val_proc)
                if gen_sizes[gen] > 0
                for rank in ranking_val_gen
            ]
        )

        ranking_pos_flat = np.array(
            [
                [gen, rank]
                for gen, ranking_pos_gen in enumerate(ranking_pos_proc)
                if gen_sizes[gen] > 0
                for rank in ranking_pos_gen
            ]
        )

        # Get ranking across generations
        ranking_inds = np.argsort(ranking_obj_flat)

        ranking_obj.append(ranking_obj_flat[ranking_inds])
        ranking_val.append(ranking_val_flat[ranking_inds])
        ranking_pos.append(ranking_pos_flat[ranking_inds])

    return ranking_obj, ranking_val, ranking_pos

def get_statistics_from_generation(
    results_gen        : RESULT_GEN,
    metric_key         : str,
    constr_optimization: dict[str, list[float, float]],
    check_constraints  : bool = True,
    constr_additional  : dict[str, list[float, float]] = None,
) -> dict[str, float] :
    ''' Return performance for a given generation and metric, satisfying constraints'''

    # Get individuals satisfying constraints
    metrics_results_inds = get_individuals_satisfying_constraints(
        results_gen        = results_gen,
        constr_optimization= constr_optimization,
        check_constraints  = check_constraints,
        constr_additional  = constr_additional,
    )

    if len(metrics_results_inds) == 0:
        return {
            'mean'  : np.nan,
            'std'   : np.nan,
            'min'   : np.nan,
            'max'   : np.nan,
            'median': np.nan,
        }

    metrics_results = np.array(results_gen[metric_key])[metrics_results_inds]

    # Get best individual
    metrics_statistics = {
        'mean'  : np.mean(metrics_results),
        'std'   : np.std(metrics_results),
        'min'   : np.amin(metrics_results),
        'max'   : np.amax(metrics_results),
        'median': np.median(metrics_results),
    }

    return metrics_statistics

def get_statistics_across_generations(
    results_all        : RESULT_TOT,
    metric_key         : str,
    constr_optimization: dict[str, list[float, float]],
    check_constraints  : bool = True,
    constr_additional  : dict[str, list[float, float]] = None,
    **kwargs,
) -> dict[str, float] :
    ''' Return performance for a given metric, satisfying constraints, across generations'''

    metrics_statistics = [
        [
            get_statistics_from_generation(
                results_gen         = results_gen,
                metric_key          = metric_key,
                constr_optimization = constr_optimization,
                check_constraints   = check_constraints,
                constr_additional   = constr_additional,
            )
            for results_gen in results_proc
        ]
        for results_proc in results_all
    ]
    return metrics_statistics

def get_individual_input_from_generation(
    result    : RESULT_GEN,
    individual: int,
) -> np.ndarray:
    ''' Parameters provided as input to the specified individual '''
    return result['inputs'][individual]

def get_quantity_distribution_from_generation(
    results_gen        : RESULT_GEN,
    metric_key         : str,
    obj_optimization   : dict[str, dict[str, Union[str,float]]],
    constr_optimization: dict[str, list[float, float]],
    check_constraints  : bool = True,
    constr_additional  : dict[str, list[float, float]] = None,
):
    ''' Get distribution of a given quantity in a generation '''

    # Get individuals satisfying constraints
    individual_inds = get_individuals_satisfying_constraints(
        results_gen         = results_gen,
        constr_optimization = constr_optimization,
        check_constraints   = check_constraints,
        constr_additional   = constr_additional,
    )

    if len(individual_inds) == 0:
        return np.array([])

    # Get the quantity distribution
    metrics_values = np.array( results_gen[metric_key] )
    quantity_distr = metrics_values

    # # Check if the metric is an objective with a target
    # is_obj = metric_key in obj_optimization.keys()
    # target = obj_optimization[metric_key]['target'] if is_obj else  None
    # quantity_distr = (
    #     metrics_values
    #     if target is None
    #     else
    #     np.abs( metrics_values - target )
    # )

    return quantity_distr[individual_inds]

def get_quantity_distribution_across_generations(
    results_proc       : RESULT_PRC,
    generations        : Union[int, list[int]],
    metric_key         : str,
    obj_optimization   : dict[str, dict[str, Union[str,float]]],
    constr_optimization: dict[str, list[float, float]],
    check_constraints  : bool = True,
    constr_additional  : dict[str, list[float, float]] = None,
):
    ''' Get distribution of a given quantity across generations '''

    generations = range(generations) if isinstance(generations, int) else generations
    quantity_distr = [
        get_quantity_distribution_from_generation(
            results_gen         = results_proc[generation],
            metric_key          = metric_key,
            obj_optimization    = obj_optimization,
            constr_optimization = constr_optimization,
            check_constraints   = check_constraints,
            constr_additional   = constr_additional,
        )
        for generation in generations
    ]
    return quantity_distr

def get_quantity_range_across_generations(
    results_proc       : RESULT_PRC,
    generations        : Union[int, list[int]],
    metric_key         : str,
    obj_optimization   : dict[str, dict[str, Union[str,float]]],
    constr_optimization: dict[str, list[float, float]],
    check_constraints  : bool = True,
    constr_additional  : dict[str, list[float, float]] = None,
):
    ''' Get range of a given quantity across generations '''

    # Get quantity distribution
    generations    = range(generations) if isinstance(generations, int) else generations
    quantity_distr = get_quantity_distribution_across_generations(
        results_proc        = results_proc,
        generations         = generations,
        metric_key          = metric_key,
        obj_optimization    = obj_optimization,
        constr_optimization = constr_optimization,
        check_constraints   = check_constraints,
        constr_additional   = constr_additional,
    )

    # Get quantity range
    quantity_range = [
        np.amin(
            [ np.amin(distr_gen) for distr_gen in quantity_distr if len(distr_gen) ]
        ),
        np.amax(
            [ np.amax(distr_gen) for distr_gen in quantity_distr if len(distr_gen) ]
        ),
    ]
    return np.array(quantity_range)

# RESUME OPTIMIZATION
def get_parameters_for_optimization_continuation(
    results_path  : str,
    folder_name   : str,
    model_name    : str = None,
    new_obj_opt   : dict[str, dict[str, Union[str,float]]] = None,
    new_constr_opt: dict[str, list[float, float]] = None,
):
    ''' Loads parameters to setup the continuation of an optimization '''

    # Get optimization data
    opt_data = load_all_optimization_data(
        results_path = results_path,
        folder_name  = folder_name,
        model_name   = model_name,
    )
    results_all = opt_data['results_all']

    if new_obj_opt is not None:
        opt_data['obj_optimization']      = new_obj_opt
        opt_data['obj_optimization_dict'] = get_expanded_optimization_objectives(new_obj_opt)

    if new_constr_opt is not None:
        opt_data['constr_optimization'] = new_constr_opt

    # Get number of previous generations
    n_gen_all = [
        results_proc[-1]['generation'] + 1
        for results_proc in results_all
    ]
    gen_index_start = np.amin(n_gen_all)

    # Get inputs of the last generation
    inputs_last_generation_all = [
        results_proc[-1]['inputs']
        for results_proc in results_all
    ]

    # Get neural simulation parameters
    n_processes = len(results_all)
    params_processes = [
        snn_simulation_data.load_parameters_process(
            folder_name  = folder_name,
            tag_process  = str(process_ind),
            results_path = results_path,
        )[0]
        for process_ind in range(n_processes)
    ]

    # Collect parameters to resume optimization
    pars_resume = opt_data | {
        'gen_index_start'         : gen_index_start,
        'inputs_last_generation'  : inputs_last_generation_all,
        'params_processes'        : params_processes,
    }
    return pars_resume

# SIMULATION REPLAY
def get_parameters_for_optimization_replay(
    results_path: str,
    folder_name : str,
    process_ind : int,
    generation  : int,
    model_name  : str = None,
    new_pars_prc: dict = None,
):
    ''' Loads parameters to setup the continuation of an optimization '''

    if new_pars_prc is None:
        new_pars_prc = {}

    # Get optimization data
    opt_data = load_all_optimization_data(
        results_path = results_path,
        folder_name  = folder_name,
        model_name   = model_name,
    )

    # Get inputs of the generation
    inputs_generation = opt_data['results_all'][process_ind][generation]['inputs']

    # Get neural simulation parameters
    params_process = snn_simulation_data.load_parameters_process(
        folder_name  = folder_name,
        tag_process  = process_ind,
        results_path = results_path,
    )[0]

    params_process = params_process | new_pars_prc

    # Collect parameters to resume optimization
    pars_replay = opt_data | {
        'tag_process'        : process_ind,
        'inputs_generation'  : inputs_generation,
        'params_process'     : params_process,
    }

    return pars_replay

def run_individual_from_generation(
    control_type    : str,
    pars_replay     : dict[str, Any],
    tag_run         : str,
    individual_ind  : int,
    load_connecivity: bool = True,
    new_pars_run    : dict = None,
    problem_class   : object = snn_optimization_problem.OptimizationPropblem,
) -> dict[str, Union[float, np.ndarray[float]]]:
    ''' Run with the parameters of a specified individual '''

    if new_pars_run is None:
        new_pars_run = {}

    if individual_ind.is_integer():
        individual_ind = int(individual_ind)
    else:
        raise ValueError('Individual index must be an integer')

    # Setup network
    (
        snn_sim,
        mech_sim_options,
        motor_output_signal_func,
    ) = snn_simulation_replay.replicate_network_setup(
        control_type = control_type,
        modname      = pars_replay['modname'],
        parsname     = pars_replay['parsname'],
        folder_name  = pars_replay['folder_name'],
        tag_folder   = pars_replay['tag_folder'],
        tag_process  = pars_replay['tag_process'],
        results_path = pars_replay['results_path'],
        run_params   = {'tag_run' : tag_run},
        load_conn    = load_connecivity,
        new_pars     = pars_replay['params_process'],
    )

    mech_sim_options         = new_pars_run.get('mech_sim_options', mech_sim_options)
    motor_output_signal_func = new_pars_run.get('motor_output_signal_func', motor_output_signal_func)

    # Define the optimization problem
    problem : snn_optimization_problem.OptimizationPropblem = problem_class(
        control_type             = control_type,
        n_sub_processes          = 1,
        net                      = snn_sim,
        vars_optimization        = pars_replay['pars_optimization'],
        obj_optimization         = pars_replay['obj_optimization'],
        constr_optimization      = pars_replay['constr_optimization'],
        mech_sim_options         = mech_sim_options,
        motor_output_signal_func = motor_output_signal_func,
    )

    # Simulation
    _success, metrics_run = problem._evaluate_single(
        individual_input = pars_replay['inputs_generation'][individual_ind],
        plot_figures     = True,
        save_prompt      = True,
        new_run_params   = new_pars_run,
    )

    return metrics_run

def run_best_individual(
    control_type     : str,
    results_path     : str,
    folder_name      : str,
    metric_key       : str,
    tag_run          : str,
    inds_processes   : list[int] = None,
    inds_generations : list[int] = None,
    model_name       : str  = None,
    new_pars_prc     : dict = None,
    new_pars_run     : dict = None,
    load_connectivity: bool = True,
    constr_additional: dict[str, list[float, float]] = None,
    problem_class    : object   = snn_optimization_problem.OptimizationPropblem,
) -> dict[str, Union[float, np.ndarray[float]]]:
    ''' Run the best individual of a generation for a given metric index '''

    # Get optimization parameters
    opt_data = load_all_optimization_data(
        results_path = results_path,
        folder_name  = folder_name,
        model_name   = model_name,
    )

    # Find best individuals across processes and generations
    (
        best_obj_evolution,
        _best_met_evolution,
        best_pos_evolution
    ) = get_best_evolution_across_generations(
        results_all         = opt_data['results_all'],
        metric_key          = metric_key,
        obj_optimization    = opt_data['obj_optimization_dict'],
        constr_optimization = opt_data['constr_optimization'],
        check_constraints   = True,
        constr_additional   = constr_additional,
        inds_processes      = inds_processes,
        inds_generations    = inds_generations,
    )

    # Minimum of the minima
    best_obj_evolution[np.isnan(best_obj_evolution)] = np.inf
    best_pos_evolution_index = np.unravel_index(best_obj_evolution.argmin(), best_obj_evolution.shape)

    best_pos_process    = best_pos_evolution_index[0]
    best_pos_generation = best_pos_evolution_index[1]
    best_pos_individual = best_pos_evolution[best_pos_evolution_index]

    # Get replay parameters for the best
    pars_replay = get_parameters_for_optimization_replay(
        results_path = results_path,
        folder_name  = folder_name,
        process_ind  = best_pos_process,
        generation   = best_pos_generation,
        model_name   = model_name,
        new_pars_prc = new_pars_prc,
    )

    # Run best individual
    metrics_run = run_individual_from_generation(
        control_type     = control_type,
        pars_replay      = pars_replay,
        tag_run          = tag_run,
        individual_ind   = best_pos_individual,
        load_connecivity = load_connectivity,
        new_pars_run     = new_pars_run,
        problem_class    = problem_class,
    )

    return metrics_run
