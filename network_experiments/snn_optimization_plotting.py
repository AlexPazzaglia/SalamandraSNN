''' Analyze results of an optimization '''

from typing import Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from network_experiments import snn_optimization_results as sor

# PLOTTING PARAMETERS
SMALL_SIZE  = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 17

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.rc(  'font', size      = SMALL_SIZE )  # controls default text sizes
plt.rc(  'axes', titlesize = SMALL_SIZE )  # fontsize of the axes title
plt.rc(  'axes', labelsize = BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc( 'xtick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc( 'ytick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize  = SMALL_SIZE )  # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title

# Distribution vs Generation
def plot_metrics_distribution_from_generation(
    results_gen        : sor.RESULT_GEN,
    metric_key_1       : str,
    metric_key_2       : str,
    obj_optimization   : dict[str, dict[str, Union[str,float]]],
    constr_optimization: dict[str, list[float, float]],
    constr_additional  : dict[str, list[float, float]] = None,
    check_constraints  : bool = True,
    figname            : str = None,
    color              : str = '0',
):
    ''' Plot the distribition of two metrics '''

    # Get metrics distributions
    metrics_results_1 = sor.get_quantity_distribution_from_generation(
        results_gen         = results_gen,
        metric_key          = metric_key_1,
        obj_optimization    = obj_optimization,
        constr_optimization = constr_optimization,
        check_constraints   = check_constraints,
        constr_additional   = constr_additional,
    )
    metrics_results_2 = sor.get_quantity_distribution_from_generation(
        results_gen         = results_gen,
        metric_key          = metric_key_2,
        obj_optimization    = obj_optimization,
        constr_optimization = constr_optimization,
        check_constraints   = check_constraints,
        constr_additional   = constr_additional,
    )

    # Plot the metrics
    figname = figname if figname is not None else f'{metric_key_1}-{metric_key_2}'

    plt.figure(figname)
    plt.scatter(
        metrics_results_1,
        metrics_results_2,
        c = color,
        s = 0.6,
    )

    # Check if metrics are objectives
    is_obj_1 = metric_key_1 in obj_optimization.keys()
    is_obj_2 = metric_key_2 in obj_optimization.keys()

    if is_obj_1:
        plt.axvline(
            obj_optimization[metric_key_1]['target'],
            color = 'k',
            ls    = '--',
            label = f'Target {metric_key_1}'
        )
    if is_obj_2:
        plt.axhline(
            obj_optimization[metric_key_2]['target'],
            color = 'b',
            ls    = '-.',
            label = f'Target {metric_key_2}'
        )

    # Decorate
    plt.grid(True)
    plt.xlabel(metric_key_1)
    plt.ylabel(metric_key_2)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    return

def plot_metrics_distribution_across_generations(
    results_proc       : sor.RESULT_PRC,
    generations        : Union[int, list[int]],
    metric_key_1       : str,
    metric_key_2       : str,
    obj_optimization   : dict[str, dict[str, Union[str,float]]],
    constr_optimization: dict[str, list[float, float]],
    constr_additional  : dict[str, list[float, float]] = None,
    check_constraints  : bool = True,
    figname            : str = None,
    range_1            : tuple[float, float] = None,
    range_2            : tuple[float, float] = None,
):
    ''' Plot the distribution of two metrics across the generations '''
    fig  = plt.figure(figname)

    generations   = range(generations) if isinstance(generations, int) else generations
    n_generations = len(generations)
    for gen_ind, generation in enumerate(generations):
        plot_metrics_distribution_from_generation(
            results_gen         = results_proc[generation],
            metric_key_1        = metric_key_1,
            metric_key_2        = metric_key_2,
            obj_optimization    = obj_optimization,
            constr_optimization = constr_optimization,
            constr_additional   = constr_additional,
            check_constraints   = check_constraints,
            figname             = figname,
            color               = str(0.8 - 0.8*gen_ind/ max(generations)),
        )

    # Set the ranges
    if range_1  is not None:
        plt.xlim(range_1)
    if range_2  is not None:
        plt.ylim(range_2)

    # Axis and grid
    axis = fig.gca()
    axis.grid(True)

    # Invert the axis if needed
    is_obj_1 = metric_key_1 in obj_optimization.keys()
    is_obj_2 = metric_key_2 in obj_optimization.keys()
    sign_1   = obj_optimization[metric_key_1]['sign'] if is_obj_1 else +1
    sign_2   = obj_optimization[metric_key_2]['sign'] if is_obj_2 else +1

    if sign_1 == -1:
        axis.invert_xaxis()
    if sign_2 == -1:
        axis.invert_yaxis()

    # Draw colorbar
    if len(fig.axes) == 1:
        gen_min = min(generations)
        gen_max = max(generations)
        gen_range = gen_max - gen_min
        ticks   = (
            gen_min + np.linspace(0, gen_range + 1, 5, dtype= int)
            if n_generations >= 4 else generations
        )
        cmap    = plt.get_cmap('Greys',n_generations)
        norm    = mpl.colors.Normalize(
            vmin = gen_min,
            vmax = gen_max,
        )
        scalmap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        scalmap.set_array([])
        plt.colorbar(
            scalmap,
            ticks      = ticks,
            boundaries = np.arange(-0.1 + gen_min, gen_max + 0.1, 0.1),
            label      = 'Generation [#]',
        )

    plt.tight_layout()
    return

def plot_metrics_distribution_last_generation(
    results_proc       : sor.RESULT_PRC,
    generations        : int,
    metric_key_1       : str,
    metric_key_2       : str,
    obj_optimization   : dict[str, dict[str, Union[str,float]]],
    constr_optimization: dict[str, list[float, float]],
    constr_additional  : dict[str, list[float, float]] = None,
    check_constraints  : bool = True,
    figname            : str = None,
    range_1            : tuple[float, float] = None,
    range_2            : tuple[float, float] = None,
):
    ''' Plot the distribution of two metrics for the last generation of the optimization '''
    fig  = plt.figure(figname)

    plot_metrics_distribution_from_generation(
        results_gen         = results_proc[generations-1],
        metric_key_1        = metric_key_1,
        metric_key_2        = metric_key_2,
        obj_optimization    = obj_optimization,
        constr_optimization = constr_optimization,
        constr_additional   = constr_additional,
        check_constraints   = check_constraints,
        figname             = figname,
        color               = 'k',
    )

    # Set the ranges
    if range_1  is not None:
        plt.xlim(range_1)
    if range_2  is not None:
        plt.ylim(range_2)

    # Axis and grid
    axis = fig.gca()
    axis.grid(True)

    # Invert the axis if needed
    is_obj_1 = metric_key_1 in obj_optimization.keys()
    is_obj_2 = metric_key_2 in obj_optimization.keys()
    sign_1   = obj_optimization[metric_key_1]['sign'] if is_obj_1 else +1
    sign_2   = obj_optimization[metric_key_2]['sign'] if is_obj_2 else +1

    if sign_1 == -1:
        axis.invert_xaxis()
    if sign_2 == -1:
        axis.invert_yaxis()

    plt.tight_layout()
    return

# Distribution vs Quantity
def plot_metrics_distribution_vs_quantitiy_from_generation(
    results_gen        : sor.RESULT_GEN,
    metric_key_1       : str,
    metric_key_2       : str,
    metric_key_3       : str,
    obj_optimization   : dict[str, dict[str, Union[str,float]]],
    constr_optimization: dict[str, list[float, float]],
    constr_additional  : dict[str, list[float, float]] = None,
    check_constraints  : bool = True,
    figname            : str = None,
    range_3            : tuple[float, float] = None,
):
    ''' Plot the distribition of two metrics '''

    # Get the metric distributions
    metrics_results_1 = sor.get_quantity_distribution_from_generation(
        results_gen         = results_gen,
        metric_key          = metric_key_1,
        obj_optimization    = obj_optimization,
        constr_optimization = constr_optimization,
        constr_additional   = constr_additional,
        check_constraints   = check_constraints,
    )
    metrics_results_2 = sor.get_quantity_distribution_from_generation(
        results_gen         = results_gen,
        metric_key          = metric_key_2,
        obj_optimization    = obj_optimization,
        constr_optimization = constr_optimization,
        constr_additional   = constr_additional,
        check_constraints   = check_constraints,
    )
    metrics_results_3 = sor.get_quantity_distribution_from_generation(
        results_gen         = results_gen,
        metric_key          = metric_key_3,
        obj_optimization    = obj_optimization,
        constr_optimization = constr_optimization,
        constr_additional   = constr_additional,
        check_constraints   = check_constraints,
    )

    if (
        np.all(np.isnan(metrics_results_1)) or
        np.all(np.isnan(metrics_results_2)) or
        np.all(np.isnan(metrics_results_3))
    ):
        return

    # Get the range
    if range_3 is not None:
        metrics_results_3 = np.clip(metrics_results_3, *range_3)

    # Plot the distribution
    figname = figname if figname is not None else f'{metric_key_1}-{metric_key_2}-{metric_key_3}'

    plt.figure(figname)
    plt.scatter(
        metrics_results_1,
        metrics_results_2,
        c    = metrics_results_3,
        s    = 0.5,
        vmin = range_3[0] if range_3 is not None else None,
        vmax = range_3[1] if range_3 is not None else None,
    )

    plt.grid(True)
    plt.xlabel(metric_key_1)
    plt.ylabel(metric_key_2)

    plt.tight_layout()
    return

def plot_metrics_distribution_vs_quantity_across_generations(
    results_proc       : sor.RESULT_PRC,
    generations        : Union[int, list[int]],
    metric_key_1       : str,
    metric_key_2       : str,
    metric_key_3       : str,
    obj_optimization   : dict[str, dict[str, Union[str,float]]],
    constr_optimization: dict[str, list[float, float]],
    constr_additional  : dict[str, list[float, float]] = None,
    check_constraints  : bool = True,
    figname            : str = None,
    range_1            : tuple[float, float] = None,
    range_2            : tuple[float, float] = None,
    range_3            : tuple[float, float] = None,
):
    ''' Plot the distribution of two metrics acriss generations of the optimization '''

    fig  = plt.figure(figname)

    # Get the range
    if range_3 is None:
        range_3 = sor.get_quantity_range_across_generations(
            results_proc        = results_proc,
            generations         = generations,
            metric_key          = metric_key_3,
            obj_optimization    = obj_optimization,
            constr_optimization = constr_optimization,
            constr_additional   = constr_additional,
            check_constraints   = check_constraints,
        )

    # Plot the distribution
    generations = range(generations) if isinstance(generations, int) else generations
    for generation in generations:
        plot_metrics_distribution_vs_quantitiy_from_generation(
            results_gen         = results_proc[generation],
            metric_key_1        = metric_key_1,
            metric_key_2        = metric_key_2,
            metric_key_3        = metric_key_3,
            obj_optimization    = obj_optimization,
            constr_optimization = constr_optimization,
            constr_additional   = constr_additional,
            check_constraints   = check_constraints,
            figname             = figname,
            range_3             = range_3,
        )

    # Set the ranges
    if range_1  is not None:
        plt.xlim(range_1)
    if range_2  is not None:
        plt.ylim(range_2)

    # Axis and grid
    axis = fig.gca()
    axis.grid(True)

    # Invert the axis if needed
    is_obj_1 = metric_key_1 in obj_optimization.keys()
    is_obj_2 = metric_key_2 in obj_optimization.keys()
    sign_1   = obj_optimization[metric_key_1]['sign'] if is_obj_1 else +1
    sign_2   = obj_optimization[metric_key_2]['sign'] if is_obj_2 else +1

    if sign_1 == -1:
        axis.invert_xaxis()
    if sign_2 == -1:
        axis.invert_yaxis()

    # Draw colorbar
    if len(fig.axes) == 1:
        # Draw colorbar
        cbar = plt.colorbar()
        cbar.set_label(metric_key_3)
        if range_3 is not None:
            plt.clim(*range_3)

    plt.tight_layout()
    return

# Best vs Generation
def plot_metrics_evolution_across_generations(
    evolution_best : np.ndarray,
    evolution_stats: list[list[dict[str, float]]],
    metric_label   : str,
    figname        : str = None,
    yrange         : tuple[float, float] = None,
    axis           : plt.Axes = None,
    **kwargs
):
    ''' Plot the evolution of the best metric across optimization evolutions '''
    figname = figname if figname is not None else metric_label
    plt.figure(figname)

    if axis is None:
        axis = plt.axes()

    # Mean and std of best evolution
    best_evolution_mean = np.mean(evolution_best, axis=0)
    best_evolution_std  = np.std(evolution_best, axis=0)

    generations = np.arange(len(best_evolution_mean))

    # Mean of best statistics
    statistics_names = kwargs.get('statistics_names', [])
    statistics_names = list( set( ['mean', 'std'] + statistics_names ) )

    average_statistics_evolution = {
        statistics: np.mean(
            [
                [
                    gen_statistics[statistics]
                    for gen_statistics in evolution_statistics_process
                ]
                for evolution_statistics_process in evolution_stats
            ],
            axis = 0,
        )
        for statistics in statistics_names
    }

    # Only plot first N generations
    max_gen             = kwargs.get('max_gen', evolution_best.shape[1])
    best_evolution_mean = best_evolution_mean[:max_gen]
    best_evolution_std  = best_evolution_std[:max_gen]
    generations         = generations[:max_gen]

    # Evolution of the best
    axis.plot(
        generations,
        best_evolution_mean,
        label = f'BEST',
        color = 'red',
    )

    axis.fill_between(
        generations,
        (best_evolution_mean - best_evolution_std),
        (best_evolution_mean + best_evolution_std),
        edgecolor   = 'salmon',
        facecolor   = 'mistyrose',
        interpolate = True
    )

    # Evolution of the statistics
    axis.plot(
        generations,
        average_statistics_evolution['mean'],
        label =  f'MEAN',
        color = '#1B2ACC'
    )

    axis.fill_between(
        generations,
        (average_statistics_evolution['mean'] - average_statistics_evolution['std']),
        (average_statistics_evolution['mean'] + average_statistics_evolution['std']),
        edgecolor   = '#1B2ACC',
        facecolor   = '#089FFF',
        interpolate = True
    )

    # Check if target is provided
    target = kwargs.get('target', None)
    if target is not None:
        axis.axhline(
            target,
            color = 'k',
            ls    = '--',
            label = 'TARGET',
        )

    # Decorate
    if yrange is not None:
        axis.set_ylim(yrange)
    axis.set_xlim(generations[0], generations[-1] )
    axis.set_xlabel('Generation [#]')
    axis.set_ylabel(metric_label)
    axis.grid(True)
    axis.legend()

    plt.tight_layout()
    return axis

# DATA LOADING AND PLOTTING
def optimization_post_processing(
    folder_name   : str,
    results_path  : str,
    processes_inds: list[int] = None,
    **kwargs
):
    ''' Post processing for an optimization job '''

    sim_name          = folder_name.replace('/', '_')
    constr_additional = kwargs.get('constr_additional', None)
    check_constraints = kwargs.get('check_constraints', True)

    # Get optimization results and parameters
    (
        results_all,
        _params_names,
        metrics_names,
    ) = sor.load_optimization_results(
        folder_name,
        results_path    = results_path,
        processess_inds = processes_inds,
    )

    (
        _vars_optimization,
        obj_optimization,
        constr_optimization,
    ) = sor.load_optimization_parameters(
        folder_name,
        results_path
    )

    # Optimization objectives
    obj_optimization_names    = [ obj_pars[0] for obj_pars in obj_optimization ]
    obj_optimization_expanded = sor.get_expanded_optimization_objectives(obj_optimization)

    ### ACROSS PROCESSES
    evolution_args = {
        'results_all'         : results_all,
        'metric_key'          : None,
        'obj_optimization'    : obj_optimization_expanded,
        'constr_optimization' : constr_optimization,
        'constr_additional'   : constr_additional,
        'check_constraints'   : check_constraints,
    }

    # SPEED evolution
    if 'speed_fwd' in obj_optimization_names:
        evolution_args['metric_key'] = 'speed_fwd'

        figname_speed = kwargs.get('figname_speed')
        axis_speed    = kwargs.get('axis_speed')
        range_speed   = kwargs.get('range_speed')

        (
            _best_speed_obj,
            best_speed_met,
            _best_speed_pos
        ) = sor.get_best_evolution_across_generations(**evolution_args)
        speed_statistics  = sor.get_statistics_across_generations(**evolution_args)

        if figname_speed is None:
            figname_speed = f'SPEED - {sim_name}'

        plot_metrics_evolution_across_generations(
            evolution_best       = best_speed_met,
            evolution_stats = speed_statistics,
            metric_label         = 'SPEED [m/s]',
            figname              = figname_speed,
            axis                 = axis_speed,
            yrange               = range_speed,
        )

    # COT evolution
    if 'cot' in obj_optimization_names:
        evolution_args['metric_key'] = 'cot'

        figname_cot = kwargs.get('figname_cot')
        axis_cot    = kwargs.get('axis_cot')
        range_cot   = kwargs.get('range_cot')

        (
            _best_cot_obj,
            best_cot_met,
            _best_cot_pos
        ) = sor.get_best_evolution_across_generations(**evolution_args)
        cot_statistics = sor.get_statistics_across_generations(**evolution_args)

        if figname_cot is None:
            figname_cot = f'COT - {sim_name}'

        plot_metrics_evolution_across_generations(
            evolution_best       = best_cot_met,
            evolution_stats = cot_statistics,
            metric_label         = 'COT [J/m]',
            figname              = figname_cot,
            axis                 = axis_cot,
            yrange               = range_cot,
        )


    ### PROCESS-SPECIFIC
    n_gen = kwargs.get(
        'n_gen',
        min( [ len( res_prc ) for res_prc in results_all ] ),
    )

    for results_process in results_all:

        # Metrics distriution vs generation
        figname_gen_distribution_all = kwargs.get('figname_gen_distribution_all')
        if figname_gen_distribution_all is None:
            figname_gen_distribution_all = f'SPEED-COT - All - {sim_name}'

        plot_metrics_distribution_across_generations(
            results_proc        = results_process,
            generations         = n_gen,
            metric_key_1        = 'speed_fwd',
            metric_key_2        = 'cot',
            obj_optimization    = obj_optimization_expanded,
            constr_optimization = constr_optimization,
            constr_additional   = constr_additional,
            check_constraints   = True,
            figname             = figname_gen_distribution_all,
            range_1             = range_speed,
            range_2             = range_cot,
        )

        figname_gen_distribution_last = kwargs.get('figname_gen_distribution_last')
        if figname_gen_distribution_last is None:
            figname_gen_distribution_last = f'SPEED-COT - Last - {sim_name}'

        plot_metrics_distribution_last_generation(
            results_proc        = results_process,
            generations         = n_gen,
            metric_key_1        = 'speed_fwd',
            metric_key_2        = 'cot',
            obj_optimization    = obj_optimization_expanded,
            constr_optimization = constr_optimization,
            constr_additional   = constr_additional,
            check_constraints   = True,
            figname             = figname_gen_distribution_last,
            range_1             = range_speed,
            range_2             = range_cot,
        )

        # Metrics distriution vs quantity (NOTE: for testing use 'speed_fwd' and 'cot')
        quantities = [
            'ptcc_ax',
            'freq_ax',
            'ipl_ax_a',
            'energy',
        ]
        for quantity_key in quantities:
            figname_distribution_all = kwargs.get(f'figname_{quantity_key}_distribution_all')
            if figname_distribution_all is None:
                figname_distribution_all = f'SPEED-COT-{quantity_key.upper()} - All - {sim_name}'

            plot_metrics_distribution_vs_quantity_across_generations(
                results_proc        = results_process,
                generations         = n_gen,
                metric_key_1        = 'speed_fwd',
                metric_key_2        = 'cot',
                metric_key_3        = quantity_key,
                obj_optimization    = obj_optimization_expanded,
                constr_optimization = constr_optimization,
                constr_additional   = constr_additional,
                check_constraints   = True,
                figname             = figname_distribution_all,
                range_1             = range_speed,
                range_2             = range_cot,
                range_3             = None,
            )

    return

