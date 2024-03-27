import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from network_experiments import (
    snn_optimization_results,
)

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
    ) = snn_optimization_results.load_optimization_results(
        folder_name,
        results_path    = results_path,
        processess_inds = processes_inds,
    )

    (
        _vars_optimization,
        obj_optimization,
        constr_optimization,
    ) = snn_optimization_results.load_optimization_parameters(
        folder_name,
        results_path
    )

    # Optimization objectives
    obj_optimization_names    = [ obj_pars[0] for obj_pars in obj_optimization ]
    obj_optimization_expanded = snn_optimization_results.get_expanded_optimization_objectives(obj_optimization)

    ### ACROSS PROCESSES
    evolution_args = {
        'results_all'         : results_all,
        'metric_key'          : None,
        'obj_optimization'    : obj_optimization_expanded,
        'constr_optimization' : constr_optimization,
        'constr_additional'   : constr_additional,
        'check_constraints'   : check_constraints,
    }

    # SPEED_BL evolution
    evolution_args['metric_key'] = 'mech_speed_fwd_bl'

    figname_speed_bl = kwargs.get('figname_speed_bl')
    axis_speed_bl    = kwargs.get('axis_speed_bl')
    range_speed_bl   = kwargs.get('range_speed_bl')

    (
        _best_speed_bl_obj,
        best_speed_bl_met,
        _best_speed_bl_pos,
    ) =  snn_optimization_results.get_best_evolution_across_generations(**evolution_args)

    (
        ranking_obj,
        ranking_val,
        ranking_pos
    ) = snn_optimization_results.get_ranking_across_generations(**evolution_args)

    best_100_inputs = np.array(
        [
            snn_optimization_results.get_individual_input_from_generation(results_all[0][gen], ind)
            for gen, ind in ranking_pos[0][:100]
        ]
    )


    speed_bl_statistics  = snn_optimization_results.get_statistics_across_generations(**evolution_args)

    if figname_speed_bl is None:
        figname_speed_bl = f'SPEED_BL - {sim_name}'

    snn_optimization_results.plot_metrics_evolution_across_generations(
        evolution_best  = best_speed_bl_met,
        evolution_stats = speed_bl_statistics,
        metric_label    = 'SPEED_BL [BL/s]',
        figname         = figname_speed_bl,
        axis            = axis_speed_bl,
        yrange          = range_speed_bl,
        target          = obj_optimization_expanded['mech_speed_fwd_bl']['target']
    )

    # TAIL_BEAT_AMP_BL evolution
    evolution_args['metric_key'] = 'mech_tail_beat_amp_bl'

    figname_tail_beat_amp_bl = kwargs.get('figname_tail_beat_amp_bl')
    axis_tail_beat_amp_bl    = kwargs.get('axis_tail_beat_amp_bl')
    range_tail_beat_amp_bl   = kwargs.get('range_tail_beat_amp_bl')

    (
        _best_tail_beat_amp_bl_obj,
        best_tail_beat_amp_bl_met,
        _best_tail_beat_amp_bl_pos,
    ) = snn_optimization_results.get_best_evolution_across_generations(**evolution_args)
    tail_beat_amp_bl_statistics = snn_optimization_results.get_statistics_across_generations(**evolution_args)

    if figname_tail_beat_amp_bl is None:
        figname_tail_beat_amp_bl = f'TAIL_BEAT_AMP_BL - {sim_name}'

    snn_optimization_results.plot_metrics_evolution_across_generations(
        evolution_best  = best_tail_beat_amp_bl_met,
        evolution_stats = tail_beat_amp_bl_statistics,
        metric_label    = 'TAIL_BEAT_AMP_BL [BL]',
        figname         = figname_tail_beat_amp_bl,
        axis            = axis_tail_beat_amp_bl,
        yrange          = range_tail_beat_amp_bl,
        target          = obj_optimization_expanded['mech_tail_beat_amp_bl']['target']
    )

    ### PROCESS-SPECIFIC
    n_gen = kwargs.get(
        'n_gen',
        min( [ len( res_prc ) for res_prc in results_all ] ),
    )

    for results_process in results_all:

        # Metrics distriution vs generation
        snn_optimization_results.plot_metrics_distribution_across_generations(
            results_proc        = results_process,
            generations         = n_gen,
            metric_key_1        = 'mech_speed_fwd_bl',
            metric_key_2        = 'mech_tail_beat_amp_bl',
            obj_optimization    = obj_optimization_expanded,
            constr_optimization = constr_optimization,
            constr_additional   = constr_additional,
            check_constraints   = check_constraints,
            figname             = f'SPEED_FWD_BL vs TAIL_BEAT_AMP_BL - All - {sim_name}',
            range_1             = range_speed_bl,
        )


        # # Metrics distriution vs quantity (NOTE: for testing use 'speed_fwd' and 'cot')
        # quantities = [
        #     'ptcc_ax',
        #     'freq_ax',
        #     'ipl_ax_a',
        #     'energy',
        # ]
        # for quantity_key in quantities:
        #     figname_distribution_all = kwargs.get(f'figname_{quantity_key}_distribution_all')
        #     if figname_distribution_all is None:
        #         figname_distribution_all = f'SPEED-COT-{quantity_key.upper()} - All - {sim_name}'

        #     snn_optimization_results.plot_metrics_distribution_vs_quantity_across_generations(
        #         results_proc        = results_process,
        #         generations         = n_gen,
        #         metric_key_1        = 'speed_fwd',
        #         metric_key_2        = 'cot',
        #         metric_key_3        = quantity_key,
        #         obj_optimization    = obj_optimization_expanded,
        #         constr_optimization = constr_optimization,
        #         constr_additional   = constr_additional,
        #         check_constraints   = True,
        #         figname             = figname_distribution_all,
        #         range_1             = range_speed,
        #         range_2             = range_cot,
        #         range_3             = None,
        #     )

    return
