''' Replay previous simulation from an optimization process'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from network_experiments import snn_simulation_setup, snn_optimization_results
from run_opt_files.salamander.swimming.position_control.drag_coefficients.opt_problem import SalamanderOptimizationProblem

def main():
    '''  Replay previous simulation from an optimization process '''

    results_path = '/data/pazzagli/simulation_results/data/optimization/swimming/drag_coefficients'

    optimization_paths = [
        [ results_path, 'net_farms_4limb_1dof_unweighted_position_control_speedbl_071_tailbl_016_OPTIMIZATION' ]
    ]

    constr_additional = {
        'mech_tail_beat_amp_bl': (0.14, 0.18),
    }

    metric_keys = [
        'mech_tail_beat_amp_bl',
    ]

    new_pars_prc = {}

    new_pars_run = {
        'mech_sim_options' : snn_simulation_setup.get_mech_sim_options(video= True),
    }

    # Model with lowest COT
    for results_path, folder_name in optimization_paths:
        for metric_key in metric_keys:
            snn_optimization_results.run_best_individual(
                control_type      = 'position_control',
                results_path      = results_path,
                folder_name       = folder_name,
                inds_processes    = None,
                inds_generations  = None,
                metric_key        = metric_key,
                tag_run           = f'best_{metric_key}',
                constr_additional = constr_additional,
                load_connectivity = True,
                new_pars_prc      = new_pars_prc,
                new_pars_run      = new_pars_run,
                problem_class     = SalamanderOptimizationProblem,
            )

    return

if __name__ == '__main__':
    main()
