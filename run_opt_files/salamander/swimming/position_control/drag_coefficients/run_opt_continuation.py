''' Replay previous simulation from an optimization process'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments import (
    snn_utils,
    snn_optimization,
    snn_optimization_parser,
    snn_optimization_results,
)

from run_opt_files.swimming.drag_coefficients.salamander_opt_problem import SalamanderOptimizationProblem

def main():
    '''  Replay previous simulation from an optimization process '''

    args = snn_optimization_parser.parse_arguments()

    results_path       = args['results_path']

    source_folder_path = '/data/pazzagli/simulation_results/data/optimization/swimming/drag_coefficients'
    source_folder_name = 'net_farms_4limb_1dof_unweighted_position_control_speedbl_071_tailbl_016_OPTIMIZATION'

    # Parameters from previous optimization
    pars_optimization = snn_optimization_results.get_parameters_for_optimization_continuation(
        results_path = source_folder_path,
        folder_name  = source_folder_name,
    )

    # Run optimization
    snn_optimization.run_optimization_position_control(
        modname             = pars_optimization['modname'],
        parsname            = pars_optimization['parsname'],
        params_processes    = pars_optimization['params_processes'],
        tag_folder          = snn_utils.prepend_date_to_tag('OPTIMIZATION_CONTINUATION'),
        results_path        = results_path,
        pars_optimization   = pars_optimization['pars_optimization'],
        obj_optimization    = pars_optimization['obj_optimization'],
        constr_optimization = pars_optimization['constr_optimization'],
        n_sub_processes     = args['n_sub_processes'],
        problem_class       = SalamanderOptimizationProblem,

        pop_inputs_processes = pars_optimization['inputs_last_generation'],
        n_gen                = args['n_gen'],
        gen_index_start      = pars_optimization['gen_index_start'],
    )

    return

if __name__ == '__main__':
    main()
