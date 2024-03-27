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
    snn_simulation,
    snn_simulation_results,
    snn_optimization,
    snn_optimization_parser,
    snn_optimization_results,
)

def main():
    '''  Replay previous simulation from an optimization process '''

    # PARAMETERS
    new_args = [
        {
            'name'   : 'source_folder_name',
            'type'   : str,
            'default': None,
        },
        {
            'name'   : 'source_folder_path',
            'type'   : str,
            'default': None,
        },
    ]
    args = snn_optimization_parser.parse_arguments(new_args)

    # Parameters from previous optimization
    pars_optimization = snn_optimization_results.get_parameters_for_optimization_continuation(
        results_path = args['source_folder_path'],
        folder_name  = args['source_folder_name'],
    )

    # Run optimization
    snn_optimization.run_optimization_closed_loop(
        modname             = pars_optimization['modname'],
        parsname            = pars_optimization['parsname'],
        params_processes    = pars_optimization['params_processes'],
        tag_folder          = snn_utils.prepend_date_to_tag('OPTIMIZATION_CONTINUATION'),
        pars_optimization   = pars_optimization['pars_optimization'],
        obj_optimization    = pars_optimization['obj_optimization'],
        constr_optimization = pars_optimization['constr_optimization'],
        results_path        = args['results_path'],
        n_sub_processes     = args['n_sub_processes'],

        pop_inputs_processes = pars_optimization['inputs_last_generation'],
        n_gen                = args['n_gen'],
        gen_index_start      = pars_optimization['gen_index_start'],
    )

    return

if __name__ == '__main__':
    main()
