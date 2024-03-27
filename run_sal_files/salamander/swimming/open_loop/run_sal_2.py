''' Script to run multiple sensitivity analysis with the desired module '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_utils import prepend_date_to_tag
from network_experiments.snn_sensitivity_analysis import sensitivity_analysis_open_loop_run

from network_experiments.snn_sensitivity_analysis_parser import parse_arguments

from run_sal_files.salamander.swimming.open_loop.sal_scalings_assigner import get_sal_run_params
import network_experiments.default_parameters.salamander.swimming.open_loop.default as default

def main():
    ''' Run sensitivity analysis '''

    # Default parameters
    default_params = default.get_default_parameters()

    # Shared by processes
    args = parse_arguments()

    results_path             = args['results_path']
    simulation_data_file_tag = 'swimming_cpg_neural_parameters_to_trefr'

    # Process parameters
    params_processes_shared = default_params['params_process'] | {
        'simulation_data_file_tag': simulation_data_file_tag,
    }

    params_processes_variable = [
        {
        }
    ]

    # Get parameters of all processes
    params_processes, params_processes_batches = get_params_processes(
        params_processes_shared   = params_processes_shared,
        params_processes_variable = params_processes_variable,
        n_processes_copies        = args['n_processes'],
        np_random_seed            = args['np_random_seed'],
        start_index               = args['index_start'],
        finish_index              = args['index_finish'],
        n_processes_batch         = args['n_processes_batch'],
    )

    # Define sensitivity analysis problem
    n_saltelli = args['n_saltelli']
    var_range  = [0.95, 1.05]

    sal_variables_keys = [
        'tau_memb',
        'R_memb',
        'tau1',
        'tau2',
        'delta_w1',
        'delta_w2',
        'w_ampa',
        'w_nmda',
        'w_glyc',
    ]

    sal_problem = {
        'names'   : sal_variables_keys,
        'num_vars': len(sal_variables_keys),
        'bounds'  : [ var_range for _ in sal_variables_keys ],
    }

    # Simulate
    tag_folder = prepend_date_to_tag('SENSITIVITY')
    for params_processes_batch in params_processes_batches:
        sensitivity_analysis_open_loop_run(
            modname          = default_params['modname'],
            parsname         = default_params['parsname'],
            n_saltelli       = n_saltelli,
            sal_problem      = sal_problem,
            params_processes = params_processes_batch,
            tag_folder       = tag_folder,
            results_path     = results_path,
            run_params_func  = get_sal_run_params,
        )

if __name__ == '__main__':
    main()
