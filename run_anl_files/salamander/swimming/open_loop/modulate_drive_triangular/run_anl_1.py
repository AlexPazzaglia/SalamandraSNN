''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_utils import prepend_date_to_tag
from network_experiments.snn_simulation import simulate_multi_net_multi_run_open_loop
from network_experiments.snn_analysis_parser import parse_arguments
import network_experiments.default_parameters.salamander.swimming.open_loop.default as default

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Default parameters
    default_params = default.get_default_parameters()

    # Shared by processes
    args = parse_arguments()

    results_path = args['results_path']

    simulation_data_file_tag = 'drive_effect_triangular'

    # Process parameters
    params_processes_shared = default_params['params_process'] | {
        'simulation_data_file_tag': simulation_data_file_tag,
    }

    params_processes_variable = [
        {
        }
    ]

    # Params runs
    params_runs = [
        {
            'stim_a_off'   : stim_a_off + np.linspace(stim_h_off, 0, 6),
        }
        for stim_a_off in np.linspace(-2, 3, 51)
        for stim_h_off in np.linspace(-1.5, 1.5, 31)
    ]

    # Get parameters of all processes
    n_network_replicas = 30

    (
        params_processes,
        params_processes_batches
    ) = get_params_processes(
        params_processes_shared,
        params_processes_variable,
        n_network_replicas,
        np_random_seed    = args['np_random_seed'],
        start_index       = args['index_start'],
        finish_index      = args['index_finish'],
        n_processes_batch = args['n_processes_batch'],
    )

    # Simulate
    tag_folder = prepend_date_to_tag('ANALYSIS')
    for params_processes_batch in params_processes_batches:
        simulate_multi_net_multi_run_open_loop(
            modname             = default_params['modname'],
            parsname            = default_params['parsname'],
            params_processes    = params_processes_batch,
            params_runs         = params_runs,
            tag_folder          = tag_folder,
            results_path        = results_path,
            delete_connectivity = True,
        )


if __name__ == '__main__':
    main()
