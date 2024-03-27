''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from network_experiments.snn_analysis_parser import parse_arguments
from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_utils import prepend_date_to_tag
from network_experiments.snn_simulation import simulate_multi_net_multi_run_closed_loop
import network_experiments.default_parameters.salamander.swimming.closed_loop.default as default
import ps_connections

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Default parameters
    default_params = default.get_default_parameters()

    # Script arguments
    args         = parse_arguments()
    results_path = args['results_path']

    # Parameters
    simulation_data_file_tag = 'feedback_topology_strength_swimming_4dim_in_up_ex_dw'

    # Process parameters
    max_ps_range = 0.025

    params_processes_shared = default_params['params_process'] | {
        'simulation_data_file_tag': simulation_data_file_tag,
        'ps_gain_axial'           : default.get_scaled_ps_gains(0.5),
        'connectivity_axial_newpars' : {
            'ps2ax': [
                ps_connections.get_ps_to_ax_in(amp= 0.5, max_ps_range = max_ps_range),
                ps_connections.get_ps_to_mn_ex(amp= 0.5, max_ps_range = max_ps_range),
                ps_connections.get_ps_to_ax_ex(amp= 0.5, max_ps_range = max_ps_range),
                ps_connections.get_ps_to_mn_in(amp= 0.5, max_ps_range = max_ps_range),
            ]
        }
    }

    params_processes_variable = [
        {
        }
    ]

    # Params runs
    params_runs = (
        [
            {
                'sig_ex_w'  : sig_w,
                'sig_ex_up' : 0,
                'sig_ex_dw' : sig_l,
                'sig_in_w'  : sig_w,
                'sig_in_up' : sig_l,
                'sig_in_dw' : 0,
            }
            for sig_w  in np.linspace(0.0, 5, 21)
            for sig_l  in np.linspace(0.0, max_ps_range, 11)
        ]
    )

    # Get parameters of all processes
    n_network_replicas = 20

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
        simulate_multi_net_multi_run_closed_loop(
            modname             = f'{CURRENTDIR}/net_farms_limbless_feedback_topology.py',
            parsname            = default_params['parsname'],
            params_processes    = params_processes_batch,
            params_runs         = params_runs,
            tag_folder          = tag_folder,
            results_path        = results_path,
            delete_connectivity = True,
        )


if __name__ == '__main__':
    main()
