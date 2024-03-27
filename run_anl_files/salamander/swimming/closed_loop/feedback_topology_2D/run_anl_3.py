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
    simulation_data_file_tag = 'feedback_topology_strength_swimming_2dim_no_mn_1'

    # Process parameters
    params_processes_shared = default_params['params_process'] | {
        'simulation_data_file_tag': simulation_data_file_tag,
        'ps_gain_axial'           : default.get_scaled_ps_gains(0.8),
    }

    params_processes_variable_ex = [
        {
            'connectivity_axial_newpars' : {
                'ps2ax': [

                    # INHIBITION
                    ps_connections.get_ps_to_ax_in(amp= 0.0, sigma_up=0.0, sigma_dw=0.0),
                    ps_connections.get_ps_to_mn_in(amp= 0.0, sigma_up=0.0, sigma_dw=0.0),

                    # EXCITATION
                    ps_connections.get_ps_to_ax_ex(amp= 0.5, sigma_up=ex_up, sigma_dw=ex_dw),
                    ps_connections.get_ps_to_mn_ex(amp= 0.0, sigma_up=0.0, sigma_dw=0.0),

                ]
            }
        }
        for ex_up in np.linspace(0, 5, 21)
        for ex_dw in np.linspace(0, 5, 21)
    ]

    params_processes_variable_in = [
        {
            'connectivity_axial_newpars' : {
                'ps2ax': [

                    # INHIBITION
                    ps_connections.get_ps_to_ax_in(amp= 0.5, sigma_up=in_up, sigma_dw=in_dw),
                    ps_connections.get_ps_to_mn_in(amp= 0.0, sigma_up=0.0, sigma_dw=0.0),

                    # EXCITATION
                    ps_connections.get_ps_to_ax_ex(amp= 0.0, sigma_up=0.0, sigma_dw=0.0),
                    ps_connections.get_ps_to_mn_ex(amp= 0.0, sigma_up=0.0, sigma_dw=0.0),
                ]
            }
        }

        for in_up in np.linspace(0, 5, 21)
        for in_dw in np.linspace(0, 5, 21)
    ]

    params_processes_variable = params_processes_variable_ex + params_processes_variable_in

    # Params runs
    params_runs = [
        {
        }
    ]

    # Get parameters of all processes
    # FIRST HALF OF THE SIMULATIONS
    n_network_replicas = 5
    process_stop_index = 2205

    (
        params_processes,
        params_processes_batches
    ) = get_params_processes(
        params_processes_shared,
        params_processes_variable,
        n_network_replicas,
        np_random_seed    = args['np_random_seed'],
        start_index       = args['index_start'],
        finish_index      = min(process_stop_index, args['index_finish']),
        n_processes_batch = args['n_processes_batch'],
    )

    # Simulate
    tag_folder = prepend_date_to_tag('ANALYSIS')
    for params_processes_batch in params_processes_batches:
        simulate_multi_net_multi_run_closed_loop(
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
