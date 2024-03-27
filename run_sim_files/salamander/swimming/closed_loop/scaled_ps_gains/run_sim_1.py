''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_closed_loop_build
import network_experiments.default_parameters.salamander.swimming.closed_loop.default as default

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = '/data/pazzagli/simulation_results'
    simulation_data_file_tag = 'scaled_ps_gains'

    # Default parameters
    default_params = default.get_default_parameters()

    # Get scaled proprioceptive feedback gains
    scaled_ps_gains = default.get_scaled_ps_gains(
        alpha_fraction_th   = 2.5,
        reference_data_name = default_params['reference_data_name'],
    )

    # Process parameters
    params_process = default_params['params_process'] | {

        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': True,

        'stim_a_off'   : 0.0,
        'ps_gain_axial': scaled_ps_gains,
    }

    # Params runs
    params_runs = [
        {
        }
    ]

    # Simulate
    simulate_single_net_multi_run_closed_loop_build(
        modname             = default_params['modname'],
        parsname            = default_params['parsname'],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = False,
        plot_figures        = True,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
    )


if __name__ == '__main__':
    main()
