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
from network_experiments.snn_simulation import simulate_single_net_multi_run_closed_loop_build
import network_experiments.default_parameters.salamander.swimming.closed_loop.default as default

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = '/data/pazzagli/simulation_results'
    simulation_data_file_tag = 'asymmetric_drive'

    # Default parameters
    default_params = default.get_default_parameters()

    # Get scaled proprioceptive feedback gains
    scaled_ps_gains = default.get_scaled_ps_gains(
        alpha_fraction_th = np.array(
            [
                0.20,
                0.20,
                0.20,
                0.20,
                0.060,
                0.060,
                0.060,
                0.060,
            ]
        ),
        reference_data_name = default_params['reference_data_name'],
    )

    # Process parameters
    params_process = default_params['params_process'] | {
        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': False,
        'ps_gain_axial'            : scaled_ps_gains,
    }

    # Params runs
    params_runs = [
        {
            'stim_a_off'   : 0.0,
            'stim_lr_asym' : 0.2,
        }
    ]

    # Simulate
    simulate_single_net_multi_run_closed_loop_build(
        modname             = f'{CURRENTDIR}/net_farms_limbless_simplified.py',
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
