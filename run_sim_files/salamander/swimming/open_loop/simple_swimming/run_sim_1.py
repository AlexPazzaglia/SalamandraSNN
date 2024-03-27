''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from network_experiments.snn_simulation import simulate_single_net_multi_run_open_loop_build
import network_experiments.default_parameters.salamander.swimming.open_loop.default as default

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = '/data/pazzagli/simulation_results'
    simulation_data_file_tag = 'swimming'

    # Default parameters
    default_params = default.get_default_parameters()

    # Process parameters
    params_process = default_params['params_process'] | {

        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': True,
    }

    # Params runs
    params_runs = [
        {
            # NOTE: AX_fraction = 0.8280493

            # Slow swimming
            # "neur_freq_ax": 2.09958,
            # "neur_wave_number_a": 1.2553600371378852 (1.0395)

            # 'stim_a_off' : 0.0,

            # Regular swimming
            # "neur_freq_ax": 3.09938,
            # "neur_wave_number_a": 1.5311890246148387 (1.2679)

            # 'stim_a_off' : 0.7,

            # Fast swimming
            # "neur_freq_ax": 3.6992599999999998,
            # "neur_wave_number_a": 1.4209298890778603 (1.1766)

            # 'stim_a_off' : 1.2,

            # High phase lag
            # "neur_freq_ax": 3.0106477500000004
            # "neur_wave_number_a": 2.213334399292409 (1.83275)

            # 'stim_a_off' : np.linspace(0.7, 0.0, 6),

            # Hip discontinuity
            # "neur_freq_ax": 2.09958,
            # "neur_wave_number_a": 0.8749479046718593

            # 'stim_a_off' : np.linspace(0.0, 0.3, 6) - 0.0,

            # Tail only
            # 'stim_a_off' : np.array([-3.0, -3.0, -3.0, 0.7, 0.7, 0.7,]),

            # Trunk only
            # 'stim_a_off' : np.array([0.7, 0.7, 0.7, -3.0, -3.0, -3.0,]),

            # Decoupled trunk - tail
            'stim_a_off' : np.array([0.0, 0.0, 0.0, 1.2, 1.2, 1.2,]),

        }
    ]

    # Simulate
    simulate_single_net_multi_run_open_loop_build(
        modname             = default_params['modname'],
        parsname            = default_params['parsname'],
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = True,
        results_path        = results_path,
        delete_files        = True,
        delete_connectivity = False,
        save_prompt         = True,
    )


if __name__ == '__main__':
    main()
