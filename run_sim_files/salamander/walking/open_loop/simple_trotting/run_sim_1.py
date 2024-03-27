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
from network_experiments.snn_simulation import simulate_single_net_multi_run_open_loop_build

import network_experiments.default_parameters.salamander.walking.open_loop.default as default


def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = '/data/pazzagli/simulation_results'
    simulation_data_file_tag = 'trotting'

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
            # Regular trotting
            # "neur_freq_ax": 0.7998399999999999

            # 'stim_l_off' : 0.0,

            # Fast trotting
            # "neur_freq_ax": 1.29974,

            'stim_l_off' : 0.5,

            # Very Fast trotting
            # "neur_freq_ax": 1.4997000000000003,

            # 'stim_l_off' : 1.0,

            # Fast trotting + Silent tail
            # "neur_freq_ax": 1.29974,

            # 'stim_l_off' : 0.5,
            # 'stim_a_off' : np.array([0.0, 0.0, 0.0, -3.0, -3.0, -3.0,]),

            # Trotting + Double bursting tail

            # 'stim_l_off' : 0.5,
            # 'stim_a_off' : np.array([0.0, 0.0, 0.0, 1.5, 1.5, 1.5,]),

            # Trotting + Travelling wave

            # 'stim_l_off' : -0.4,
            # 'stim_a_off' : np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0,]),

        }
    ]

    # Simulate
    simulate_single_net_multi_run_open_loop_build(
        modname             = default_params['modname'],
        parsname            = default_params['parsname'],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = True,
        results_path        = results_path,
        delete_files        = True,
        delete_connectivity = False,
    )


if __name__ == '__main__':
    main()
