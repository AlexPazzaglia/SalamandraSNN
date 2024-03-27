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
from network_experiments.snn_simulation import simulate_single_net_multi_run_signal_driven_build
import network_experiments.default_parameters.salamander.swimming.signal_driven.default as default

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Parameters
    results_path   = '/data/pazzagli/simulation_results'
    default_params = default.get_default_parameters()

    simulation_data_file_tag = 'frequency_vs_wavelength'

    # Process parameters
    params_process = default_params['params_process'] | {
        'simulation_data_file_tag': simulation_data_file_tag,
    }
    params_process['mech_sim_options']['save_all_metrics_data'] = True

    # Get params
    params_runs = [
        {
            'motor_output_signal_pars' : default.get_gait_params(
                frequency    = frequency,
                wave_number  = wave_number,
                joints_amps  = default_params['joints_amps'],
                sig_function = lambda phase : np.tanh( 20 * np.cos( 2*np.pi * phase ) )
            )
        }
        for frequency in np.linspace(1.5, 5.0, 15)
        for wave_number in np.linspace(0.5, 1.5, 11)
    ]

    # Simulate
    metrics_runs = simulate_single_net_multi_run_signal_driven_build(
        modname             = default_params['modname'],
        parsname            = default_params['parsname'],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'ANALYSIS',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = False,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = False,
    )

if __name__ == '__main__':
    main()