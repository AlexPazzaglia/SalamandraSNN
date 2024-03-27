''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_modules.parameters.network_parameters import SnnParameters
from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_closed_loop_build
import network_experiments.default_parameters.salamander.swimming.closed_loop.default as default

import numpy as np

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = '/data/pazzagli/simulation_results'
    simulation_data_file_tag = 'scaled_ps_weight'

    # Default parameters
    default_params = default.get_default_parameters()

    # Process parameters
    params_process = default_params['params_process'] | {

        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': True,

        'ps_gain_axial': default.get_scaled_ps_gains(
            alpha_fraction_th   = 0.1,
            reference_data_name = default_params['reference_data_name'],
        ),

        'noise_term' : True,
    }

    # Params runs
    default_snn_pars = SnnParameters(
        parsname     = default_params['parsname'],
        results_path = results_path,
    )

    indices_cpg = default_snn_pars.topology.network_modules['cpg']['axial'].indices

    params_runs = [
        {
            'stim_a_off'   : np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
            'ps_weight'    : 1.5,

            # NOISE
            'scalings' : {
                'neural_params' : [
                    {
                        'neuron_group_ind': 0,
                        'var_name'        : 'sigma',
                        'indices'         : indices_cpg,
                        'nominal_value'   : [1.0, 'mV'],
                        'scaling'         : 7.0,
                    },
                ],
            },
        }
    ]

    # Simulate
    simulate_single_net_multi_run_closed_loop_build(
        modname             = f'{CURRENTDIR}/net_farms_ps_weight_noisy.py',
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
