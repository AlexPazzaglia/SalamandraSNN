''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments.snn_simulation import simulate_single_net_multi_run_open_loop_build
from network_modules.parameters.network_parameters import SnnParameters
import network_experiments.default_parameters.salamander.swimming.open_loop.default as default

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = '/data/pazzagli/simulation_results'
    simulation_data_file_tag = 'noisy_swimming'

    # Default parameters
    default_params = default.get_default_parameters()

    # Process parameters
    params_process = default_params['params_process'] | {

        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': True,

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
            'scalings' : {
                'neural_params' : [
                    {
                        'neuron_group_ind': 0,
                        'var_name'        : 'sigma',
                        'indices'         : indices_cpg,
                        'nominal_value'   : [1.0, 'mV'],
                        'scaling'         : 10.0,
                    }
                ],
            },
            'stim_a_off' : 0.7,
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
    )


if __name__ == '__main__':
    main()
