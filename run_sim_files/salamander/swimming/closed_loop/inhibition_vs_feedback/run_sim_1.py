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
    simulation_data_file_tag = 'inhibition_vs_feedback'

    # Default parameters
    default_params = default.get_default_parameters()

    # Process parameters
    inhibition_amplitude = 0.5

    params_process = default_params['params_process'] | {
        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': False,

        'ps_gain_axial': default.get_scaled_ps_gains(
            alpha_fraction_th   = 10.0,
            reference_data_name = default_params['reference_data_name'],
        ),

        # CONNECTIVITY
        'connectivity_axial_newpars' : {
            'ax2ax': [
                {
                    'name'      : 'AX_in -> AX_all Contra',
                    'synapse'   : 'syn_in',
                    'type'      : 'gaussian_identity',
                    'parameters': {
                        'y_type'    : 'y_neur',
                        'amp'       : inhibition_amplitude,
                        'sigma_up'  : 0.8,
                        'sigma_dw'  : 1.2,
                    },
                    'cond_list' : [ ['', 'opposite', 'ax', 'in', 'ax', ['ex', 'in']] ],
                    'cond_str'  : '',
                }
            ]
        }

    }

    # Params runs
    params_runs = [
        {
            'stim_a_off' : 0.7,
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
