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
from network_experiments.snn_analysis_parser import parse_arguments
import network_experiments.default_parameters.salamander.swimming.closed_loop.default as default

# Get the proprioceptive feedback connections
def get_ps_to_ax_ex(amp, sigma_up, sigma_dw):
    ''' Get the proprioceptive feedback gains '''
    return {
        'name'      : 'AX_ps -> AX_all Ipsi',
        'synapse'   : 'syn_ex',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_mech',
            'amp'     : amp,
            'sigma_up': sigma_up,
            'sigma_dw': sigma_dw,
        },
        'cond_list' : [
            [ '', 'ipsi', 'ax', 'ps', 'ax', ['ex', 'in']]
        ],
        'cond_str'  : ''
    }

def get_ps_to_mn_ex(amp, sigma_up, sigma_dw):
    ''' Get the proprioceptive feedback gains '''
    return {
        'name'      : 'AX_ps -> AX_mn Ipsi',
        'synapse'   : 'syn_ex',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_mech',
            'amp'     : amp,
            'sigma_up': sigma_up,
            'sigma_dw': sigma_dw,
        },
        'cond_list' : [
            [ '', 'ipsi', 'ax', 'ps', 'ax', 'mn']
        ],
        'cond_str'  : ''
    }

def get_ps_to_ax_in(amp, sigma_up, sigma_dw):
    ''' Get the proprioceptive feedback gains '''
    return {
        'name'      : 'AX_ps -> AX_all Contra',
        'synapse'   : 'syn_in',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_mech',
            'amp'     : amp,
            'sigma_up': sigma_up,
            'sigma_dw': sigma_dw,
        },
        'cond_list' : [
            ['', 'contra', 'ax', 'ps', 'ax', ['ex', 'in']]
        ],
        'cond_str'  : ''
    }

def get_ps_to_mn_in(amp, sigma_up, sigma_dw):
    ''' Get the proprioceptive feedback gains '''
    return {
        'name'      : 'AX_ps -> AX_mn Contra',
        'synapse'   : 'syn_in',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_mech',
            'amp'     : amp,
            'sigma_up': sigma_up,
            'sigma_dw': sigma_dw,
        },
        'cond_list' : [
            ['', 'contra', 'ax', 'ps', 'ax', 'mn']
        ],
        'cond_str'  : ''
    }

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Shared by processes
    args = parse_arguments()

    results_path             = args['results_path']
    simulation_data_file_tag = 'vary_fb_strength'

    # Default parameters
    default_params = default.get_default_parameters()

    # Process parameters
    ex_up = 5.0
    ex_dw = 5.0
    in_up = 5.0
    in_dw = 5.0

    params_process = default_params['params_process'] | {

        'simulation_data_file_tag': simulation_data_file_tag,
        'load_connectivity_indices': False,

        'ps_gain_axial': default.get_scaled_ps_gains(
            alpha_fraction_th   = 0.80,
            reference_data_name = default_params['reference_data_name'],
        ),

        'connectivity_axial_newpars' : {
            'ps2ax': [

                # INHIBITION
                get_ps_to_ax_in(amp= 0.5, sigma_up=in_up, sigma_dw=in_dw),
                get_ps_to_mn_in(amp= 0.5, sigma_up=in_up, sigma_dw=in_dw),

                # EXCITATION
                get_ps_to_ax_ex(amp= 0.5, sigma_up=ex_up, sigma_dw=ex_dw),
                get_ps_to_mn_ex(amp= 0.5, sigma_up=ex_up, sigma_dw=ex_dw),

            ]
        }
    }

    # Params runs
    params_runs = [
        {
        }
    ]

    # Simulate
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
