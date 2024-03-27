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
MAX_PS_RANGE = 0.05

def get_ps_to_ax_ex(amp):
    ''' Get the proprioceptive feedback connections '''
    return {
        'name'      : 'AX_ps -> AX_all Ipsi',
        'synapse'   : 'syn_ex',
        'type'      : 'connect_identity',
        'parameters': {
            'amp'     : amp,
        },
        'cond_list' : [
            [ '', 'ipsi', 'ax', 'ps', 'ax', ['ex', 'in']]
        ],
        'cond_str'  : f'( abs( y_mech_post - y_mech_pre ) < {MAX_PS_RANGE} * metre)'
    }

def get_ps_to_mn_ex(amp):
    ''' Get the proprioceptive feedback connections '''
    return {
        'name'      : 'AX_ps -> AX_mn Ipsi',
        'synapse'   : 'syn_ex',
        'type'      : 'connect_identity',
        'parameters': {
            'amp'     : amp,
        },
        'cond_list' : [
            [ '', 'ipsi', 'ax', 'ps', 'ax', 'mn']
        ],
        'cond_str'  : f'( abs( y_mech_post - y_mech_pre ) < {MAX_PS_RANGE} * metre)'
    }

def get_ps_to_ax_in(amp):
    ''' Get the proprioceptive feedback connections '''
    return {
        'name'      : 'AX_ps -> AX_all Contra',
        'synapse'   : 'syn_in',
        'type'      : 'connect_identity',
        'parameters': {
            'amp'     : amp,
        },
        'cond_list' : [
            ['', 'contra', 'ax', 'ps', 'ax', ['ex', 'in']]
        ],
        'cond_str'  : f'( abs( y_mech_post - y_mech_pre ) < {MAX_PS_RANGE} * metre)'
    }

def get_ps_to_mn_in(amp):
    ''' Get the proprioceptive feedback connections '''
    return {
        'name'      : 'AX_ps -> AX_mn Contra',
        'synapse'   : 'syn_in',
        'type'      : 'connect_identity',
        'parameters': {
            'amp'     : amp,
        },
        'cond_list' : [
            ['', 'contra', 'ax', 'ps', 'ax', 'mn']
        ],
        'cond_str'  : f'( abs( y_mech_post - y_mech_pre ) < {MAX_PS_RANGE} * metre)'
    }

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Shared by processes
    args = parse_arguments()

    results_path             = args['results_path']
    simulation_data_file_tag = 'feedback_topology'

    # Default parameters
    default_params = default.get_default_parameters()

    # Process parameters
    params_process = default_params['params_process'] | {
        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': True,
        'stim_a_off'               : 0.7,

        'ps_gain_axial': default.get_scaled_ps_gains(
            alpha_fraction_th   = 0.80,
            reference_data_name = default_params['reference_data_name'],
        ),

        'connectivity_axial_newpars' : {
            'ps2ax': [
                get_ps_to_ax_in(amp= 0.5),
                get_ps_to_mn_ex(amp= 0.5),
                get_ps_to_ax_ex(amp= 0.5),
                get_ps_to_mn_in(amp= 0.5),
            ]
        }
    }

    # Params runs
    params_runs = (
        [
            {
                'sig_ex_up' : MAX_PS_RANGE,
                'sig_ex_dw' : MAX_PS_RANGE,
                'sig_in_up' : 0,
                'sig_in_dw' : 0,
            },
            {
                'sig_ex_up' : 0,
                'sig_ex_dw' : 0,
                'sig_in_up' : MAX_PS_RANGE,
                'sig_in_dw' : MAX_PS_RANGE,
            }
        ]
    )

    # Simulate
    simulate_single_net_multi_run_closed_loop_build(
        modname             = f'{CURRENTDIR}/net_farms_limbless_feedback_topology.py',
        parsname            = default_params['parsname'],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = False,
        plot_figures        = False,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
    )

if __name__ == '__main__':
    main()
