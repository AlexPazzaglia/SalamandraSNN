''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_position_control_build
import network_experiments.default_parameters.salamander.swimming.position_control.default as default

DURATION = 10
TIMESTEP = 0.001

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = '/data/pazzagli/simulation_results_test'
    simulation_data_file_tag = 'swimming_kinematics'

    # Default parameters
    default_params = default.get_default_parameters()

    # Create kinematics file
    kinematics_file = (
        f'{PACKAGEDIR}/experimental_data/salamander_kinematics_karakasiliotis/'
        'swimming/generated_kinematics/'
        'data_karakasiliotis_swimming_3170_mHz_10000_ms.csv'
    )

    # Params process
    params_process = default_params['params_process'] | {
        'simulation_data_file_tag': simulation_data_file_tag,
        'duration'                : DURATION,
        'timestep'                : TIMESTEP,
    }

    params_process['mech_sim_options'].update(
        {
            'video'     : False,
            'video_fps' : 30,
        }
    )

    # Params runs
    params_runs = [
        {
            'mech_sim_options' : {

                'position_control_parameters_options' : {
                    'kinematics_file' : kinematics_file,
                }

            }
        }
    ]

    # Simulate
    metrics_runs = simulate_single_net_multi_run_position_control_build(
        modname             = default_params['modname'],
        parsname            = default_params['parsname'],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = True,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = False,
    )

if __name__ == '__main__':
    main()