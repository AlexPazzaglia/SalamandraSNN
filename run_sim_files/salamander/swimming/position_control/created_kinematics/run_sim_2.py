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
from network_experiments.snn_simulation import simulate_single_net_multi_run_position_control_build
import network_experiments.default_parameters.salamander.swimming.position_control.default as default

DURATION = 10
TIMESTEP = 0.001

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = 'simulation_results'
    simulation_data_file_tag = 'swimming_kinematics'

    # Default parameters
    default_params = default.get_default_parameters()

    # Create kinematics file
    kinematics_data = default_params['kinematics_data']
    kinematics_files_list = [
        (
        'experimental_data/salamander_kinematics_karakasiliotis/swimming/generated_kinematics/'
        f'data_karakasiliotis_swimming_{int(freq*1000)}_mHz_10000_ms.csv'
        )
        for freq in [2.50, 2.75, 3.00, 3.25, 3.5, 3.75, 4.0]
    ]

    # Params process
    params_process = default_params['params_process'] | {
        'duration'                : DURATION,
        'timestep'                : TIMESTEP,
        'simulation_data_file_tag': simulation_data_file_tag,
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
        for kinematics_file in kinematics_files_list
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
        plot_figures        = False,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = False,
    )

    # Save metrics_runs in current folder
    np.save(f'{results_path}/{simulation_data_file_tag}_metrics_runs', metrics_runs)


if __name__ == '__main__':
    main()