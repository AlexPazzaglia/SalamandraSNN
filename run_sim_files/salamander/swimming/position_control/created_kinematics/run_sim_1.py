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

DURATION = 20
TIMESTEP = 0.001

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = '/data/pazzagli/simulation_results_test'
    simulation_data_file_tag = 'swimming_kinematics'

    # Default parameters
    default_params = default.get_default_parameters()

    # Create kinematics file
    kinematics_data = default_params['kinematics_data']
    kinematics_file = (
        'farms_experiments/experiments/salamandra_v4_position_control_swimming/'
        'kinematics/data_swimming_test.csv'
    )

    default.create_kinematics_file(
        kinematics_file = kinematics_file,
        frequency       = kinematics_data['frequency'],
        wave_number     = kinematics_data['wave_number_ax'],
        joints_amps     = kinematics_data['joints_displ_amp'],
        times           = np.arange(0, DURATION, TIMESTEP),
    )

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

                    # 'position_control_gains' : [
                    #     [
                    #         [f'joint_body_{joint}' for joint in range(default.N_JOINTS_AXIS)],
                    #         [
                    #             5.0e-04,
                    #             1.0e-08,
                    #         ]
                    #     ]
                    # ]

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