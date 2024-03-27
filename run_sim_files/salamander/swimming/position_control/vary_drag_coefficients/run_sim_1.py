''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import matplotlib.pyplot as plt

from typing import Any

from network_experiments.snn_simulation_setup import (
    get_params_processes,
    get_mech_sim_options,
)

from network_experiments.snn_simulation import simulate_single_net_multi_run_position_control_build
import network_experiments.default_parameters.salamander.swimming.position_control.default as default

DURATION = 10
TIMESTEP = 0.001

REF_KINEMATICS_DATA = default.KINEMATICS_DATA_DICT[default.REFERENCE_DATA_NAME]
REF_FREQUENCY       = REF_KINEMATICS_DATA['frequency']
REF_WAVE_NUMBER     = REF_KINEMATICS_DATA['wave_number_ax']
REF_ANGLE_AMP       = REF_KINEMATICS_DATA['joints_displ_amp']
REF_DISP_AMP        = REF_KINEMATICS_DATA['links_displ_amp']
REF_SPEED           = REF_KINEMATICS_DATA['speed_fwd_bl']
REF_TAIL_BEAT       = REF_KINEMATICS_DATA['tail_beat_bl']

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path = '/data/pazzagli/simulation_results_test'

    # Default parameters
    default_params = default.get_default_parameters()

    # Create kinematics file
    kinematics_file = (
        'farms_experiments/experiments/salamandra_v4_position_control_swimming/'
        'kinematics/data_swimming_test.csv'
    )

    default.create_kinematics_file(
        kinematics_file = kinematics_file,
        frequency       = REF_KINEMATICS_DATA['frequency'],
        wave_number     = REF_KINEMATICS_DATA['wave_number_ax'],
        joints_amps     = REF_KINEMATICS_DATA['joints_displ_amp'],
        times           = np.arange(0, DURATION, TIMESTEP),
    )

    # Drag coefficients
    coeff_x = np.array( [ 0.06700, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 ] )
    coeff_y = np.array( [ 0.06700, 0.06400, 0.07075, 0.06863, 0.06597, 0.06370, 0.05880, 0.05530, 0.03675 ] )
    coeff_z = np.array( [ 0.09380, 0.08960, 0.09905, 0.09608, 0.04713, 0.04550, 0.04200, 0.03950, 0.02625 ] )

    drag_coeffs = np.array( [ coeff_x, coeff_y, coeff_z ] ).T

    # Params process
    params_process = default_params['params_process'] | {
        'simulation_data_file_tag': 'test_signal',
        'gaitflag'                : 0,
        'mech_sim_options'        : get_mech_sim_options(video= False, video_fps=30, video_speed=0.2),
        'duration'                : DURATION,
        'timestep'                : TIMESTEP,
    }

    params_process['mech_sim_options'].update(
        {
            'video'      : False,
            'video_fps'  : 30,
            'video_speed': 0.2,

            'position_control_parameters_options' : {
                'kinematics_file' : kinematics_file,

                'position_control_gains' : default.get_position_control_options(
                    gains_p = 7.0e-03,
                    gains_d = 3.0e-06,
                )
            },

            'drag_coefficients_options' : default.get_drag_coefficients_options(
                drag_coeffs = drag_coeffs,
            ),
        }
    )

    # Params runs
    params_runs = [
        {
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
        plot_figures        = False,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = False,
    )

    # Print errors
    speed_fwd = metrics_runs['mech_speed_fwd_bl'][0]
    tail_beat = metrics_runs['mech_tail_beat_amp_bl'][0]
    joint_amp = metrics_runs['mech_joints_disp_amp'][0]
    link_disp = metrics_runs['mech_links_disp_amp_bl'][0]

    joint_error = np.mean( ( joint_amp - REF_ANGLE_AMP) / REF_ANGLE_AMP )
    link_error  = np.mean( ( link_disp - REF_DISP_AMP) / REF_DISP_AMP )
    speed_error = (speed_fwd - REF_SPEED) / REF_SPEED
    tail_error  = (tail_beat - REF_TAIL_BEAT) / REF_TAIL_BEAT

    print(f'Joint angles error: { joint_error * 100 :.5f} %')
    print(f'Link displacements error: { link_error * 100 :.5f} %')
    print(f'Speed error: { speed_error * 100.0 :.5f} %')
    print(f'Tail beat error: { tail_error * 100.0 :.5f} %')

    # Plot
    plt.subplot(2,1,1)
    plt.plot(joint_amp, label='sim')
    plt.plot(REF_ANGLE_AMP, label= 'ref')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(
        default.POINTS_POSITIONS_R_MODEL,
        link_disp,
        label='sim',
    )
    plt.plot(
        default.POINTS_POSITIONS_R_MODEL,
        REF_DISP_AMP,
        label='ref',
    )
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()