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

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_signal_driven_build
import network_experiments.default_parameters.salamander.swimming.signal_driven.default as default

MUSCLE_PARAMETERS_TAG = 'FN_15000_ZC_1000_G0_785_gen_99'
REFERENCE_DATA_NAME   = 'karakasiliotis'

REF_KINEMATICS_DATA = default.KINEMATICS_DATA_DICT[REFERENCE_DATA_NAME]
REF_FREQUENCY       = REF_KINEMATICS_DATA['frequency']
REF_WAVE_NUMBER     = REF_KINEMATICS_DATA['wave_number_ax']
REF_ANGLE_AMP       = REF_KINEMATICS_DATA['joints_displ_amp']
REF_DISP_AMP        = REF_KINEMATICS_DATA['links_displ_amp']
REF_SPEED           = REF_KINEMATICS_DATA['speed_fwd_bl']
REF_TAIL_BEAT       = REF_KINEMATICS_DATA['tail_beat_bl']

def get_sim_params(
    joints_amps : np.ndarray,
    video       : bool = False,
):
    '''
    Get the parameters for the simulation
    '''

    # Default params
    default_params = default.get_default_parameters(
        muscle_parameters_tag = MUSCLE_PARAMETERS_TAG,
        reference_data_name   = REFERENCE_DATA_NAME,
    )

    muscle_params = default_params['params_process']['mech_sim_options']['muscle_parameters_options']

    for muscle_ind in [0, 7]:
        muscle_params[muscle_ind][1]['alpha'] *= 10
        muscle_params[muscle_ind][1]['beta'] *= 10

    # Drag coefficients
    coeff_x = np.array( [ 0.00938, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 ] )
    coeff_y = np.array( [ 0.09380, 0.08960, 0.09905, 0.09608, 0.09425, 0.09100, 0.08400, 0.07900, 0.05250 ] )
    coeff_z = np.array( [ 0.13400, 0.12800, 0.14150, 0.13725, 0.06597, 0.06370, 0.05880, 0.05530, 0.03675 ] )

    drag_coeffs = np.array( [ coeff_x, coeff_y, coeff_z ] ).T

    # Reduce speed and tail beat
    drag_coeffs[ :, 0] *= 20.0
    drag_coeffs[ :, 1] *= 15.0
    drag_coeffs[ :, 2] *= 15.0

    # Reduce trunk oscillations
    drag_coeffs[ 0:6, 1] *= 2.5
    drag_coeffs[ 0:6, 2] *= 2.5

    # Params runs
    params_runs = [
        {
            'motor_output_signal_pars' : default.get_gait_params(
                frequency    = REF_FREQUENCY,
                wave_number  = REF_WAVE_NUMBER,
                joints_amps  = joints_amps,
                sig_function = lambda phase : np.cos( 2*np.pi * phase ),
            ),

            'mech_sim_options' : {
                'drag_coefficients_options' : [
                    [
                        [f'link_body_{link}'],
                        - drag_coeffs[link],
                    ]
                    for link in range(default.N_JOINTS_AXIS+1)
                ],
                'video'      : video,
                'video_fps'  : 30,
                'video_speed': 0.5,
            }
        }
    ]

    # Params process
    params_process = default_params['params_process'] | {
        'simulation_data_file_tag': 'sine_wave_vary_drag',
    }
    params_process['mech_sim_options']['save_all_metrics_data'] = True

    return params_runs, params_process

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Parameters
    results_path     = '/data/pazzagli/simulation_results'
    default_params = default.get_default_parameters(
        muscle_parameters_tag = MUSCLE_PARAMETERS_TAG,
        reference_data_name   = REFERENCE_DATA_NAME,
    )

    # Loop to get joint amplitudes
    joints_amps = np.array(

        # # 1.0 IPL + 20.0, 15.0, 15.0 (rigid 0, 7) + 1.0 2.5 2.5 (0:6)
        [0.14039, 0.20927, 0.20978, 0.22694, 0.42267, 0.61914, 0.34952, 0.55872]

    )

    for trial in range(0):

        # Get params
        params_runs, params_process = get_sim_params(joints_amps)

        # Simulate
        metrics_runs = simulate_single_net_multi_run_signal_driven_build(
            modname             = default_params['modname'],
            parsname            = default_params['parsname'],
            params_process      = get_params_processes(params_process)[0][0],
            params_runs         = params_runs,
            tag_folder          = 'signal_amplitude_ANALYSIS',
            tag_process         = '0',
            save_data           = True,
            plot_figures        = False,
            results_path        = results_path,
            delete_files        = False,
            delete_connectivity = False,
            save_prompt         = False,
        )

        # Joint angles
        joint_angles = metrics_runs['mech_joints_disp_amp'][0]

        # Update joints_amps_scalings
        joints_amps *= REF_ANGLE_AMP / joint_angles

        # Print mse
        mse = np.linalg.norm( joint_angles - REF_ANGLE_AMP)
        print(f'Trial {trial}: mse = {mse}')
        print(
            'Scalings: ['
            + ', '.join( [f'{joint_amp:.5f}' for joint_amp in joints_amps] )
            + ']'
        )


    # SIMULATE WITH THE NEW JOINT AMPLITUDES
    video = False
    plot  = False
    save  = False

    # Get params
    params_runs, params_process = get_sim_params(joints_amps, video= video)

    # Simulate
    metrics_runs = simulate_single_net_multi_run_signal_driven_build(
        modname             = default_params['modname'],
        parsname            = default_params['parsname'],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'signal_amplitude_ANALYSIS',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = plot,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = save,
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

    # Save scalings
    np.savetxt(
        (
            'run_anl_files/salamander/swimming/signal_driven/'
            'motor_output_gains_vs_drag_coefficient/joints_signals_amps.csv'
        ),
        joints_amps,
        delimiter=',',
    )


if __name__ == '__main__':
    main()