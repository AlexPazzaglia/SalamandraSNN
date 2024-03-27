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
REFERENCE_DATA_NAME   = 'canada'

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
    video_fps   : int  = 30,
    video_speed : float = 0.5,
):
    '''
    Get the parameters for the simulation
    '''

    # Default params
    default_params = default.get_default_parameters(
        muscle_parameters_tag = MUSCLE_PARAMETERS_TAG,
        reference_data_name   = REFERENCE_DATA_NAME,
    )

    # Increase stiffness of head and tail
    # muscle_params = default_params['params_process']['mech_sim_options']['muscle_parameters_options']
    # for muscle_ind in [0, 7]:
    #     muscle_params[muscle_ind][1]['alpha'] *= 10
    #     muscle_params[muscle_ind][1]['beta'] *= 10

    # Drag coefficients
    # drag_coeffs.reshape((27))
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
                sig_function = lambda phase : np.cos( 2*np.pi * phase )
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
                'video_fps'  : video_fps,
                'video_speed': video_speed,
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
        # # 1.1 IPL + 1.0, 1.0, 1.0
        # [0.08558, 0.13959, 0.17301, 0.27187, 0.34845, 0.34428, 0.32820, 0.16874]

        # # 1.1 IPL + 2.0, 2.0, 2.0
        # [0.08635, 0.14279, 0.17610, 0.27540, 0.34950, 0.34058, 0.31015, 0.10201]

        # # 1.1 IPL + 3.0, 3.0, 3.0
        # [0.08051, 0.14733, 0.18178, 0.28250, 0.35215, 0.33513, 0.28203, 0.02722]

        # # 1.1 IPL + 6.0, 6.0, 6.0
        # [0.06176, 0.15218, 0.18790, 0.28962, 0.35445, 0.32957, 0.25900, 0.00375]

        # # 1.1 IPL + 10.0, 10.0, 10.0
        # [0.00036, 0.15952, 0.20166, 0.30619, 0.36056, 0.31981, 0.22039, 0.00015]

        # # 1.1 IPL + 10.0, 10.0, 10.0 (rigid 0, 7)
        # [0.05131, 0.15693, 0.19731, 0.30270, 0.36128, 0.32446, 0.20528, 0.25233]

        # # 1.1 IPL + 15.0, 15.0, 15.0 (rigid 0, 7)
        # [0.04991, 0.16319, 0.20855, 0.31624, 0.36616, 0.31561, 0.16176, 0.23612]

        # # 1.1 IPL + 15.0, 15.0, 15.0 (rigid 0, 7) + 1.0 1.5 1.5 (1:5)
        # [0.04924, 0.16299, 0.21267, 0.32239, 0.36798, 0.31420, 0.15736, 0.23695]

        # # 1.1 IPL + 15.0, 15.0, 15.0 (rigid 0, 7) + 1.0 2.0 2.0 (1:6)
        # [0.04912, 0.16236, 0.21427, 0.32576, 0.37093, 0.31423, 0.15213, 0.23715]

        # # 1.0 IPL + 15.0, 15.0, 15.0 (rigid 0, 7) + 1.0 2.0 2.0 (0:6)
        # [0.04969, 0.18103, 0.23051, 0.33984, 0.37729, 0.31303, 0.11295, 0.22953]

        # # 1.0 IPL + 20.0, 15.0, 15.0 (rigid 0, 7) + 1.0 2.0 2.0 (0:6)
        # [0.04988, 0.18776, 0.23680, 0.34543, 0.37817, 0.30549, 0.08383, 0.22282]

        # # 1.0 IPL + 20.0, 15.0, 15.0 (rigid 0, 7) + 1.0 2.5 2.5 (0:6) --> BEST
        [0.04851, 0.18704, 0.24170, 0.35185, 0.38100, 0.30478, 0.07681, 0.22384]

        # # 1.0 IPL + 25.0, 15.0, 15.0 (rigid 0, 7) + 1.0 2.0 2.0 (0:6)
        # [0.04997, 0.19307, 0.24192, 0.34998, 0.37887, 0.29912, 0.05919, 0.21853]

        # # 1.0 IPL + 25.0, 15.0, 15.0 (rigid 0, 7) + 1.0 2.5 2.5 (0:6)
        # [0.04856, 0.19245, 0.24715, 0.35669, 0.38173, 0.29806, 0.05184, 0.21972]

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
    video = True
    plot  = False
    save  = False

    # Get params
    params_runs, params_process = get_sim_params(joints_amps, video= video, video_speed= 1.0)

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
    plt.figure()
    plt.plot(joint_amp, label='sim')
    plt.plot(REF_ANGLE_AMP, label= 'ref')
    plt.legend()
    plt.title('Joint angles')
    plt.xlabel('Joint Index')
    plt.ylabel('Amplitude [rad]')

    plt.figure()
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
    plt.title('Points displacements')
    plt.xlabel('Axial position [%BL]')
    plt.ylabel('Amplitude [%BL]')

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