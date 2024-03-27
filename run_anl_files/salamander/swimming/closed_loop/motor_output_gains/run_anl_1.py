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
from network_experiments.snn_simulation import simulate_single_net_multi_run_closed_loop_build
import network_experiments.default_parameters.salamander.swimming.closed_loop.default as default

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
    joints_amps_scalings: np.ndarray,
    load_connectivity   : bool = False,
    video               : bool = False,
):
    '''
    Get the parameters for the simulation
    '''

    # Default parameters
    default_params = default.get_default_parameters(
        muscle_parameters_tag = MUSCLE_PARAMETERS_TAG,
        reference_data_name   = REFERENCE_DATA_NAME,
    )

    # Params process
    params_process = default_params['params_process'] | {

        'simulation_data_file_tag' : 'cpg_driven_swimming',
        'load_connectivity_indices': load_connectivity,
        'mc_gain_axial'            : joints_amps_scalings,
        # 'stim_a_off'               : np.linspace(0.0, 0.42, 6) + 1.066,
        'stim_a_off'               : 0.7,
    }

    # Optional video
    params_process['mech_sim_options']['video'] = video

    # Params runs
    params_runs = [
        {
        }
    ]

    return params_runs, params_process


def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Default parameters
    results_path   = '/data/pazzagli/simulation_results'
    default_params = default.get_default_parameters(
        muscle_parameters_tag = MUSCLE_PARAMETERS_TAG,
        reference_data_name   = REFERENCE_DATA_NAME,
    )

    # Loop to get joint amplitudes
    joints_amps_scalings = np.array(
        [0.56809, 0.75710, 1.00703, 1.41958, 1.60032, 1.41649, 1.24181, 1.63056]
    )

    for trial in range(0):

        # Get params
        params_runs, params_process = get_sim_params(
            joints_amps_scalings = joints_amps_scalings,
            load_connectivity    = True,
        )

        # Simulate
        metrics_runs = simulate_single_net_multi_run_closed_loop_build(
            modname             = default_params['modname'],
            parsname            = default_params['parsname'],
            params_process      = get_params_processes(params_process)[0][0],
            params_runs         = params_runs,
            tag_folder          = 'ANALYSIS',
            tag_process         = '0',
            save_data           = True,
            plot_figures        = False,
            results_path        = results_path,
            delete_files        = False,
            delete_connectivity = False,
        )

        # Joint angles
        joint_angles = metrics_runs['mech_joints_disp_amp'][0]

        # Update joints_amps_scalings
        joints_amps_scalings *= ( 1 + 0.5 * (-1 + REF_ANGLE_AMP / joint_angles) )

        # Print mse
        mse = np.linalg.norm( joint_angles - REF_ANGLE_AMP)
        print(f'Trial {trial}: mse = {mse}')
        print(
            'Scalings: ['
            + ', '.join( [f'{joint_amp:.5f}' for joint_amp in joints_amps_scalings] )
            + ']'
        )

    # SIMULATE WITH THE NEW JOINT AMPLITUDES
    video = False
    plot  = False
    save  = False

    # Get params
    params_runs, params_process = get_sim_params(
        joints_amps_scalings = joints_amps_scalings,
        load_connectivity    = True,
        video                = video
    )

    # Simulate
    metrics_runs = simulate_single_net_multi_run_closed_loop_build(
        modname             = default_params['modname'],
        parsname            = default_params['parsname'],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'ANALYSIS',
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
            'run_anl_files/salamander/swimming/closed_loop/'
            'motor_output_gains/joints_signals_amps.csv'
        ),
        joints_amps_scalings,
        delimiter=',',
    )


if __name__ == '__main__':
    main()
