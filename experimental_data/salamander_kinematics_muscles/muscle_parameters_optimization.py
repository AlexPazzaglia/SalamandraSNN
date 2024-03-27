''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import dill
import copy
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import shutil
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Process
from typing import Any, Callable

from network_experiments.snn_simulation_setup import (
    get_params_processes,
    get_mech_sim_options,
)

from network_experiments.snn_signals_neural import get_motor_output_signal

from network_experiments.snn_simulation import simulate_single_net_multi_run_signal_driven_build
from network_experiments.snn_utils import MODELS_FARMS

from experimental_data.salamander_kinematics_muscles import muscle_parameters_dynamic_system_analysis


DEFAULT_ALPHA   = 3.14e-4
DEFAULT_BETA    = 2.00e-4
DEFAULT_DELTA   = 10.0e-6
DEFAULT_GAMMA   = 1.0
DEFAULT_EPSILON = 0.0

### MOTOR OUTPUT FUNCTION
def get_motor_output_function_params_multi_joint(
    n_joints_axis : int,
    n_joints_limbs: int,
    joints_indices: np.ndarray[int],
    sig_function  : Callable,
    sig_amplitude : float,
    sig_frequency : float,
    amp_arr       : np.ndarray[float] = None,
    ipl_arr       : np.ndarray[float] = None,
    off_arr       : np.ndarray[float] = None,
) -> dict:
    ''' Get motor output function params for multiple joints '''

    if amp_arr is None:
        amp_arr = np.ones(n_joints_axis)

    if ipl_arr is None:
        ipl_arr = np.zeros(n_joints_axis)

    if off_arr is None:
        off_arr = np.zeros(n_joints_axis)

    amp_array = np.zeros(n_joints_axis)
    amp_array[joints_indices] = sig_amplitude
    amp_array = amp_array * amp_arr

    motor_output_function_params = {
        'sig_function' : sig_function,

        'axis': {
            'frequency': sig_frequency,
            'n_copies' : 1,
            'ipl_off'  : [0.0],
            'names'    : ['AXIS'],

            'n_joints': n_joints_axis,
            'ipl_arr' : ipl_arr,
            'off_arr' : off_arr,
            'amp_arr' : amp_array,
        },

        'limbs': {
            'frequency': sig_frequency,
            'n_copies' : 4,
            'ipl_off'  : [ 0.0,  0.0,  0.0,  0.0],
            'names'    : ['LF', 'RF', 'LH', 'RH'],

            'n_joints': n_joints_limbs,
            'ipl_arr' : np.zeros(n_joints_limbs),
            'amp_arr' : np.zeros(n_joints_limbs),
            'off_arr' : np.zeros(n_joints_limbs),
        },
    }

    return motor_output_function_params

def get_motor_output_function_params_single_joint(
    n_joints_axis : int,
    n_joints_limbs: int,
    joint_index   : int,
    sig_function  : Callable,
    sig_amplitude : float,
    sig_frequency : float,
) -> dict:
    ''' Get motor output function params for a single joint '''
    return get_motor_output_function_params_multi_joint(
        n_joints_axis  = n_joints_axis,
        n_joints_limbs = n_joints_limbs,
        joints_indices = np.array([joint_index]),
        sig_function   = sig_function,
        sig_amplitude  = sig_amplitude,
        sig_frequency  = sig_frequency,
    )

### RUN PARAMS
def get_run_params_for_multi_joint_activations(
    waterless           : bool,
    n_joints_axis       : int,
    n_joints_limbs      : int,
    tested_joints       : list[int],
    sig_function        : Callable,
    sig_amplitude       : float,
    sig_frequency       : float,
    gains_scalings_alpha: np.ndarray,
    gains_scalings_beta : np.ndarray,
    gains_scalings_delta: np.ndarray,
    amp_arr             : np.ndarray[float] =  None,
    ipl_arr             : np.ndarray[float] =  None,
    off_arr             : np.ndarray[float] =  None,
):
    ''' Get run params for multiple joints '''

    sig_pars = get_motor_output_function_params_multi_joint(
        n_joints_axis  = n_joints_axis,
        n_joints_limbs = n_joints_limbs,
        joints_indices = np.array(tested_joints),
        sig_function   = sig_function,
        sig_amplitude  = sig_amplitude,
        sig_frequency  = sig_frequency,
        amp_arr        = amp_arr,
        ipl_arr        = ipl_arr,
        off_arr        = off_arr,
    )

    water_options = {} if not waterless else {
        'gravity' : (0.0, 0.0, 0.0),
        'spawn_parameters_options'  :
            {
                'pose' : ( 0.0, 0.0, 1.01, 0.0, 0.0, 3.141592653589793 )
            },
        'drag_coefficients_options' : [
            [
                [f'link_body_{i}' for i in range(n_joints_axis+1)],
                [-0.0, -0.0, -0.0]
            ]
        ],
    }

    params_runs = [
        {
            'tag_run' : 0,

            'motor_output_signal_pars': sig_pars,

            'mech_sim_options' : {

                'save_all_metrics_data' : True,

                'muscle_parameters_options': [
                    [
                        [f'joint_body_{muscle}'],
                        {
                            'alpha'  : 3.0e-4    * gains_scalings_alpha[muscle],    # 3.1e-4,
                            'beta'   : 2.0e-4    * gains_scalings_beta[muscle],     # 2.0e-4,
                            'delta'  : 10.0e-06  * gains_scalings_delta[muscle],    # 5.0e-6,
                            'gamma'  : 1.0,                                         # 1.0,
                            'epsilon': 0,
                        }
                    ]
                    for muscle in range(n_joints_axis)
                ],
            } | water_options
        }
    ]

    return params_runs

def get_run_params_for_single_joint_activations(
    waterless           : bool,
    n_joints_axis       : int,
    n_joints_limbs      : int,
    tested_joints       : list[int],
    sig_function        : Callable,
    sig_amplitude       : float,
    sig_frequency       : float,
    gains_scalings_alpha: np.ndarray,
    gains_scalings_beta : np.ndarray,
    gains_scalings_delta: np.ndarray,
):
    ''' Get run params for a single joint '''

    sig_pars = [
        get_motor_output_function_params_single_joint(
            n_joints_axis  = n_joints_axis,
            n_joints_limbs = n_joints_limbs,
            joint_index    = joint_index,
            sig_function   = sig_function,
            sig_amplitude  = sig_amplitude,
            sig_frequency  = sig_frequency,
        )
        for joint_index in tested_joints
    ]

    water_options = {} if not waterless else {
        'gravity' : (0.0, 0.0, 0.0),
        'spawn_parameters_options'  :
            {
                'pose' : ( 0.0, 0.0, 1.01, 0.0, 0.0, 3.141592653589793 )
            },
        'drag_coefficients_options' : [
            [
                [f'link_body_{i}' for i in range(n_joints_axis+1)],
                [-0.0, -0.0, -0.0]
            ]
        ],
    }

    params_runs = [
        {
            'tag_run'                 : run_index,
            'motor_output_signal_pars': sig_pars[run_index],

            'mech_sim_options' : {

                'save_all_metrics_data' : True,

                'muscle_parameters_options': [
                    [
                        [f'joint_body_{muscle}'],
                        {
                            'alpha'  : DEFAULT_ALPHA * gains_scalings_alpha[muscle],  # 3.1e-4,
                            'beta'   : DEFAULT_BETA  * gains_scalings_beta[muscle],   # 2.0e-4,
                            'delta'  : DEFAULT_DELTA * gains_scalings_delta[muscle],  # 5.0e-6,
                            'gamma'  : DEFAULT_GAMMA,                                 # 1.0,
                            'epsilon': DEFAULT_EPSILON,
                        }
                    ]
                    for muscle in range(n_joints_axis)
                ],
            } | water_options
        }
        for run_index, joint_index in enumerate(tested_joints)
    ]

    return params_runs

### PARAMETERS ESTIMATION
def get_muscle_inertial_scaling_factors():
    ''' Get muscle scaling factors '''

    n_joints_axis = 8

    # Data from the model
    links_ref_frames_x = np.array(
        [
            0.0000,
            0.0150,
            0.0257,
            0.0364,
            0.0471,
            0.0578,
            0.0685,
            0.0792,
            0.0899,
        ]
    )

    joints_x = np.array(
        [
            0.0150,
            0.0257,
            0.0364,
            0.0471,
            0.0578,
            0.0685,
            0.0792,
            0.0899,
        ]
    )

    links_masses = np.array(
        [
            0.0003314,
            0.0005139,
            0.0006302,
            0.0005277,
            0.0002672,
            0.0002091,
            0.0001574,
            0.0001188,
            0.0000754,
        ]
    )

    links_inertias = np.array(
        [
            4.2878e-09,
            7.8835e-09,
            1.1056e-08,
            8.6788e-09,
            3.3960e-09,
            2.3738e-09,
            1.7043e-09,
            1.2165e-09,
            5.0731e-10,
        ]
    )

    # Transfer to joints frames
    inertias_in_joints_pos = np.array(
        [
            [
                links_inertias[link] +
                links_masses[link] * (joints_x[joint] - links_ref_frames_x[link]) ** 2
                for link in range(n_joints_axis+1)
            ]
            for joint in range(n_joints_axis)
        ]
    )

    # Rostral inertia
    inertias_in_joint_pos_rost = np.array(
        [
            np.sum( inertias_in_joints_pos[joint, :joint+1] )
            for joint in range(n_joints_axis)
        ]
    )

    # Caudal inertia
    inertias_in_joint_pos_caud = np.array(
        [
            np.sum( inertias_in_joints_pos[joint, joint+1:] )
            for joint in range(n_joints_axis)
        ]
    )

    # Consider the minimum
    # inertias_seen_by_joints = (
    #     (inertias_in_joint_pos_caud * inertias_in_joint_pos_rost) /
    #     (inertias_in_joint_pos_caud + inertias_in_joint_pos_rost)
    # )

    inertias_seen_by_joints = np.amin(
        [
            inertias_in_joint_pos_rost,
            inertias_in_joint_pos_caud
        ],
        axis=0
    )

    inertia_factor = inertias_seen_by_joints / np.amax(inertias_seen_by_joints)

    return inertia_factor, inertias_seen_by_joints

def get_estimated_natural_frequencies_and_damping_ratios(
    n_joints_axis          : int,
    muscle_pars            : list,
    inertias_seen_by_joints: np.ndarray,
):
    ''' Get estimated natural frequencies and damping ratios '''

    # NOTE:
    # muscle_pars = [
    #   params_runs[0]['mech_sim_options']['muscle_parameters_options'][joint][1]
    #   for joint in range(n_joints_axis)
    # ]

    stiffnesses = np.array(
        [
            muscle_pars[joint]['beta'] + muscle_pars[joint]['gamma'] * muscle_pars[joint]['beta']
            for joint in range(n_joints_axis)
        ]
    )
    dampings = np.array(
        [
            muscle_pars[joint]['delta']
            for joint in range(n_joints_axis)
        ]
    )

    natural_frequencies = np.sqrt( stiffnesses / inertias_seen_by_joints )
    damping_ratios      = dampings / ( 2 * np.sqrt( inertias_seen_by_joints * stiffnesses ) )

    print('Natural frequencies:')
    print(natural_frequencies)

    print('Damping ratios:')
    print(damping_ratios)

    return natural_frequencies, damping_ratios

### MUSCLE SCALING
def get_muscle_scalings_from_inertias(
    inertial_scaling_factor      : float,
):
    ''' Get muscle scaling factors '''

    (
        inertia_factor,
        _inertias_seen_by_joints,
    ) = get_muscle_inertial_scaling_factors()

    # NOTE: With a global_factor 20 inertia becomes dominant
    # NOTE: With a global_factor 5 the model becomes unstable

    total_scaling_factor = inertia_factor / inertial_scaling_factor

    gains_scalings_alpha = total_scaling_factor
    gains_scalings_beta  = total_scaling_factor
    gains_scalings_delta = np.sqrt(total_scaling_factor)

    return gains_scalings_alpha, gains_scalings_beta, gains_scalings_delta

def _get_performance_from_iteration(
    optimization_name: str,
    iteration_ind    : int,
):
    ''' Get performance from iteration '''

    folder_name = f'experimental_data/salamander_kinematics_muscles/results/{optimization_name}'
    file_name   = f'{folder_name}/performance_iteration_{iteration_ind}.dill'
    with open(file_name, 'rb') as file:
        performance = dill.load(file)

    return performance

def get_muscle_scalings_from_iteration(
    optimization_name: str,
    iteration_ind    : int,
):
    ''' Get muscle scaling factors from iteration'''

    performance = _get_performance_from_iteration(
        optimization_name = optimization_name,
        iteration_ind     = iteration_ind,
    )

    return (
        performance['gains_scalings_alpha'],
        performance['gains_scalings_beta'],
        performance['gains_scalings_delta']
    )

def get_muscle_scalings(
    n_joints_axis            : int,
    manual_scaling_alpha     : np.ndarray,
    manual_scaling_beta      : np.ndarray,
    manual_scaling_delta     : np.ndarray,
    use_original_scalings    : bool  = None,
    use_inertial_scalings    : bool  = None,
    use_optimization_scalings: bool  = None,
    inertial_scaling_factor  : float = None,
    optimization_name        : str   = None,
    optimization_iteration   : int   = None,
):
    ''' Get muscle scaling factors '''

    if np.count_nonzero((use_inertial_scalings, use_optimization_scalings, use_original_scalings)) != 1:
        raise ValueError('Only one of the three options can be True')

    # Scalings of the original model
    if use_original_scalings:
        scalings = (
            np.ones(n_joints_axis),
            np.ones(n_joints_axis),
            np.ones(n_joints_axis),
        )

    # Scalings from optimization results
    if use_optimization_scalings:
            scalings =  get_muscle_scalings_from_iteration(
                optimization_name = optimization_name,
                iteration_ind     = optimization_iteration,
            )

    # Scalings from inertias
    if use_inertial_scalings:
        scalings = get_muscle_scalings_from_inertias(
            inertial_scaling_factor = inertial_scaling_factor,
        )

    gains_scalings_alpha, gains_scalings_beta, gains_scalings_delta = scalings

    gains_scalings_alpha *= manual_scaling_alpha
    gains_scalings_beta  *= manual_scaling_beta
    gains_scalings_delta *= manual_scaling_delta

    return gains_scalings_alpha, gains_scalings_beta, gains_scalings_delta

def get_muscle_scalings_from_solution(
    n_joints_axis            : int,
    optimization_name        : str   = None,
    optimization_iteration   : int   = None,
):
    ''' Get muscle scaling factors from an optimization solution '''


    return get_muscle_scalings(
        n_joints_axis            = n_joints_axis,
        manual_scaling_alpha     = np.ones(n_joints_axis),
        manual_scaling_beta      = np.ones(n_joints_axis),
        manual_scaling_delta     = np.ones(n_joints_axis),
        use_original_scalings    = False,
        use_inertial_scalings    = False,
        use_optimization_scalings= True,
        inertial_scaling_factor  = None,
        optimization_name        = optimization_name,
        optimization_iteration   = optimization_iteration,
    )

def get_muscle_parameters_options_from_solution(
    n_joints_axis            : int,
    optimization_name        : str   = None,
    optimization_iteration   : int   = None,
):
    ''' Get muscle parameters options from an optimization solution '''

    (
        gains_scalings_alpha,
        gains_scalings_beta,
        gains_scalings_delta,
    ) = get_muscle_scalings_from_solution(
        n_joints_axis          = n_joints_axis,
        optimization_name      = optimization_name,
        optimization_iteration = optimization_iteration,
    )

    muscle_parameters_options = [
        [
            [f'joint_body_{muscle}'],
            {
                'alpha'  : DEFAULT_ALPHA * gains_scalings_alpha[muscle],
                'beta'   : DEFAULT_BETA  * gains_scalings_beta[muscle],
                'delta'  : DEFAULT_DELTA * gains_scalings_delta[muscle],
                'gamma'  : DEFAULT_GAMMA,
                'epsilon': DEFAULT_EPSILON,
            }
        ]
        for muscle in range(n_joints_axis)
    ]

    return muscle_parameters_options

### OPTIMIZATION
def _get_muscle_parameters_joint_error_iteration(
    params_muscles       : list[dict[str, Any]],
    filename             : str,
    run_index            : int,
    target_wn            : float,
    target_zc            : float,
):
    ''' Get improved muscle parameters for a single joint '''

    with open(filename, 'rb') as file:
        data_run = dill.load(file)

    joints_commands : np.ndarray = data_run['joints_commands']
    joints_angles   : np.ndarray = data_run['joints_positions']

    if joints_commands.shape[0] == joints_angles.shape[0] + 1:
        joints_commands = joints_commands[1:, :]
    if joints_commands.shape[0] == joints_angles.shape[0] - 1:
        joints_angles   = joints_angles[1:, :]

    joint_index    = TESTED_JOINTS[run_index]
    joint_angles   = joints_angles[:, joint_index]

    joint_alpha    = params_muscles[run_index]['alpha']
    joint_commands = joints_commands[:, 2*joint_index+1] - joints_commands[:, 2*joint_index]
    joint_torques  = joint_alpha * joint_commands

    G_cont, _G_disc = muscle_parameters_dynamic_system_analysis.get_fitting_second_order_system(
        signal_input     = joint_torques,
        signal_response  = joint_angles,
        times            = np.arange(0.0, DURATION, TIMESTEP),
        freq_sampling    = 1/TIMESTEP,
        train_percentage = 0.7,
        plot             = PLOTTING,
    )

    # Estimated parameters
    # NOTE: Alpha was not included in the optimization (torques provided)

    #              alpha * B0                       G0                           alpha / M
    # H(s) = ------------------------ = ------------------------------- = ---------------------------
    #         s^2 + 2 ZC WN s + WN^2     (WN^-2)s^2 + (2 ZC / WN)s + 1     s^2 + ( c/M ) s + ( k/M )

    num = G_cont.num[0] / G_cont.den[0][0]
    den = G_cont.den[0] / G_cont.den[0][0]

    B0_hat  = num[0]
    G0_hat  = joint_alpha * num[0] / den[2]
    WN_hat  = np.sqrt( den[2] )
    ZC_hat  = den[1] / (2 * WN_hat)

    M_hat = 1 / B0_hat
    K_hat = den[2] / B0_hat
    C_hat = den[1] / B0_hat

    err_WN_rel = (WN_hat - target_wn)
    err_ZC_rel = (ZC_hat - target_zc)

    # print(f'Joint {joint_index}')
    # print(f'WN_hat: {WN_hat :.3f} -> err_WN_rel: {err_WN_rel / target_wn * 100:.3f} %')
    # print(f'ZC_hat: {ZC_hat :.3f} -> err_ZC_rel: {err_ZC_rel / target_zc * 100:.3f} %')

    # Compute differentials
    d_beta  = err_WN_rel / WN_hat
    d_delta = err_ZC_rel / ZC_hat

    # Store results
    estimated_params = {
        'G0_hat': G0_hat,
        'ZC_hat': ZC_hat,
        'WN_hat': WN_hat,
        'M_hat' : M_hat,
        'K_hat' : K_hat,
        'C_hat' : C_hat,
    }

    return d_beta, d_delta, estimated_params

def get_improved_muscle_parameters_iteration(
    process_ind          : int,
    optimization_ind     : int,
    optimization_name    : str,
    target_g0            : float,
    target_wn            : float,
    target_zc            : float,
    gains_scalings_alpha : np.ndarray,
    gains_scalings_beta  : np.ndarray,
    gains_scalings_delta : np.ndarray,
    rate_beta            : np.ndarray,
    rate_delta           : np.ndarray,
):
    ''' Get improved muscle parameters '''

    # Params runs (frequency=1 -> phase = time)
    params_runs = get_run_params_for_single_joint_activations(
        waterless            = WATERLESS,
        n_joints_axis        = N_JOINTS_AXIS,
        n_joints_limbs       = N_JOINTS_LIMB,
        tested_joints        = TESTED_JOINTS,
        sig_function         = SWEEP_FUNCTION,
        sig_amplitude        = AMPLITUDE,
        sig_frequency        = 1.0,
        gains_scalings_alpha = gains_scalings_alpha,
        gains_scalings_beta  = gains_scalings_beta,
        gains_scalings_delta = gains_scalings_delta,
    )

    # Keep a copy of the original run parameters
    n_runs           = len(TESTED_JOINTS)
    params_runs_copy = copy.deepcopy(params_runs)

    # Params process
    data_file_tag = 'sweep_activation'

    params_process = {
        'animal_model'            : 'salamandra_v4',
        'simulation_data_file_tag': data_file_tag,
        'gaitflag'                : 0,
        'mech_sim_options'        : get_mech_sim_options(video= False),
        'motor_output_signal_func': get_motor_output_signal,
        'duration'                : DURATION,
        'timestep'                : TIMESTEP,
    }

    # Simulate
    _metrics_runs = simulate_single_net_multi_run_signal_driven_build(
        modname             = MODELS_FARMS[MODEL_NAME][0],
        parsname            = MODELS_FARMS[MODEL_NAME][1],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = optimization_name,
        tag_process         = optimization_ind ,
        save_data           = True,
        plot_figures        = False,
        results_path        = RESULTS_PATH,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = False,
    )

    # Retrieve data
    # print('')
    # print(f'Process {optimization_ind} results:')

    process_folder  = (
        f'{RESULTS_PATH}/'
        f'data/{MODELS_FARMS[MODEL_NAME][0]}_{data_file_tag}_100_{optimization_name}/'
        f'process_{optimization_ind}'
    )

    files_list_runs = [
        f'run_{i}/farms/mechanics_metrics.dill'
        for i in range(n_runs)
    ]

    params_muscles = [
        params_runs_copy[run_index]['mech_sim_options']['muscle_parameters_options'][joint][1]
        for run_index, joint in enumerate(TESTED_JOINTS)
    ]

    performance_iteration = {
        'iteration_ind'       : optimization_ind,
        'process_ind'         : process_ind,
        # Estimated parameters
        'G0_hat'              : np.zeros(n_runs),
        'ZC_hat'              : np.zeros(n_runs),
        'WN_hat'              : np.zeros(n_runs),
        'M_hat'               : np.zeros(n_runs),
        'K_hat'               : np.zeros(n_runs),
        'C_hat'               : np.zeros(n_runs),
        # Target parameters
        'G0_target'           : np.ones(n_runs) * target_g0,
        'ZC_target'           : np.ones(n_runs) * target_zc,
        'WN_target'           : np.ones(n_runs) * target_wn,
        # Scaling factors
        'gains_scalings_alpha_pre': np.copy(gains_scalings_alpha),
        'gains_scalings_beta_pre' : np.copy(gains_scalings_beta),
        'gains_scalings_delta_pre': np.copy(gains_scalings_delta),
        'gains_scalings_alpha'    : np.copy(gains_scalings_alpha),
        'gains_scalings_beta'     : np.copy(gains_scalings_beta),
        'gains_scalings_delta'    : np.copy(gains_scalings_delta),
        # Muscle parameters
        'alpha'  : np.zeros(N_JOINTS_AXIS),
        'beta'   : np.zeros(N_JOINTS_AXIS),
        'delta'  : np.zeros(N_JOINTS_AXIS),
        'gamma'  : np.zeros(N_JOINTS_AXIS),
        'epsilon': np.zeros(N_JOINTS_AXIS),
    }

    for run_index, run_file in enumerate(files_list_runs):
        filename    = f'{process_folder}/{run_file}'
        joint_index = TESTED_JOINTS[run_index]

        d_beta, d_delta, estimated_params = _get_muscle_parameters_joint_error_iteration(
            params_muscles = params_muscles,
            filename       = filename,
            run_index      = run_index,
            target_wn      = target_wn,
            target_zc      = target_zc,
        )

        # Update scaling factors
        factor_beta  = 1 - d_beta  * rate_beta[process_ind]
        factor_delta = 1 - d_delta * rate_delta[process_ind]

        gains_scalings_alpha[joint_index] *= factor_beta
        gains_scalings_beta[joint_index]  *= factor_beta
        gains_scalings_delta[joint_index] *= factor_beta * factor_delta

        # Update parameters
        performance_iteration['G0_hat'][run_index] = estimated_params['G0_hat']
        performance_iteration['ZC_hat'][run_index] = estimated_params['ZC_hat']
        performance_iteration['WN_hat'][run_index] = estimated_params['WN_hat']
        performance_iteration['M_hat'][run_index]  = estimated_params['M_hat']
        performance_iteration['K_hat'][run_index]  = estimated_params['K_hat']
        performance_iteration['C_hat'][run_index]  = estimated_params['C_hat']
        performance_iteration['gains_scalings_alpha'][joint_index] = gains_scalings_alpha[joint_index]
        performance_iteration['gains_scalings_beta'][joint_index]  = gains_scalings_beta[joint_index]
        performance_iteration['gains_scalings_delta'][joint_index] = gains_scalings_delta[joint_index]

        performance_iteration['alpha']   = gains_scalings_alpha * DEFAULT_ALPHA
        performance_iteration['beta']    = gains_scalings_beta  * DEFAULT_BETA
        performance_iteration['delta']   = gains_scalings_delta * DEFAULT_DELTA
        performance_iteration['gamma']   = DEFAULT_GAMMA
        performance_iteration['epsilon'] = DEFAULT_EPSILON

    # Delete previous iteration
    previous_process_folder  = (
        f'{RESULTS_PATH}/'
        f'data/{MODELS_FARMS[MODEL_NAME][0]}_{data_file_tag}_100_{optimization_name}/'
        f'process_{optimization_ind-1}'
    )

    if os.path.exists(previous_process_folder):
        shutil.rmtree(previous_process_folder)

    return performance_iteration

def optimize_muscle_parameters(
    get_muscle_scalings_options: dict,
    optimization_name          : str,
    target_g0                  : float,
    target_wn                  : float,
    target_zc                  : float,
    rate_beta                  : np.ndarray,
    rate_delta                 : np.ndarray,
    start_iteration            : int = 0,
):
    ''' Run the spinal cord model together with the mechanical simulator '''

    optimization_name = (
        f'{optimization_name}_'
        f'FN_{round(target_wn/2/np.pi*1e3)}_'
        f'ZC_{round(target_zc*1e3)}_'
        f'G0_{round(target_g0*1e3)}'
    )

    # Get muscle scaling factors
    (
        gains_scalings_alpha,
        gains_scalings_beta,
        gains_scalings_delta,
    ) = get_muscle_scalings(**get_muscle_scalings_options)

    # Iteration
    for process_ind in range(N_ITERATIONS):
        optimization_ind  = process_ind + start_iteration

        performance_iteration = get_improved_muscle_parameters_iteration(
            process_ind          = process_ind,
            optimization_ind     = optimization_ind,
            optimization_name    = optimization_name,
            target_g0            = target_g0,
            target_wn            = target_wn,
            target_zc            = target_zc,
            gains_scalings_alpha = gains_scalings_alpha,
            gains_scalings_beta  = gains_scalings_beta,
            gains_scalings_delta = gains_scalings_delta,
            rate_beta            = rate_beta,
            rate_delta           = rate_delta,
        )

        # Updae scaling factors
        gains_scalings_alpha = performance_iteration['gains_scalings_alpha']
        gains_scalings_beta  = performance_iteration['gains_scalings_beta']
        gains_scalings_delta = performance_iteration['gains_scalings_delta']

        # Save iteration
        if not SAVE_ITERATIONS:
            continue

        folder_name   = (
            f'experimental_data/salamander_kinematics_muscles/results/{optimization_name}'
        )

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        file_name     = f'{folder_name}/performance_iteration_{optimization_ind}.dill'
        with open(file_name, 'wb') as outfile:
            dill.dump(performance_iteration, outfile)

    if PLOTTING:
        plt.show()

    return

def main():

    # TARGET PARAMETERS
    target_G0 = np.pi/ 4
    # target_Wn_list = np.array( ( 10.0, 12.5, 15.0) ) * 2 * np.pi
    # target_Zc_list = np.array( ( 1.25, 1.50, 1.75, 2.00) )

    target_Wn_list = np.array( ( 11.0) ) * 2 * np.pi
    target_Zc_list = np.array( ( 1.00) )

    # OPTIMIZATION
    rate_beta  = np.linspace(0.3, 0.01, N_ITERATIONS)
    rate_delta = np.linspace(0.3, 0.01, N_ITERATIONS)

    get_muscle_scalings_options = {
        'n_joints_axis'          : N_JOINTS_AXIS,
        'use_original_scalings'  : True,
        'manual_scaling_alpha'   : np.ones(N_JOINTS_AXIS),
        'manual_scaling_beta'    : np.ones(N_JOINTS_AXIS),
        'manual_scaling_delta'   : np.ones(N_JOINTS_AXIS),
    }

    # Define processes
    processes = [
        Process(
            target= optimize_muscle_parameters,
            args = (
                get_muscle_scalings_options,
                'muscle_parameters_optimization',
                target_G0,
                target_Wn,
                target_Zc,
                rate_beta,
                rate_delta,
            )
        )
        for target_Wn in target_Wn_list
        for target_Zc in target_Zc_list
    ]

    n_processes = len(processes)
    n_batch     = 6
    n_batches   = int( np.ceil( n_processes / n_batch) )

    for batch_ind in range(n_batches):
        processes_batch = processes[batch_ind*n_batch : (batch_ind+1)*n_batch]

        # Start processes
        for process in processes_batch:
            process.start()

        # Join processes
        for process in processes_batch:
            process.join()



if __name__ == '__main__':

    # SIMULATION
    RESULTS_PATH = '/data/pazzagli/simulation_results'
    MODEL_NAME   = '4limb_1dof_unweighted'

    DURATION     = 10.0
    TIMESTEP     = 0.001
    PLOTTING     = False
    WATERLESS    = True

    # TOPOLOGY
    N_JOINTS_TRUNK = 4
    N_JOINTS_TAIL  = 4
    N_JOINTS_LIMB  = 4
    N_JOINTS_AXIS  = N_JOINTS_TAIL + N_JOINTS_TRUNK

    # MUSCLE ACTIVATION
    AMPLITUDE = 0.3
    F_MIN     = 0.0
    F_MAX     = 5.0

    SWEEP_FUNCTION = lambda time : np.cos( 2*np.pi * (F_MIN + (F_MAX - F_MIN) / DURATION * time) * time )

    # OPTIMIZATION
    N_ITERATIONS    = 100
    SAVE_ITERATIONS = True
    TESTED_JOINTS   = range(N_JOINTS_AXIS)

    main()