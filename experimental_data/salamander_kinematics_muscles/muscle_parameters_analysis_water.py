''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import json
import dill
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from typing import Any

from network_experiments.snn_simulation_setup import (
    get_params_processes,
    get_mech_sim_options,
)

from network_experiments.snn_signals_neural import get_motor_output_signal

from network_experiments.snn_simulation import simulate_single_net_multi_run_signal_driven_build
from network_experiments.snn_utils import MODELS_FARMS

from experimental_data.salamander_kinematics_muscles.muscle_parameters_optimization import (
    get_run_params_for_multi_joint_activations,
    get_muscle_scalings,
)

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Scaling factors of the muscle parameters
    (
        gains_scalings_alpha,
        gains_scalings_beta,
        gains_scalings_delta,
    ) = get_muscle_scalings(
        n_joints_axis             = N_JOINTS_AXIS,
        use_original_scalings     = USE_ORIGINAL_SCALINGS,
        use_inertial_scalings     = USE_INERTIAL_SCALINGS,
        use_optimization_scalings = USE_OPTIMIZATION_SCALINGS,
        optimization_name         = OPTIMIZATION_NAME,
        optimization_iteration    = OPTIMIZATION_ITERATION,
        inertial_scaling_factor   = GLOBAL_SCALING,
        manual_scaling_alpha      = MANUAL_SCALING_ALPHA,
        manual_scaling_beta       = MANUAL_SCALING_BETA,
        manual_scaling_delta      = MANUAL_SCALING_DELTA,
    )

    # Params runs
    params_runs = get_run_params_for_multi_joint_activations(
        waterless            = WATERLESS,
        n_joints_axis        = N_JOINTS_AXIS,
        n_joints_limbs       = N_JOINTS_LIMB,
        tested_joints        = TESTED_JOINTS,
        sig_function         = ACTIVATION_FUNCTION,
        sig_amplitude        = AMPLITUDE,
        sig_frequency        = FREQUENCY,
        gains_scalings_alpha = gains_scalings_alpha,
        gains_scalings_beta  = gains_scalings_beta,
        gains_scalings_delta = gains_scalings_delta,
        ipl_arr              = np.linspace(0, 1.0, N_JOINTS_AXIS, endpoint= False),
    )

    # Params process
    params_process = {
        'animal_model'            : 'salamandra_v4',
        'simulation_data_file_tag': 'water',
        'gaitflag'                : 0,
        'mech_sim_options'        : get_mech_sim_options(video= True, video_fps=15),
        'motor_output_signal_func': get_motor_output_signal,
        'duration'                : DURATION,
        'timestep'                : TIMESTEP,
    }

    # Simulate
    metrics_runs = simulate_single_net_multi_run_signal_driven_build(
        modname             = MODELS_FARMS[MODEL_NAME][0],
        parsname            = MODELS_FARMS[MODEL_NAME][1],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'scaling_muscle_parameters',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = True,
        results_path        = RESULTS_PATH,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = True,
    )

if __name__ == '__main__':

    # SIMULATION
    RESULTS_PATH = '/data/pazzagli/simulation_results'
    MODEL_NAME   = '4limb_1dof_unweighted'

    DURATION  = 10.0
    TIMESTEP  = 0.001
    WATERLESS = False

    # TOPOLOGY
    N_JOINTS_TRUNK = 4
    N_JOINTS_TAIL  = 4
    N_JOINTS_LIMB  = 4
    N_JOINTS_AXIS  = N_JOINTS_TAIL + N_JOINTS_TRUNK

    # MUSCLE ACTIVATION
    FREQUENCY     = 2.0
    AMPLITUDE     = 0.3
    TESTED_JOINTS = range(N_JOINTS_AXIS)

    ACTIVATION_FUNCTION = lambda phase : np.tanh( 10 * np.cos( 2*np.pi * phase ) )

    # MUSCLE SCALING
    USE_ORIGINAL_SCALINGS     = False

    USE_OPTIMIZATION_SCALINGS = True
    OPTIMIZATION_NAME         = 'muscle_parameters_optimization_FN_6000_ZC_1250_G0_785'
    OPTIMIZATION_ITERATION    = 99

    USE_INERTIAL_SCALINGS     = False
    GLOBAL_SCALING            = 10

    # scaled_muscle_parameters_FN_3000_ZC_1000_G0_785_joint_3_activation_2Hz

    # MANUAL_SCALING_ALPHA = np.array([2.50, 1.50, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
    # MANUAL_SCALING_BETA  = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
    # MANUAL_SCALING_DELTA = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00])

    MANUAL_SCALING_ALPHA = np.ones(N_JOINTS_AXIS)
    MANUAL_SCALING_BETA  = np.ones(N_JOINTS_AXIS)
    MANUAL_SCALING_DELTA = np.ones(N_JOINTS_AXIS)

    main()