import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from typing import Callable

from network_experiments.snn_signals_neural import get_motor_output_signal
from network_experiments.snn_simulation_setup import get_mech_sim_options
from network_experiments.default_parameters.salamander.swimming.swimming_default import *

###############################################################################
## CONSTANTS ##################################################################
###############################################################################

MODEL_NAME = '4limb_1dof_unweighted'
MODNAME    = MODELS_FARMS[MODEL_NAME][0]
PARSNAME   = MODELS_FARMS[MODEL_NAME][1]

###############################################################################
## FUNCTIONS ##################################################################
###############################################################################

def get_gait_params(
    frequency   : float,
    wave_number : float,
    joints_amps : np.ndarray,
    sig_function: Callable   = None,
):
    ''' Get the gait parameters '''

    # Ex: Square signal
    # sig_function = lambda phase : np.tanh( 20 * np.cos( 2*np.pi * phase ) )

    if sig_function is None:
        sig_function = lambda phase : np.cos( 2*np.pi * phase )

    params_gait = {
        'sig_function' : sig_function,

        'axis': {
            'frequency': frequency,
            'n_copies' : 1,
            'ipl_off'  : [0.0],
            'names'    : ['AXIS'],

            'n_joints': N_JOINTS_AXIS,
            'ipl_arr' : np.linspace(0.0, wave_number, N_JOINTS_AXIS),

            'amp_arr' : joints_amps,
            'off_arr' : np.zeros(N_JOINTS_AXIS),
        },

        'limbs': {
            'frequency': 0,
            'n_copies' : 4,
            'ipl_off'  : [0.0, 0.0, 0.0, 0.0],
            'names'    : ['LF', 'RF', 'LH', 'RH'],

            'n_joints': N_JOINTS_LIMB,
            'ipl_arr' : np.zeros(N_JOINTS_LIMB),
            'amp_arr' : np.zeros(N_JOINTS_LIMB),
            'off_arr' : np.zeros(N_JOINTS_LIMB),
        },
    }

    return params_gait

###############################################################################
## PARAMETERS #################################################################
###############################################################################

def get_default_parameters(
    muscle_parameters_tag: str = MUSCLE_PARAMETERS_TAG,
    reference_data_name  : str = REFERENCE_DATA_NAME,
):
    ''' Get the default parameters for the analysis '''

    kinematics_data = KINEMATICS_DATA_DICT[reference_data_name]

    # Muscle parameters
    # NOTE: Increase stiffness of head and tail
    muscle_parameters_options = get_muscle_parameters_options(
        muscle_parameters_tag = muscle_parameters_tag,
        scalings_alpha        = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]),
        scalings_beta         = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]),
        scalings_delta        = np.array([ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0]),
    )

    # Drag coefficients
    drag_coefficients_options = get_drag_coefficients_options(
        drag_coeffs = kinematics_data['drag_coeffs'],
    )

    # Mech sim options
    mech_sim_options = get_mech_sim_options(
        muscle_parameters_options = muscle_parameters_options,
        drag_coefficients_options = drag_coefficients_options,
    )

    # Process parameters
    params_process = PARAMS_SIMULATION | {
        'mech_sim_options'         : mech_sim_options,
        'motor_output_signal_func' : get_motor_output_signal,
    }

    return {
        'muscle_parameters_tag': muscle_parameters_tag,
        'modname'              : MODNAME,
        'parsname'             : PARSNAME,
        'params_process'       : params_process,
        'kinematics_data'      : kinematics_data,
    }