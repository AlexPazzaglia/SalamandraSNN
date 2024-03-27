import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from typing import Callable

from network_experiments.snn_signals_kinematics import get_kinematics_output_signal
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

def create_kinematics_file(
    kinematics_file,
    frequency   : float,
    wave_number : float,
    joints_amps : np.ndarray,
    times       : np.ndarray,
    sig_function: Callable   = None,
):
    ''' Create kinematics file '''

    # Ex: Cosine signal
    # sig_function =  lambda phase : np.cos( 2*np.pi * phase )

    motor_output_signal_pars = {
        'axis': {
            'frequency': frequency,
            'n_copies' : 1,
            'ipl_off'  : [0.0],
            'names'    : ['AXIS'],

            'n_joints': N_JOINTS_AXIS,
            'amp_arr' : np.rad2deg(joints_amps),
            'off_arr' : np.zeros(N_JOINTS_AXIS),
            'ipl_arr' : np.linspace(0.0, wave_number, N_JOINTS_AXIS),
        },
        'limbs': {
            'frequency': 0.0,
            'n_copies' : N_LIMBS,
            'ipl_off'  : [0] * N_LIMBS,
            'names'    : ['LF', 'RF', 'LH', 'RH'],

            'n_joints': N_JOINTS_LIMB,
            'amp_arr' : np.array([-70, 0, 0, 0]),
            'off_arr' : np.zeros(N_JOINTS_LIMB),
            'ipl_arr' : np.zeros(N_JOINTS_LIMB),
        }
    }

    # Create kinematics file
    get_kinematics_output_signal(
        times         = times,
        chains_params = motor_output_signal_pars,
        sig_funcion   = sig_function,
        save_file     = kinematics_file,
    )

    return

###############################################################################
## PARAMETERS #################################################################
###############################################################################

def get_default_parameters(
    reference_data_name  : str = REFERENCE_DATA_NAME,
):
    ''' Get the default parameters for the analysis '''

    kinematics_data = KINEMATICS_DATA_DICT[reference_data_name]

    # Drag coefficients
    drag_coefficients_options = get_drag_coefficients_options(
        drag_coeffs = kinematics_data['drag_coeffs'],
    )

    # Mech sim options
    mech_sim_options = get_mech_sim_options(
        drag_coefficients_options = drag_coefficients_options,
    )

    # Process parameters
    params_process = PARAMS_SIMULATION | {
        'mech_sim_options': mech_sim_options,
    }

    return {
        'reference_data_name'  : reference_data_name,
        'modname'              : MODNAME,
        'parsname'             : PARSNAME,
        'params_process'       : params_process,
        'kinematics_data'      : kinematics_data,
    }