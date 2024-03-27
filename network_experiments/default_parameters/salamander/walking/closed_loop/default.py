import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from network_experiments.snn_simulation_setup import get_mech_sim_options
from network_experiments.default_parameters.salamander.walking.walking_default import *

###############################################################################
## CONSTANTS ##################################################################
###############################################################################

MODEL_NAME = '4limb_1dof_unweighted'
MODNAME    = MODELS_FARMS[MODEL_NAME][0]
PARSNAME   = MODELS_FARMS[MODEL_NAME][1]

###############################################################################
## FUNCTIONS ##################################################################
###############################################################################

def get_scaled_ps_gains(
    alpha_fraction_th  : float,
    reference_data_name: str  = REFERENCE_DATA_NAME,
):
    ''' Get the scaled proprioceptive feedback gains '''
    joints_displ_amp = KINEMATICS_DATA_DICT[reference_data_name]['joints_displ_amp']
    return RHEOBASE_CURRENT_PS / (alpha_fraction_th * joints_displ_amp)

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
    # NOTE: Increase stiffness of head and tail to avoid excessive oscillations
    muscle_parameters_options = get_muscle_parameters_options(
        muscle_parameters_tag = muscle_parameters_tag,
        scalings_alpha        = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]),
        scalings_beta         = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]),
        scalings_delta        = np.array([ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0]),
    )

    # Mech sim options
    mech_sim_options = get_mech_sim_options(
        muscle_parameters_options = muscle_parameters_options,
    )

    # Process parameters
    params_process = PARAMS_SIMULATION | {
        'stim_a_off'               : kinematics_data['stim_a_off'],
        'mc_gain_axial'            : kinematics_data['mc_gains'],
        'mech_sim_options'         : mech_sim_options,
    }

    return {
        'muscle_parameters_tag': muscle_parameters_tag,
        'reference_data_name'  : reference_data_name,
        'kinematics_data'      : kinematics_data,
        'modname'              : MODNAME,
        'parsname'             : PARSNAME,
        'params_process'       : params_process,
    }