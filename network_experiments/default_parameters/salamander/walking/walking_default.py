import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from network_experiments.snn_utils import MODELS_FARMS, MODELS_OPENLOOP

from typing import Union
from brian2.units.allunits import second, msecond

from experimental_data.salamander_kinematics_muscles. \
    muscle_parameters_optimization_load_results import (
    load_muscle_parameters_options_from_optimization
)

###############################################################################
## CONSTANTS ##################################################################
###############################################################################

### MODEL DATA
MODEL_NAME     = '4limb_1dof_unweighted'
N_JOINTS_TRUNK = 4
N_JOINTS_TAIL  = 4
N_JOINTS_LIMB  = 4
N_LIMBS        = 4
N_JOINTS_AXIS  = N_JOINTS_TAIL + N_JOINTS_TRUNK
N_JOINTS_LIMBS = N_JOINTS_LIMB * N_LIMBS
N_JOINTS       = N_JOINTS_AXIS + N_JOINTS_LIMBS

POINTS_POSITIONS_MODEL = np.array(
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
        0.1000,
    ]
)

LENGTH_AXIS_MODEL      = POINTS_POSITIONS_MODEL[-1]
JOINTS_POSITIONS_MODEL = POINTS_POSITIONS_MODEL[1:-1]
LENGTHS_LINKS_MODEL    = np.diff(POINTS_POSITIONS_MODEL)

POINTS_POSITIONS_R_MODEL = POINTS_POSITIONS_MODEL / LENGTH_AXIS_MODEL
JOINTS_POSITIONS_R_MODEL = JOINTS_POSITIONS_MODEL / LENGTH_AXIS_MODEL
LENGTHS_LINKS_R_MODEL    = LENGTHS_LINKS_MODEL / LENGTH_AXIS_MODEL

### SIMULATION PARAMETERS
PARAMS_SIMULATION = {
    'animal_model'             : 'salamandra_v4',
    'timestep'                 : 1 * msecond,
    'duration'                 : 10 * second,
    'verboserun'               : False,
    'load_connectivity_indices': False,
    'set_seed'                 : True,
    'seed_value'               : 3678586,
    'gaitflag'                 : 1,
    'ps_gain_axial'            : 0.0,
    'stim_a_off'               : 0.0,
    'stim_l_off'               : 0.0,
}

### FEEDBACK
RHEOBASE_CURRENT_PS = 50.0

### MUSCLE PARAMETERS
MUSCLE_PARAMETERS_TAG = 'FN_15000_ZC_1000_G0_785_gen_99'

### REFERENCE DATA
REFERENCE_DATA_NAME  = 'karakasiliotis'

### KINEMATICS DATA
KINEMATICS_DATA_DICT = {

    # From Karakasiliotis et al. 2016
    'karakasiliotis' : {
        'stim_a_off'    : 0.0,

        # TBD
        'frequency'     : None,
        'wave_number_ax': None,
        'speed_fwd_bl'  : None,
        'tail_beat_bl'  : None,

        'joints_ipls'   : None,

        # Joints and points positions in the tracking data
        'joints_pos' : None,
        'points_pos' : None,

        # Joints and links amplitudes (mapped to the model) (TBD)
        'joints_displ_amp': np.deg2rad(
            [ 10.833, 13.403, 12.954, 12.806, 23.161, 31.773, 28.020, 34.042,]
        ),
        'links_displ_amp': np.interp(
            x  = POINTS_POSITIONS_R_MODEL,
            xp = [   0.0,   0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7,   0.8,   0.9,   1.0 ],
            fp = [ 0.070, 0.070, 0.070, 0.070, 0.050, 0.030, 0.050, 0.062, 0.050, 0.025, 0.025 ],
        ),

        # Signal-driven control (TBD)
        'joints_signals_amps' : None,

        # Closed-loop control (TBD)
        'mc_gains' : np.array(
            [ 1.1437, 0.8992, 0.8511, 0.8362, 1.5248, 2.0978, 1.9653, 2.5621 ]
        ),
    },

}

###############################################################################
## FUNCTIONS ##################################################################
###############################################################################

def get_position_control_options(
    gains_p : Union[float, list, np.ndarray],
    gains_d : Union[float, list, np.ndarray],
) -> list:
    ''' Get the drag coefficients '''

    if isinstance(gains_p, (float, int)):
        gains_p = np.ones(N_JOINTS_AXIS) * gains_p
    if isinstance(gains_d, (float, int)):
        gains_d = np.ones(N_JOINTS_AXIS) * gains_d

    position_control_gains = [
        [
            [f'joint_body_{link}'],
            [
                gains_p[link],
                gains_d[link],
            ]
        ]
        for link in range(N_JOINTS_AXIS)
    ]

    return position_control_gains

def get_muscle_parameters_options(
    muscle_parameters_tag: str,
    scalings_alpha   : np.ndarray = np.ones(N_JOINTS_AXIS),
    scalings_beta    : np.ndarray = np.ones(N_JOINTS_AXIS),
    scalings_delta   : np.ndarray = np.ones(N_JOINTS_AXIS),
):
    ''' Get the muscle parameters '''
    muscle_parameters_options = load_muscle_parameters_options_from_optimization(
        optimization_name = f'muscle_parameters_optimization_{muscle_parameters_tag}.csv',
        scalings_alpha    = scalings_alpha,
        scalings_beta     = scalings_beta,
        scalings_delta    = scalings_delta,
    )
    return muscle_parameters_options


