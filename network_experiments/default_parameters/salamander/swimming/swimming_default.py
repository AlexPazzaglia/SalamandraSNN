import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from typing import Union
from brian2.units.allunits import second, msecond

from network_experiments.snn_utils import MODELS_FARMS, MODELS_OPENLOOP

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
    'gaitflag'                 : 0,
    'ps_gain_axial'            : 0.0,
    'stim_a_off'               : 0.7,
}

### FEEDBACK
RHEOBASE_CURRENT_PS = 50.0

### MUSCLE PARAMETERS
# MUSCLE_PARAMETERS_TAG = 'FN_8000_ZC_1000_G0_785_gen_99'
MUSCLE_PARAMETERS_TAG = 'FN_15000_ZC_1000_G0_785_gen_99'

### REFERENCE DATA
REFERENCE_DATA_NAME = 'karakasiliotis'

### KINEMATICS DATA
KINEMATICS_DATA_DICT = {

    # From Karakasiliotis et al. 2016
    'karakasiliotis' : {
        'stim_a_off'    : 0.7,
        'frequency'     : 3.17,

        # 'wave_number_ax': 1.120933521923621
        # 'wave_number_ax': sum(joints_ipls) / range(joints_pos) * range(joints_pos_m)
        # 'wave_number_ax': 0.87175 / (0.8888 - 0.1111) * (0.90 - 0.15)
        'wave_number_ax': 0.840,
        'speed_fwd_bl'  : 0.714,
        'tail_beat_bl'  : 0.130,

        'joints_ipls'   : np.array(
            [ 0.00000, 0.19337, 0.08242, 0.08242, 0.10778, 0.14899, 0.13948, 0.11729,]
        ),

        # Joints and points positions in the tracking data
        'joints_pos' : np.array(
            [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
        'points_pos' : np.array(
            [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),

        # Joints and links amplitudes (mapped to the model)
        'joints_displ_amp': np.deg2rad(
            [ 10.833, 13.403, 12.954, 12.806, 23.161, 31.773, 28.020, 34.042,]
        ),
        'links_displ_amp': np.interp(
            x  = POINTS_POSITIONS_R_MODEL,
            xp = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            fp = [ 0.040, 0.025, 0.022, 0.035, 0.043, 0.050, 0.065, 0.080, 0.095, 0.110, 0.130 ],
        ),

        # Drag coefficients
        'drag_coeffs' : np.array(
            [
                [ 0.01728, 0.00295, 0.00324, 0.00318, 0.00212, 0.00177, 0.00130, 0.00097, 0.00097 ],
                [ 0.01925, 0.02000, 0.02200, 0.02400, 0.02100, 0.01800, 0.01650, 0.01650, 0.00825 ],
                [ 0.02200, 0.02500, 0.02750, 0.02700, 0.01800, 0.01500, 0.01100, 0.00825, 0.00825 ],
            ]
        ).T,

        # Signal-driven control
        'joints_signals_amps' : np.array(
            [0.16111, 0.20262, 0.19146, 0.18708, 0.38577, 0.60331, 0.48243, 0.66150 ]
        ),

        # Closed-loop control
        'mc_gains' : np.array(
            [ 1.1437, 0.8992, 0.8511, 0.8362, 1.5248, 2.0978, 1.9653, 2.5621 ]
        ),
    },

    # From Dimitri Ryczko's Lab, 2023
    'canada' : {
        'stim_a_off'    : 0.7, # 1.2
        'frequency'     : 3.64,

        # 'wave_number_ax': 1.0286940198444623
        # 'wave_number_ax': sum(joints_ipls) / range(joints_pos) * range(joints_pos_m)
        # 'wave_number_ax': 0.7672 / (0.8688 - 0.1230) * (0.90 - 0.15)
        'wave_number_ax': 0.7715,
        'speed_fwd_bl'  : 1.6841,
        'tail_beat_bl'  : 0.10631,

        # Joints and points positions in the tracking data
        'joints_pos' : np.array(
            [ 0.1230, 0.1911, 0.2757, 0.3595, 0.4437, 0.5049, 0.6060, 0.7363, 0.8688,]
        ),
        'points_pos' : np.array(
            [ 0.0, 0.1230, 0.1911, 0.2757, 0.3595, 0.4437, 0.5049, 0.6060, 0.7363, 0.8688, 1.0]
        ),

        # Joints and links amplitudes (mapped to the model)
        'joints_displ_amp': np.deg2rad(
            [ 4.012, 8.443, 11.123, 16.714, 20.775, 21.345, 22.279, 20.820,]
        ),
        'links_displ_amp': np.interp(
            x  = POINTS_POSITIONS_R_MODEL,
            xp = [ 0.0, 0.1230, 0.1911, 0.2757, 0.3595, 0.4437, 0.5049, 0.6060, 0.7363, 0.8688, 1.0],
            fp = [ 0.035, 0.017, 0.015, 0.023, 0.028, 0.039, 0.046, 0.061, 0.072, 0.081, 0.106,],
        ),

        # Drag coefficients
        # NOTE: Computed by matching the links displacements and speed_fwd_bl when imposing
        #       the same joints displacements and frequency as in the tracking data (signal-driven control)
        'drag_coeffs' : np.array(
            [
                [0.1876  , 0.      , 0.      , 0.      , 0.      , 0.      ,0.      , 0.      , 0.      ],
                [3.5175  , 3.36    , 3.714375, 3.603   , 3.534375, 3.4125  ,1.26    , 1.185   , 0.7875  ],
                [5.025   , 4.8     , 5.30625 , 5.146875, 2.473875, 2.38875 ,0.882   , 0.8295  , 0.55125 ]
            ]
        ).T,

        # Signal-driven control
        # NOTE: The ones to match the joints displacements at the reference frequency and wave number
        'joints_signals_amps' : np.array(
            [ 0.07329, 0.13583, 0.17209, 0.27240, 0.34729, 0.34922, 0.36641, 0.34336 ]
        ),

        # Closed-loop control
        # NOTE: Found by matching 1.1 * joints_displ_amp at a frequency of 3.4 Hz (stim_a_off = 1.0)
        'mc_gains' : np.array(
            [0.56809, 0.75710, 1.00703, 1.41958, 1.60032, 1.41649, 1.24181, 1.63056]
        ),

    }
}

###############################################################################
## FUNCTIONS ##################################################################
###############################################################################

def get_drag_coefficients_options(
    drag_coeffs        : np.ndarray,
    scalings_x         : np.ndarray = np.ones(N_JOINTS_AXIS+1),
    scalings_y         : np.ndarray = np.ones(N_JOINTS_AXIS+1),
    scalings_z         : np.ndarray = np.ones(N_JOINTS_AXIS+1),
) -> list:
    ''' Get the drag coefficients '''

    drag_coeffs[:, 0] *= scalings_x
    drag_coeffs[:, 1] *= scalings_y
    drag_coeffs[:, 2] *= scalings_z

    drag_coefficients_options = [
        [
            [f'link_body_{link}'],
            - drag_coeffs[link],
        ]
        for link in range(N_JOINTS_AXIS+1)
    ]

    return drag_coefficients_options

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


