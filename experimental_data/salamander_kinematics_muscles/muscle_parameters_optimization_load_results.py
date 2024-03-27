''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import dill
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import pandas as pd

N_JOINTS_AXIS = 8

def load_muscle_parameters_options_from_optimization(
    optimization_name: str,
    folder_name      : str = (
        'experimental_data/salamander_kinematics_muscles/optimized_parameters'
    ),
    scalings_alpha   : np.ndarray = np.ones(N_JOINTS_AXIS),
    scalings_beta    : np.ndarray = np.ones(N_JOINTS_AXIS),
    scalings_delta   : np.ndarray = np.ones(N_JOINTS_AXIS),
):
    ''' Load the muscle parameters '''

    # Load the parameter
    muscle_params_df = pd.read_csv(f'{folder_name}/{optimization_name}')

    # Organize for simulation
    # muscle_parameters_options = [
    #     [['joint_body_0'], {'alpha': 8.540e-05, 'beta': 5.439e-05, 'delta': 2.565e-06 }],
    #     [['joint_body_1'], {'alpha': 0.0004947, 'beta': 0.0003151, 'delta': 1.364e-05 }],
    #     [['joint_body_2'], {'alpha': 0.0011407, 'beta': 0.0007265, 'delta': 2.991e-05 }],
    #     [['joint_body_3'], {'alpha': 0.0011480, 'beta': 0.0007312, 'delta': 2.968e-05 }],
    #     [['joint_body_4'], {'alpha': 0.0010348, 'beta': 0.0006591, 'delta': 2.620e-05 }],
    #     [['joint_body_5'], {'alpha': 0.0004632, 'beta': 0.0002950, 'delta': 1.186e-05 }],
    #     [['joint_body_6'], {'alpha': 9.245e-05, 'beta': 5.888e-05, 'delta': 2.527e-06 }],
    #     [['joint_body_7'], {'alpha': 4.564e-06, 'beta': 2.907e-06, 'delta': 1.382e-07 }],
    # ]

    muscle_parameters_options = [
        [
            [f'joint_body_{i}'],
            {
                'alpha'  : muscle_params_df.loc[i, 'alpha'] * scalings_alpha[i],
                'beta'   : muscle_params_df.loc[i, 'beta']  * scalings_beta[i],
                'delta'  : muscle_params_df.loc[i, 'delta'] * scalings_delta[i],
                'gamma'  : 1.0,
                'epsilon': 0,
            }
        ]
        for i in range(N_JOINTS_AXIS)
    ]

    return muscle_parameters_options

if __name__ == '__main__':

    optimization_name = 'muscle_parameters_optimization'

    target_G0 = np.pi/ 4
    target_Wn = 6.00 * 2 * np.pi
    target_Zc = 1.00

    index_iteration = 99

    original_alpha = 3.14e-4
    original_beta  = 2.00e-4
    original_delta = 10.0e-6

    optimization_name = (
        f'{optimization_name}_'
        f'FN_{round(target_Wn/2/np.pi*1e3)}_'
        f'ZC_{round(target_Zc*1e3)}_'
        f'G0_{round(target_G0*1e3)}_'
        f'gen_{index_iteration}.csv'
    )

    load_muscle_parameters_options_from_optimization(
        optimization_name = optimization_name,
    )