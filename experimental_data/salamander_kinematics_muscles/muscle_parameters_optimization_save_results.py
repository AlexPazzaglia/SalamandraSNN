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

def save_muscle_parameters_from_single_optimization(
    optimization_name: str,
    target_g0        : float,
    target_wn        : float,
    target_zc        : float,
    index_iteration  : int,
    original_alpha   : float,
    original_beta    : float,
    original_delta   : float,
):
    ''' Save the muscle parameters '''

    optimization_name = (
        f'{optimization_name}_'
        f'FN_{round(target_wn/2/np.pi*1e3)}_'
        f'ZC_{round(target_zc*1e3)}_'
        f'G0_{round(target_g0*1e3)}'
    )

    # Load the optimization results
    folder_name = f'experimental_data/salamander_kinematics_muscles/results/{optimization_name}'
    file_name   = f'{folder_name}/performance_iteration_{index_iteration}.dill'

    with open(file_name, 'rb') as file:
        performance_iteration = dill.load(file)

    # Extract the parameters from the optimization results
    muscle_params = {
        'alpha': performance_iteration['gains_scalings_alpha'] * original_alpha,
        'beta' : performance_iteration['gains_scalings_beta']  * original_beta,
        'delta': performance_iteration['gains_scalings_delta'] * original_delta,
    }

    # Save the parameters
    muscle_params_file_name = (
        'experimental_data/salamander_kinematics_muscles/optimized_parameters/'
        f'{optimization_name}_gen_{index_iteration}.csv'
    )

    muscle_params_df = pd.DataFrame(muscle_params)
    muscle_params_df.to_csv(muscle_params_file_name)

    return

def save_muscle_parameters_from_multiple_optimization():
    ''' Save the muscle parameters '''

    # Parameters of the target optimizations
    optimization_name = 'muscle_parameters_optimization'

    target_G0 = np.pi/ 4

    # target_Wn_list = np.array( ( 10.0, 12.5, 15.0, 17.5, 20.0) ) * 2 * np.pi
    # target_Zc_list = np.array( (1.00,) )

    target_Wn_list = np.array( ( 10.0, 12.5, 15.0) ) * 2 * np.pi
    target_Zc_list = np.array( ( 1.25, 1.50, 1.75, 2.00) )

    index_iteration = 99

    original_alpha = 3.14e-4
    original_beta  = 2.00e-4
    original_delta = 10.0e-6

    # Save the parameters for each target optimization
    for target_Wn in target_Wn_list:
        for target_Zc in target_Zc_list:
            save_muscle_parameters_from_single_optimization(
                optimization_name = optimization_name,
                target_g0         = target_G0,
                target_wn         = target_Wn,
                target_zc         = target_Zc,
                index_iteration   = index_iteration,
                original_alpha    = original_alpha,
                original_beta     = original_beta,
                original_delta    = original_delta,
            )


if __name__ == '__main__':
    save_muscle_parameters_from_multiple_optimization()