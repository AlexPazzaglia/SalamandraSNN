''' Run the analysis-specific post-processing script '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import matplotlib.pyplot as plt
from network_experiments import (
    snn_optimization_results,
)

def main():
    ''' Run the analysis-specific post-processing script '''

    results_path = '/data/pazzagli/simulation_results/{}/optimization/walking/target_speed'
    analysis_names = [
        'net_farms_4limb_1dof_walking_target_speed_100',
        # 'net_farms_4limb_1dof_walking_target_speed_101',
    ]

    folder_names = [
        # [ 'ps0_speed1', 0.01 ],
        # [ 'ps0_speed2', 0.02 ],
        # [ 'ps0_speed3', 0.03 ],
        # [ 'ps4_speed1', 0.01 ],
        # [ 'ps4_speed2', 0.02 ],
        [ 'ps4_speed3', 0.03 ],
    ]

    for analysis_name in analysis_names:
        for folder_name, speed_target in folder_names:
            snn_optimization_results.optimization_post_processing(
                folder_name    = f'{analysis_name}/{folder_name}',
                results_path   = results_path.format('data'),

                range_speed = [ 0.0, speed_target],
                range_cot   = [0.00,        0.15],
            )

    # User prompt
    snn_optimization_results.save_prompt(
        folder_name  = 'net_farms_4limb_1dof_OPTIMIZATION',
        results_path = results_path.format('images'),
    )

    plt.show()


if __name__ == '__main__':
    main()
