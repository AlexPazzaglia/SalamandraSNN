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

    results_path = '/data/pazzagli/simulation_results/{}/optimization/swimming/max_speed'

    folder_names = [
        'net_farms_4limb_1dof_swimming_openloop_max_speed_100',   #
        # 'net_farms_4limb_1dof_swimming_openloop_max_speed_101',
        'net_farms_4limb_1dof_swimming_closedloop_max_speed_100', #
        # 'net_farms_4limb_1dof_swimming_closedloop_max_speed_101',
    ]

    for folder_name in folder_names:
        snn_optimization_results.optimization_post_processing(
            folder_name    = folder_name,
            results_path   = results_path.format('data'),

            range_speed = [ 0.1,  0.55],
            range_cot   = [0.00, 0.04],
        )

    # User prompt
    snn_optimization_results.save_prompt(
        folder_name  = 'net_farms_4limb_1dof_OPTIMIZATION',
        results_path = results_path.format('images'),
    )

    plt.show()


if __name__ == '__main__':
    main()
