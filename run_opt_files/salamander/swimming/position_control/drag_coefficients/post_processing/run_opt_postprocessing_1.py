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
from run_opt_post_processing.swimming.drag_coefficients import salamander_opt_farms_postprocessing

def main():
    ''' Run the analysis-specific post-processing script '''

    results_path = '/data/pazzagli/simulation_results/{}/optimization/swimming/drag_coefficients'

    folder_names = [
        'net_farms_4limb_1dof_unweighted_position_control_speedbl_071_tailbl_016_OPTIMIZATION',
    ]

    constr_additional = {
        'mech_tail_beat_amp_bl': (0.14, 0.18),
    }

    for folder_name in folder_names:
        salamander_opt_farms_postprocessing.optimization_post_processing(
            folder_name       = folder_name,
            results_path      = results_path.format('data'),
            constr_additional = constr_additional,
            check_constraints = False,
        )

    # User prompt
    snn_optimization_results.save_prompt(
        folder_name  = 'net_farms_4limb_1dof_OPTIMIZATION',
        results_path = results_path.format('images'),
    )

    plt.show()


if __name__ == '__main__':
    main()
