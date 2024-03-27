''' Run the analysis-specific post-processing script '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import importlib
import matplotlib.pyplot as plt

def main():
    ''' Run the analysis-specific post-processing script '''

    # Plot
    metrics_limits = {
        'freq_ax'        : [ 3.00, 4.50],
        'ptcc_ax'        : [ 0.00, 2.00],
        'wavenumber_ax_t': [-0.20, 1.70],
        'wavenumber_ax_a': [-0.20, 1.70],
        'speew_fwd'      : [-0.10, 1.00],
        'tail_amp'       : [-0.05, 0.25],
        'cot'            : [ 0.00, 0.10],
    }

    for anl_ind in [1, 2, 3, 4]:
        # Import the analysis-specific post-processing script
        anl_post_processing = importlib.import_module(
            'run_anl_files.salamander.swimming.closed_loop.'
            'feedback_topology_4D_same_net_th_010.'
            'post_processing.run_anl_post_processing_' + str(anl_ind)
        )

        # Run the analysis-specific post-processing script
        anl_post_processing.main(
            metrics_limits = metrics_limits,
            save_prompt    = False,
            save_figures   = True,
            tag_figures    = False,
            plot_figures   = False,
        )

        plt.close('all')


if __name__ == '__main__':
    main()
