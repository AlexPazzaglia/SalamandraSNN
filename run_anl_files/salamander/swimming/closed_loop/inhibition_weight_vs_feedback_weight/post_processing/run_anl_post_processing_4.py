''' Run the analysis-specific post-processing script '''
import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import re
import numpy as np
import matplotlib.pyplot as plt

from network_experiments import (
    snn_simulation_results,
    snn_utils,
)

from scipy.ndimage import gaussian_filter
from run_anl_files import analysis_plotting

def main():
    ''' Run the analysis-specific post-processing script '''

    results_path = '/data/pazzagli/simulation_results/data/0_analysis/salamander/swimming/closed_loop/inhibition_weight_vs_feedback_weight'
    folder_name  = 'net_farms_limbless_inhibition_weight_ps_weight_inhibition_weight_vs_feedback_weight_zoom_alpha_fraction_th_050_100_2023_11_07_11_03_16_ANALYSIS'

    (
        params_processes_list,
        params_runs_processes_list,
        metrics_processes
    ) = snn_simulation_results.get_analysis_results(
        folder_name       = folder_name,
        results_data_path = results_path,
    )

    # Plot
    studied_metrics = {
        'ptcc_ax'  : {
                'name'  : 'PTCC',
                'label' : 'PTCC [#]',
                'values': np.array( metrics_processes['neur_ptcc_ax'] ),
                'limits': [0, 2],
            },
    }

    # Grid search values
    # NOTE: 20 processes copies, 31 in strengths, 21 ps strengths
    in_weight = np.linspace(0, 1.0, 31)
    ps_weight      = np.linspace(0.0, 5.0, 21)

    for _metric_name, metric_pars in studied_metrics.items():

        metric_vals = np.array( metric_pars['values'] ).reshape(20, 31, 21)

        metric_mean = np.mean(metric_vals, axis=0)
        metric_std  = np.std(metric_vals, axis=0)

        # Plots
        analysis_plotting.plot_2d_grid_search(
            metric_name      = metric_pars['name'],
            metric_label     = metric_pars['label'],
            metric_mean      = metric_mean,
            metric_std       = metric_std,
            metric_limits    = metric_pars['limits'],
            par_0_vals       = in_weight,
            par_1_vals       = ps_weight,
            par_0_name       = 'inhibition_strength',
            par_0_label      = 'Inhibition strenght [#]',
            par_1_name       = 'ps_strength',
            par_1_label      = 'PS strength [#]',
            tag              = '',
            log_scale_metric = False,
            log_scale_p0     = False,
            log_scale_p1     = False,
            invert_colors    = False,
            filter_metric    = False
        )

        analysis_plotting.plot_2d_grid_search(
            metric_name      = metric_pars['name'],
            metric_label     = metric_pars['label'],
            metric_mean      = metric_mean,
            metric_std       = metric_std,
            metric_limits    = metric_pars['limits'],
            par_0_vals       = in_weight,
            par_1_vals       = ps_weight,
            par_0_name       = 'inhibition_strength',
            par_0_label      = 'Inhibition strenght [#]',
            par_1_name       = 'ps_strength',
            par_1_label      = 'PS strength [#]',
            tag              = '',
            log_scale_metric = False,
            log_scale_p0     = False,
            log_scale_p1     = False,
            invert_colors    = False,
            filter_metric    = True
        )

    # User prompt
    save = input('Save figures? [Y/n] - ')

    if save in ['y','Y','1']:
        figure_path = f'{CURRENTDIR}/results/'
        folder_tag  = input('Analysis tag: ')
        folder_tag  = f'_{folder_tag}' if folder_tag else ''
        snn_utils.save_all_figures(
            f'{folder_name}{folder_tag}',
            results_path = figure_path
        )

    plt.show()



if __name__ == '__main__':
    main()
