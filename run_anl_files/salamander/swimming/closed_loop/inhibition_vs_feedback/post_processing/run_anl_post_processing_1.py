''' Run the analysis-specific post-processing script '''
import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

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

    results_path = '/data/pazzagli/simulation_results/data/0_analysis/salamander/swimming/closed_loop/inhibition_vs_feedback'
    folder_name  = 'net_farms_limbless_stability_vs_inhibition_vs_feedback_100_2023_10_22_20_26_28_ANALYSIS'

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
    # NOTE: 10 processes copies, 40 inhibition amplitudes, 20 ps_gain_ax
    inhibition_amplitude = np.linspace(0, 3.9, 40)
    ps_gain_scaling      = 1 / np.linspace(0.05, 1, 20)

    for _metric_name, metric_pars in studied_metrics.items():

        metric_vals = np.array( metric_pars['values'] ).reshape(40, 10, 20)

        metric_mean = np.mean(metric_vals, axis=1)
        metric_std  = np.std(metric_vals, axis=1)

        # Plots
        analysis_plotting.plot_2d_grid_search(
            metric_name      = metric_pars['name'],
            metric_label     = metric_pars['label'],
            metric_mean      = metric_mean,
            metric_std       = metric_std,
            metric_limits    = metric_pars['limits'],
            par_0_vals       = inhibition_amplitude,
            par_1_vals       = ps_gain_scaling,
            par_0_name       = 'inhibition_amplitude',
            par_0_label      = 'Inhibition strenght [#]',
            par_1_name       = 'ps_gain_scaling',
            par_1_label      = 'PS_gain [#]',
            tag              = '',
            log_scale_metric = False,
            log_scale_p0     = False,
            log_scale_p1     = True,
            invert_colors    = True,
            filter_metric    = False
        )

        analysis_plotting.plot_2d_grid_search(
            metric_name      = metric_pars['name'],
            metric_label     = metric_pars['label'],
            metric_mean      = metric_mean,
            metric_std       = metric_std,
            metric_limits    = metric_pars['limits'],
            par_0_vals       = inhibition_amplitude,
            par_1_vals       = ps_gain_scaling,
            par_0_name       = 'inhibition_amplitude',
            par_0_label      = 'Inhibition strenght [#]',
            par_1_name       = 'ps_gain_scaling',
            par_1_label      = 'PS_gain [#]',
            tag              = '',
            log_scale_metric = False,
            log_scale_p0     = False,
            log_scale_p1     = True,
            invert_colors    = True,
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
