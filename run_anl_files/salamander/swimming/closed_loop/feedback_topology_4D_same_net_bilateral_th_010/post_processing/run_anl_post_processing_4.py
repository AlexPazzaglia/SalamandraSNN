''' Run the analysis-specific post-processing script '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from network_experiments import (
    snn_simulation_results,
    snn_utils,
)

from run_anl_files import analysis_plotting

def main(
    metrics_limits = None,
    save_prompt     = False,
    save_figures    = False,
    tag_figures     = False,
    plot_figures    = True,
):
    ''' Run the analysis-specific post-processing script '''

    results_path = '/data/pazzagli/simulation_results/data/0_analysis/salamander/swimming/closed_loop/feedback_topology_4D_bilateral_th_010'
    folder_name  = 'net_farms_limbless_feedback_topology_feedback_topology_strength_swimming_4dim_in_up_ex_dw_100_2023_11_10_00_28_23_ANALYSIS'

    # Get the list of parameters and metrics
    (
        params_processes_list,
        params_runs_processes_list,
        metrics_processes
    ) = snn_simulation_results.get_analysis_results(
        folder_name       = folder_name,
        results_data_path = results_path,
    )

    # Plot
    metrics_limits = metrics_limits if metrics_limits is not None else {}
    studied_metrics = {
        'freq_ax'  : {
                'name'   : 'Frequency',
                'label'  : 'Frequency [Hz]',
                'values' : np.array( metrics_processes['neur_freq_ax'] ),
                'limits' : metrics_limits.get('freq_ax', [2.0, 5.5]),
            },
        'ptcc_ax'  : {
                'name'   : 'PTCC',
                'label'  : 'PTCC [#]',
                'values' : np.array( metrics_processes['neur_ptcc_ax'] ),
                'limits' : metrics_limits.get('ptcc_ax', [0, 2]),
            },
        'wavenumber_ax_t' : {
                'name'   : 'Wave number trunk',
                'label'  : 'Wave number trunk [BL/cycle]',
                'values' : np.array( metrics_processes['neur_ipl_ax_t'] ) * 40,
                'limits' : metrics_limits.get('wavenumber_ax_t', [-0.4, 1.8]),
            },
        'wavenumber_ax_a' : {
                'name'   : 'Wave number axis',
                'label'  : 'Wave number axis [BL/cycle]',
                'values' : np.array( metrics_processes['neur_ipl_ax_a'] ) * 40,
                'limits' : metrics_limits.get('wavenumber_ax_a', [-0.4, 1.8]),
            },
        'speew_fwd' : {
                'name'   : 'Speed',
                'label'  : 'Speed [BL/s]',
                'values' : np.array( metrics_processes['mech_speed_fwd_bl'] ),
                'limits' : metrics_limits.get('speew_fwd', [-0.1, 1.0]),
            },
        'tail_amp' : {
                'name'   : 'Tail amplitude',
                'label'  : 'Tail amplitude [BL]',
                'values' : np.array( metrics_processes['mech_tail_beat_amp_bl'] ),
                'limits' : metrics_limits.get('tail_amp', [-0.05, 0.25]),
            },
        'cot' : {
                'name'   : 'COT',
                'label'  : 'COT [Js/m]',
                'values' : np.array( metrics_processes['mech_cot'] ),
                'limits' : metrics_limits.get('cot', [ 0, 0.1]),
            },
    }

    # Grid search values
    ps_w  = np.linspace(0.0, 5, 21)
    ps_l = np.linspace(0.0, 0.025, 11)

    for _metric_name, metric_pars in studied_metrics.items():

        metric_vals = np.array( metric_pars['values'] ).reshape(20, 21, 11)

        metric_mean = np.mean(metric_vals, axis=0)
        metric_std  = np.std(metric_vals, axis=0)

        # Plots
        analysis_plotting.plot_2d_grid_search(
            metric_name      = metric_pars['name'],
            metric_label     = metric_pars['label'],
            metric_mean      = metric_mean,
            metric_std       = metric_std,
            metric_limits    = metric_pars['limits'],
            par_0_vals       = ps_w,
            par_1_vals       = ps_l,
            par_0_name       = 'PS_weight',
            par_0_label      = 'PS weight [#]',
            par_1_name       = 'PS_range',
            par_1_label      = 'PS range [m]',
            tag              = '',
            log_scale_metric = False,
            log_scale_p0     = False,
            log_scale_p1     = False,
            invert_colors    = False,
            filter_metric    = False,
            grid             = False,
        )

        analysis_plotting.plot_2d_grid_search(
            metric_name      = metric_pars['name'],
            metric_label     = metric_pars['label'],
            metric_mean      = metric_mean,
            metric_std       = metric_std,
            metric_limits    = metric_pars['limits'],
            par_0_vals       = ps_w,
            par_1_vals       = ps_l,
            par_0_name       = 'PS_weight',
            par_0_label      = 'PS weight [#]',
            par_1_name       = 'PS_range',
            par_1_label      = 'PS range [m]',
            tag              = '',
            log_scale_metric = False,
            log_scale_p0     = False,
            log_scale_p1     = False,
            invert_colors    = False,
            filter_metric    = True,
            grid             = False,
        )

    # User prompt
    save = None if not save_prompt else input('Save figures? [Y/n] - ')

    if save_figures or save in ['y','Y','1']:
        figure_path = f'{CURRENTDIR}/results/'

        folder_tag  = '' if not tag_figures else input('Analysis tag: ')
        folder_tag  = f'_{folder_tag}' if folder_tag else ''

        snn_utils.save_all_figures(
            f'{folder_name}{folder_tag}',
            results_path = figure_path
        )

    if plot_figures:
        plt.show()


if __name__ == '__main__':
    main()
