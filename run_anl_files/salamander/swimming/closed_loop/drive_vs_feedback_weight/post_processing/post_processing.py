''' Run the analysis-specific post-processing script '''
# Copy taken from run_analysis_post_processing file

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from network_experiments import (
    snn_simulation_results,
    snn_utils,
)

from run_anl_files import analysis_plotting

from scipy.ndimage import gaussian_filter

def run_post_processing(
    folder_name  : str,
    n_processes  : int = 20,
):
    ''' Run the analysis-specific post-processing script '''

    results_path = '/data/pazzagli/simulation_results/data/0_analysis/salamander/swimming/closed_loop/drive_vs_feedback_weight'

    (
        params_processes,
        params_runs_processes,
        metrics_processes
    ) = snn_simulation_results.get_analysis_results(
        folder_name       = folder_name,
        results_data_path = results_path,
    )

    # Plot
    studied_metrics = {
        'freq_ax'  : {
                'name'   : 'Frequency',
                'label'  : 'Frequency [Hz]',
                'values' : np.array( metrics_processes['neur_freq_ax'] ),
                'limits' : [0.0, 5.0],
            },
        'ptcc_ax'  : {
                'name'   : 'PTCC',
                'label'  : 'PTCC [#]',
                'values' : np.array( metrics_processes['neur_ptcc_ax'] ),
                'limits' : [0.0, 2.00],
            },
        # 'wavenumber_ax_t' : {
        #         'name'   : 'Wave number trunk',
        #         'label'  : 'Wave number trunk [BL/cycle]',
        #         'values' : np.array( metrics_processes['neur_ipl_ax_t'] ) * 40,
        #         'limits' : [-0.4, 1.5],
        #     },
        'wavenumber_ax_a' : {
                'name'   : 'Wave number axis',
                'label'  : 'Wave number axis [BL/cycle]',
                'values' : np.array( metrics_processes['neur_ipl_ax_a'] ) * 40,
                'limits' : [-0.4, 1.5],
            },
        'speew_fwd' : {
                'name'   : 'Speed',
                'label'  : 'Speed [BL/s]',
                'values' : np.array( metrics_processes['mech_speed_fwd_bl'] ),
                'limits' : [-0.1, 1.0],
            },
        'tail_amp' : {
                'name'   : 'Tail amplitude',
                'label'  : 'Tail amplitude [BL]',
                'values' : np.array( metrics_processes['mech_tail_beat_amp_bl'] ),
                'limits' : [ 0.00, 0.20],
            },
        'stride_len' : {
                'name'   : 'Stride length',
                'label'  : 'Stride length [BL]',
                'values' : np.array( metrics_processes['mech_stride_len_bl'] ),
                'limits' : [ 0.00, 0.25],
            },
        'cot' : {
                'name'   : 'COT',
                'label'  : 'COT [Js/m]',
                'values' : np.array( metrics_processes['mech_cot'] ),
                'limits' : [ 0.000, 0.100],
            },
    }

    # Grid search values
    stim_a_amp = 5.0 + np.unique(
        [
            pars_run['stim_a_off']
            for params_runs_process in params_runs_processes
            for pars_run in params_runs_process
        ]
    )
    ps_weight  = np.unique(
        [
            pars_run['ps_weight']
            for params_runs_process in params_runs_processes
            for pars_run in params_runs_process
        ]
    )

    stim_a_amp = np.sort(stim_a_amp)
    ps_weight  = np.sort(ps_weight)

    n_stim_a_amp  = len(stim_a_amp)
    n_ps_weight   = len(ps_weight)
    metrics_shape = (n_processes, n_stim_a_amp, n_ps_weight)

    # First element greater than 4.0
    stim_a_amp_start = np.where(stim_a_amp > 3.0)[0][0]
    stim_a_amp_end   = np.where(stim_a_amp < 10.0)[0][-1]
    stim_a_amp       = stim_a_amp[stim_a_amp_start:stim_a_amp_end]

    # PTCC MASK
    ptcc           = np.array( metrics_processes['neur_ptcc_ax'] ).reshape(metrics_shape)
    ptcc           = ptcc[:, stim_a_amp_start:stim_a_amp_end, :]
    ptcc_mean      = np.nanmean(ptcc, axis=0)
    ptcc_mean_filt = gaussian_filter(ptcc_mean, sigma=1.0)
    ptcc_mask      = ptcc_mean_filt > 0.00

    for _metric_name, metric_pars in studied_metrics.items():

        print(f'Plotting {_metric_name}...')

        metric_vals = np.array( metric_pars['values'] ).reshape(metrics_shape)
        metric_vals = metric_vals[:, stim_a_amp_start:stim_a_amp_end, :]

        # Exclude outliers
        val0 = metric_pars['limits'][0]
        val1 = metric_pars['limits'][1]
        valr = val1 - val0

        metric_vals[metric_vals < val0] = np.nan
        metric_vals[metric_vals > val1] = np.nan

        metric_mean = np.nanmean(metric_vals, axis=0)
        metric_std  = np.nanstd(metric_vals, axis=0)

        # # Plots
        # analysis_plotting.plot_2d_grid_search(
        #     metric_name      = metric_pars['name'],
        #     metric_label     = metric_pars['label'],
        #     metric_mean      = metric_mean,
        #     metric_std       = metric_std,
        #     metric_limits    = metric_pars['limits'],
        #     par_0_vals       = stim_a_amp,
        #     par_1_vals       = ps_weight,
        #     par_0_name       = 'stimulation_amplitude',
        #     par_0_label      = 'Stimulation amplitude [pA]',
        #     par_1_name       = 'ps_weight',
        #     par_1_label      = 'PS_weight [#]',
        #     tag              = '',
        #     log_scale_metric = False,
        #     log_scale_p0     = False,
        #     log_scale_p1     = False,
        #     invert_colors    = False,
        #     filter_metric    = False,
        #     metric_mask      = ptcc_mask,
        # )

        analysis_plotting.plot_2d_grid_search(
            metric_name      = metric_pars['name'],
            metric_label     = metric_pars['label'],
            metric_mean      = metric_mean,
            metric_std       = metric_std,
            metric_limits    = metric_pars['limits'],
            par_0_vals       = stim_a_amp,
            par_1_vals       = ps_weight,
            par_0_name       = 'stimulation_amplitude',
            par_0_label      = 'Stimulation amplitude [pA]',
            par_1_name       = 'ps_weight',
            par_1_label      = 'PS_weight [#]',
            tag              = '',
            log_scale_metric = False,
            log_scale_p0     = False,
            log_scale_p1     = False,
            invert_colors    = False,
            filter_metric    = True,
            metric_mask      = ptcc_mask,
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
