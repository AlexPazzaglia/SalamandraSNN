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

def main():
    ''' Run the analysis-specific post-processing script '''

    results_path = '/data/pazzagli/simulation_results/data/0_analysis/salamander/swimming/closed_loop/drive_vs_feedback'
    folder_name  = 'net_farms_4limb_1dof_unweighted_drive_effect_vs_feedback_extended_100_2023_10_23_12_27_32_ANALYSIS'

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
                'limits' : [0, 5.5],
            },
        'ptcc_ax'  : {
                'name'   : 'PTCC',
                'label'  : 'PTCC [#]',
                'values' : np.array( metrics_processes['neur_ptcc_ax'] ),
                'limits' : [0, 2],
            },
        'wavelength_ax_t' : {
                'name'   : 'Wavelength trunk',
                'label'  : 'Wavelength trunk [BL]',
                'values' : np.array( metrics_processes['neur_ipl_ax_t'] ) * 40,
                'limits' : [-0.4, 1.5],
            },
        'wavelength_ax_a' : {
                'name'   : 'Wavelength axis',
                'label'  : 'Wavelength axis [BL]',
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
                'limits' : [-0.05, 0.25],
            },
    }

    # ptcc = np.array( metrics_processes['neur_ptcc_ax'] )
    # for metric_pars in studied_metrics.values():
    #     metric_pars['values'][ptcc < 0.5] = np.nan

    # Grid search values
    # NOTE: 30 processes, 21 stim_a_off, 20 ps_gain_ax
    stim_a_amp = np.linspace(-2, 5, 21) + 5.0
    ps_gain_scaling = 1 / np.linspace(0.05, 1, 20)

    # First element greater than 4.0
    stim_a_amp_start = np.where(stim_a_amp > 3.0)[0][0]
    stim_a_amp_end   = np.where(stim_a_amp < 8.0)[0][-1]
    stim_a_amp       = stim_a_amp[stim_a_amp_start:stim_a_amp_end]


    for _metric_name, metric_pars in studied_metrics.items():

        metric_vals = np.array( metric_pars['values'] ).reshape(30, 21, 20)
        metric_vals = metric_vals[:, stim_a_amp_start:stim_a_amp_end, :]

        metric_mean = np.mean(metric_vals, axis=0)
        metric_std  = np.std(metric_vals, axis=0)

        # Plots
        analysis_plotting.plot_2d_grid_search(
            metric_name      = metric_pars['name'],
            metric_label     = metric_pars['label'],
            metric_mean      = metric_mean,
            metric_std       = metric_std,
            metric_limits    = metric_pars['limits'],
            par_0_vals       = stim_a_amp,
            par_1_vals       = ps_gain_scaling,
            par_0_name       = 'stimulation_amplitude',
            par_0_label      = 'Stimulation amplitude [pA]',
            par_1_name       = 'ps_gain_scaling',
            par_1_label      = 'PS_gain [#]',
            tag              = '',
            log_scale_metric = False,
            log_scale_p0     = False,
            log_scale_p1     = True,
            invert_colors    = True,
            filter_metric    = False,
        )

        analysis_plotting.plot_2d_grid_search(
            metric_name      = metric_pars['name'],
            metric_label     = metric_pars['label'],
            metric_mean      = metric_mean,
            metric_std       = metric_std,
            metric_limits    = metric_pars['limits'],
            par_0_vals       = stim_a_amp,
            par_1_vals       = ps_gain_scaling,
            par_0_name       = 'stimulation_amplitude',
            par_0_label      = 'Stimulation amplitude [pA]',
            par_1_name       = 'ps_gain_scaling',
            par_1_label      = 'PS_gain [#]',
            tag              = '',
            log_scale_metric = False,
            log_scale_p0     = False,
            log_scale_p1     = True,
            invert_colors    = True,
            filter_metric    = True,
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