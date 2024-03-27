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

    results_path = '/data/pazzagli/simulation_results/data/0_analysis/salamander/swimming/open_loop/drive_effect_rostral'
    folder_name  = 'net_openloop_4limb_1dof_unweighted_drive_effect_rostral_100_2023_10_21_21_57_43_ANALYSIS'

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
                'limits' : [-100.0, 3.0],
            },
        'wavelength_ax_a' : {
                'name'   : 'Wavelength axis',
                'label'  : 'Wavelength axis [BL]',
                'values' : np.array( metrics_processes['neur_ipl_ax_a'] ) * 40,
                'limits' : [-100.0, 3.0],
            },
    }

    # ptcc = np.array( metrics_processes['neur_ptcc_ax'] )
    # for metric_pars in studied_metrics.values():
    #     metric_pars['values'][ptcc < 0.5] = np.nan

    # Grid search values
    # NOTE: 30 processes, 51 stim_a_off, 31 stim_h_off
    stim_a_amp = np.linspace(-2, 3, 51) + 5.0
    stim_h_off =  np.linspace(-1.5, 1.5, 31)

    for _metric_name, metric_pars in studied_metrics.items():

        metric_vals = np.array( metric_pars['values'] ).reshape(30, 51, 31)

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
            par_1_vals       = stim_h_off,
            par_0_name       = 'stimulation_amplitude',
            par_0_label      = 'Stimulation amplitude [pA]',
            par_1_name       = 'stimulation offset',
            par_1_label      = 'Stimulation offset [pA]',
            tag              = '',
            log_scale_metric = False,
            log_scale_p0     = False,
            log_scale_p1     = False,
            invert_colors    = False,
            filter_metric    = False,
        )

        analysis_plotting.plot_2d_grid_search(
            metric_name      = metric_pars['name'],
            metric_label     = metric_pars['label'],
            metric_mean      = metric_mean,
            metric_std       = metric_std,
            metric_limits    = metric_pars['limits'],
            par_0_vals       = stim_a_amp,
            par_1_vals       = stim_h_off,
            par_0_name       = 'stimulation_amplitude',
            par_0_label      = 'Stimulation amplitude [pA]',
            par_1_name       = 'stimulation offset',
            par_1_label      = 'Stimulation offset [pA]',
            tag              = '',
            log_scale_metric = False,
            log_scale_p0     = False,
            log_scale_p1     = False,
            invert_colors    = False,
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

    # plt.show()


if __name__ == '__main__':
    main()