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

def main():
    ''' Run the analysis-specific post-processing script '''

    results_path = '/data/pazzagli/simulation_results/data'
    folder_name  = 'net_openloop_4limb_1dof_unweighted_drive_effect_non_uniform_100_2023_10_05_18_19_52_ANALYSIS'

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
                'values' : np.array( metrics_processes['neur_freq_ax'] ),
                'limits' : [0, 4],
            },
        'ptcc_ax'  : {
                'values' : np.array( metrics_processes['neur_ptcc_ax'] ),
                'limits' : [0, 2],
            },
        'ipl_ax_t' : {
                'values' : np.array( metrics_processes['neur_ipl_ax_t'] ),
                'limits' : [-0.01, 0.05],
            },
        'ipl_ax_a' : {
                'values' : np.array( metrics_processes['neur_ipl_ax_a'] ),
                'limits' : [-0.01, 0.05],
            },
        'wavelength_ax_t' : {
                'values' : np.array( metrics_processes['neur_ipl_ax_t'] ) * 40,
                'limits' : [-0.4, 2.0],
            },
        'wavelength_ax_a' : {
                'values' : np.array( metrics_processes['neur_ipl_ax_a'] ) * 40,
                'limits' : [-0.4, 2.0],
            },
    }

    # Grid search values
    # NOTE: 30 processes, 41 stim_a_off, 11 stim_h_off
    stim_a_amp = np.linspace(-1, 3, 41) + 5.0
    stim_h_off = np.linspace( 0, 1, 11)

    colors = pl.cm.jet(np.linspace(0,1,11))

    for metric_name, metric_pars in studied_metrics.items():

        metric_vals = np.array( metric_pars['values'] ).reshape(30, 41, 11)
        metric_mean = np.mean(metric_vals, axis=0)
        metric_std  = np.std(metric_vals, axis=0)

        # Plot metric values for each column separately
        plt.figure(metric_name, figsize=(10,5))

        for stim_h_ind, stim_h_val in enumerate(stim_h_off):
            plt.plot(
                stim_a_amp,
                metric_mean[:, stim_h_ind],
                label = f'stim_h_off = {stim_h_val:.2f}',
                color = colors[stim_h_ind],
            )

            plt.fill_between(
                stim_a_amp,
                metric_mean[:, stim_h_ind] - metric_std[:, stim_h_ind],
                metric_mean[:, stim_h_ind] + metric_std[:, stim_h_ind],
                alpha = 0.2,
                color = colors[stim_h_ind],
            )

        # Increase size of text
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

        plt.xlim( stim_a_amp[0], stim_a_amp[-1] )
        plt.ylim( metric_pars['limits'] )
        plt.xlabel('Stimulation amplitude')
        plt.ylabel(metric_name)
        plt.title(metric_name, size=20)
        plt.legend()
        plt.grid()


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