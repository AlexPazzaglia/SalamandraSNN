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

    results_path = '/data/pazzagli/simulation_results/data/0_analysis/salamander/swimming/open_loop/increasing_drive_openloop'
    folder_name  = 'net_openloop_4limb_1dof_unweighted_drive_effect_100_2023_09_26_22_24_03_ANALYSIS'

    (
        params_processes,
        params_runs_processes,
        metrics_processes
    ) = snn_simulation_results.get_analysis_results(
        folder_name       = folder_name,
        results_data_path = results_path,
    )


    # Plot
    stim_a_amp = np.array( [ pars['stim_a_off'] for pars in params_runs_processes[0] ] ) + 5.00
    inds_mask  = np.where(stim_a_amp > 0)[0]
    stim_a_amp = stim_a_amp[inds_mask]

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



    for metric_name, metric_pars in studied_metrics.items():

        metric_mean = np.mean(metric_pars['values'], axis=0)[inds_mask]
        metric_std  = np.std(metric_pars['values'], axis=0)[inds_mask]

        # Plot metric
        plt.figure(metric_name, figsize=(10,5))

        plt.plot(
            stim_a_amp,
            metric_mean,
        )
        plt.fill_between(
            stim_a_amp,
            metric_mean - metric_std,
            metric_mean + metric_std,
            alpha = 0.2,
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