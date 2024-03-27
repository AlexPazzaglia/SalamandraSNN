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

    results_path = '/data/pazzagli/simulation_results/data/0_analysis/salamander/swimming/closed_loop/increasing_drive'
    folder_name  = 'net_farms_4limb_1dof_unweighted_drive_effect_100_2023_10_22_16_22_23_ANALYSIS'

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
        'ptcc' : {
                'values' : np.array( metrics_processes['neur_ptcc_ax'][0] ),
                'limits' : [0.0, 2.0],
            },
        'speew_fwd' : {
                'values' : np.array( metrics_processes['mech_speed_fwd_bl'][0] ),
                'limits' : [0.0, 1.0],
            },
        'tail_amp' : {
                'values' : np.array( metrics_processes['mech_tail_beat_amp_bl'][0] ),
                'limits' : [0.0, 0.3],
            },
    }

    # Grid search values
    stim_a_off = np.linspace(-2, 5, 71) + 4.0

    stim_a_off_ind = (stim_a_off >= 3.5) & (stim_a_off <= 6)

    for metric_name, metric_pars in studied_metrics.items():

        metric_vals = np.array( metric_pars['values'] )

        # Plot metric values for each column separately
        plt.figure(f'{metric_name} - plot_1D', figsize=(10,5))

        plt.plot(
            stim_a_off[stim_a_off_ind],
            metric_vals[stim_a_off_ind],
            marker = 'o',
        )

        # Increase size of text
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

        plt.xlim( stim_a_off[stim_a_off_ind][0], stim_a_off[stim_a_off_ind][-1] )
        plt.ylim( metric_pars['limits'] )
        plt.xlabel('Stimulation amplitude [pA]')
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