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

    results_path = '/data/pazzagli/simulation_results/data/0_analysis/salamander/swimming/signal_driven/increasing_frequency'
    folder_name  = 'net_farms_4limb_1dof_unweighted_frequency_effect_100_ANALYSIS'

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
        'speed_fwd_bl' : {
                'values' : np.array( metrics_processes['mech_speed_fwd_bl'][0] ),
                'limits' : [0.3, 1.0],
                'ylabel' : 'Specific speed [bl/s]'
            },
        'tail_amp_bl' : {
                'values' : np.array( metrics_processes['mech_tail_beat_amp_bl'][0] ),
                'limits' : [0.0, 0.2],
                'ylabel' : 'Specific tail amplitude [bl]',
            },
    }

    # Grid search values
    frequency = np.linspace(1.5, 5.0, 15)


    for metric_name, metric_pars in studied_metrics.items():

        metric_vals = np.array( metric_pars['values'] )

        # Plot metric values for each column separately
        plt.figure(f'{metric_name} - plot_1D', figsize=(10,5))

        plt.plot(
            frequency,
            metric_vals,
            marker = 'o',
        )

        # Increase size of text
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

        plt.xlim( frequency[0], frequency[-1] )
        plt.ylim( metric_pars['limits'] )
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(metric_name)
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