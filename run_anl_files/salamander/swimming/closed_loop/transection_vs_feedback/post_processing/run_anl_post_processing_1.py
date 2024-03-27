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

    results_path = '/data/pazzagli/simulation_results/data/0_analysis/salamander/swimming/closed_loop/transection_vs_feedback'
    folder_name  = 'net_farms_limbless_transection_transection_cpg_ps_vs_feedback_100_2023_10_22_20_34_24_ANALYSIS'

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
        # 'freq_ax'  : {
        #         'values' : np.array( metrics_processes['neur_freq_ax'] ),
        #         'limits' : [0, 5],
        #     },
        'ptcc_ax'  : {
                'values' : np.array( metrics_processes['neur_ptcc_ax'] ),
                'limits' : [0, 2],
            },
        # 'wavelength_ax_t' : {
        #         'values' : np.array( metrics_processes['neur_ipl_ax_t'] ) * 40,
        #         'limits' : [-0.4, 2.0],
        #     },
        # 'wavelength_ax_a' : {
        #         'values' : np.array( metrics_processes['neur_ipl_ax_a'] ) * 40,
        #         'limits' : [-0.4, 2.0],
        #     },
    }

    # Grid search values
    # NOTE: 20 processes, 21 transections, 20 ps_gain_ax
    n_transections  = np.arange(21)
    ps_gain_scaling = 1 / np.linspace(0.05, 1, 20)

    colors = pl.cm.jet(np.linspace(0,1,20))[::-1]

    for metric_name, metric_pars in studied_metrics.items():

        metric_vals = np.array( metric_pars['values'] ).reshape(20, 21, 20)
        metric_mean = np.mean(metric_vals, axis=0)
        metric_std  = np.std(metric_vals, axis=0)

        # Plot metric values for each column separately
        plt.figure(f'{metric_name} - plot_1D', figsize=(10,5))

        for ps_ind, ps_val in enumerate(ps_gain_scaling):
            plt.plot(
                n_transections,
                metric_mean[:, ps_ind],
                label = f'ps_scaling = {ps_val:.2f}',
                color = colors[ps_ind],
            )

            plt.fill_between(
                n_transections,
                metric_mean[:, ps_ind] - metric_std[:, ps_ind],
                metric_mean[:, ps_ind] + metric_std[:, ps_ind],
                alpha = 0.2,
                color = colors[ps_ind],
            )

        # Increase size of text
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

        plt.xlim( n_transections[0], n_transections[-1] )
        plt.ylim( metric_pars['limits'] )
        plt.xlabel('Transections [#]')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid()

        # Plot heatmap
        X, Y = np.meshgrid(n_transections, ps_gain_scaling)

        fig = plt.figure(f'{metric_name} - plot_2D', figsize=(10,5))
        ax  = plt.axes()

        contours = ax.contour(X, Y, metric_mean.T, 4, colors='black')
        ax.clabel(contours, inline= True, fontsize=8)

        im = ax.contourf(X, Y, metric_mean.T, 20, cmap='RdBu_r', alpha= 0.5)

        ax.set_xlabel('Transections [#]')
        ax.set_ylabel('ps_gain_ax')
        fig.colorbar(im)


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