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
    snn_optimization_results,
)

def main():
    ''' Run the analysis-specific post-processing script '''

    results_path = '/data/pazzagli/simulation_results/data/0_analysis/salamander/swimming/closed_loop/feedback_topology_2D'
    folder_name  = 'net_farms_limbless_feedback_topology_feedback_topology_strength_swimming_2dim_100_2023_10_22_21_26_31_ANALYSIS'

    # Get the list of parameters and metrics
    (
        params_processes_list,
        params_runs_processes_list,
        metrics_processes
    ) = snn_simulation_results.get_analysis_results(
        folder_name       = folder_name,
        results_data_path = results_path,
    )


    # Parameters changing over processe
    ex_up = np.linspace(0.0, 0.025, 11)
    ex_dw = np.linspace(0.0, 0.025, 11)
    in_up = np.linspace(0.0, 0.025, 11)
    in_dw = np.linspace(0.0, 0.025, 11)

    # Create dataframe of parameters
    target_metrics = [
        'neur_ptcc_ax',
        'neur_freq_ax',
        'neur_ipl_ax_t',
    ]

    for metric_key in target_metrics:

        metric_val_ex_rel = np.zeros((10, 11, 11))
        metric_val_in_rel = np.zeros((10, 11, 11))

        for prc_ind, metric_val_prc in enumerate( metrics_processes[metric_key] ):

            metric_val_prc_ex = metric_val_prc[:121].reshape(11, 11)
            metric_val_prc_in = metric_val_prc[121:].reshape(11, 11)

            metric_val_ex_rel[prc_ind] = 100 * ( metric_val_prc_ex / metric_val_prc_ex[0, 0] - 1 )
            metric_val_in_rel[prc_ind] = 100 * ( metric_val_prc_in / metric_val_prc_in[0, 0] - 1 )

        metric_val_ex_rel_mean = np.mean(metric_val_ex_rel, axis=0)
        metric_val_in_rel_mean = np.mean(metric_val_in_rel, axis=0)

        # CONTOUR PLOT
        fig  = plt.figure(f'{metric_key} - EX - plot_2D', figsize=(10,5))
        ax   = plt.axes()
        X, Y = np.meshgrid(ex_up, ex_dw)
        contours = ax.contour(X, Y, metric_val_ex_rel_mean.T, 3, colors='black')
        ax.clabel(contours, inline= True, fontsize=8)
        cp   = ax.contourf(X, Y, metric_val_ex_rel_mean.T, cmap='RdBu_r', alpha= 0.5)
        fig.colorbar(cp)
        ax.set_xlabel('ex_up')
        ax.set_ylabel('ex_dw')

        fig  = plt.figure(f'{metric_key} - IN - plot_2D', figsize=(10,5))
        ax   = plt.axes()
        X, Y = np.meshgrid(in_up, in_dw)

        contours = ax.contour(X, Y, metric_val_in_rel_mean.T, 3, colors='black')
        ax.clabel(contours, inline= True, fontsize=8)
        cp   = ax.contourf(X, Y, metric_val_in_rel_mean.T, cmap='RdBu_r', alpha= 0.5)
        fig.colorbar(cp)
        ax.set_xlabel('in_up')
        ax.set_ylabel('in_dw')




    # User prompt
    snn_simulation_results.user_prompt(
        folder_name  = folder_name,
        results_path = results_path,
    )

    # Display
    plt.show()


if __name__ == '__main__':
    main()
