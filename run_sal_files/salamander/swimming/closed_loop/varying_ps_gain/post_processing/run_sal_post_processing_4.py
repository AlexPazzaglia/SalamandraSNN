''' Script to run multiple sensitivity analysis with the desired module '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import matplotlib.pyplot as plt

from network_experiments import snn_sensitivity_analysis_results, snn_utils

def main():
    ''' Run sensitivity analysis '''

    results_path = '/data/pazzagli/simulation_results/data/1_sensitivity_analysis/salamander/swimming/closed_loop/varying_ps_gain'

    # folder_name = 'net_farms_limbless_swimming_cpg_neural_parameters_ps_095_100_2023_10_23_05_06_32_SENSITIVITY'
    # folder_name = 'net_farms_limbless_swimming_cpg_neural_parameters_ps_095_100_2023_10_26_21_31_49_SENSITIVITY'
    # folder_name = 'net_farms_limbless_swimming_cpg_neural_parameters_ps_095_100_2023_10_30_22_01_53_SENSITIVITY'
    folder_name = 'net_farms_limbless_swimming_cpg_neural_parameters_ps_095_100_2023_10_31_23_45_46_SENSITIVITY'

    # Load sensitivity analysis parameters
    sal_analysis_info = snn_sensitivity_analysis_results.load_sensitivity_analysis_parameters(
        folder_name,
        results_path,
    )

    # Load simulation results
    (
        params_processes_list,
        params_runs_processes_list,
        metrics_processes
    ) = snn_sensitivity_analysis_results.get_sensitivity_analysis_results(
        folder_name       = folder_name,
        results_data_path = results_path,
    )

    # Filter nan or zero metrics
    metrics_processes_filt = {
        metric_name : metric_values
        for metric_name, metric_values in metrics_processes.items()
        if not np.all(np.isnan(metric_values)) and np.any( metric_values )
    }

    # Metrics of interest
    sensitivity_metrics_keys = [
        'neur_freq_ax',
        'neur_ptcc_ax',
        'neur_ipl_ax_t',
        'neur_ipl_ax_a',
    ]

    # Compute indices
    (
        sensitivity_indices_1,
        sensitivity_indices_t,
    ) = snn_sensitivity_analysis_results.compute_sensitivity_indices(
        analysis_info     = sal_analysis_info,
        metrics_processes = metrics_processes_filt,
        metrics_keys      = sensitivity_metrics_keys,
    )

    # Plot sensitivity indices
    snn_sensitivity_analysis_results.plot_sensitivity_indices_distribution(
        analysis_info       = sal_analysis_info,
        sensitivity_indices = sensitivity_indices_1,
        figure_tag          = 'S1',
        excluded_pars       = ['tau2', 'delta_w2']
    )
    snn_sensitivity_analysis_results.plot_sensitivity_indices_distribution(
        analysis_info       = sal_analysis_info,
        sensitivity_indices = sensitivity_indices_t,
        figure_tag          = 'ST',
        excluded_pars       = ['tau2', 'delta_w2']
    )

    # User prompt
    save = input(f'Save {folder_name} figures? [Y/n] - ')

    if save in ['y','Y','1']:
        folder_tag = input('Analysis tag: ')
        folder_tag = f'_{folder_tag}' if folder_tag else ''
        snn_utils.save_all_figures(
            f'{folder_name}{folder_tag}',
            results_path = f'{CURRENTDIR}/figures'
        )

    plt.show()

if __name__ == '__main__':
    main()
