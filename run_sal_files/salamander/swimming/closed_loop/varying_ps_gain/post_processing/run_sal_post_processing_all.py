''' Script to run multiple sensitivity analysis with the desired module '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from network_experiments import snn_sensitivity_analysis_results, snn_utils


def plot_sensitivity_indices(
    sens_indices,
    metric_label,
    sens_indices_tag,
):
    ''' Plot sensitivity indices '''

    # Select parameters columns
    selected_columns = sens_indices.columns[3:]

    # Reshape for boxplot
    s0_vals_melted = sens_indices.melt(
        id_vars    = ['PS_gain', 'repetition'],
        value_vars = selected_columns,
        var_name   = 'variable',
        value_name = 'value'
    )

    # Set the style of seaborn
    sns.set(style="whitegrid")

    # Create a boxplot for each variable grouped by PS_gain
    colormap = sns.color_palette("RdBu_r", n_colors=len(sens_indices['PS_gain'].unique()))

    plt.figure(f'{sens_indices_tag} - {metric_label}', figsize=(16, 10))
    sns.boxplot(
        data       = s0_vals_melted,
        x          = 'variable',
        y          = 'value',
        hue        = 'PS_gain',
        palette    = colormap,
        width      = 0.7,
        fliersize  = 0,
        linewidth  = 1.5,
        showfliers = False,
    )

    # Set labels and title
    plt.title(metric_label, fontsize=20)
    plt.xlabel('')
    plt.ylabel('Sensitivity Index', fontsize=15)

    # Show the plot
    plt.xticks(rotation=45, ha='right', fontsize=15)

    # Force legend position just outside the plot
    plt.legend(
        title          = 'PS gain',
        bbox_to_anchor = (1.01, 1),
        loc            = 'upper left',
        borderaxespad  = 0.,
        fontsize       = 15,
    )

    plt.tight_layout()

def main():
    ''' Run sensitivity analysis '''

    results_path = '/data/pazzagli/simulation_results/data/1_sensitivity_analysis/salamander/swimming/closed_loop/varying_ps_gain'

    analysis_name = 'net_farms_limbless_swimming_cpg_neural_parameters_ps_100_2023_10_26_SENSITIVITY'

    # folder_names = [
    #     'net_farms_limbless_swimming_cpg_neural_parameters_ps_110_100_2023_10_26_21_31_48_SENSITIVITY',
    #     'net_farms_limbless_swimming_cpg_neural_parameters_ps_095_100_2023_10_26_21_31_49_SENSITIVITY',
    #     'net_farms_limbless_swimming_cpg_neural_parameters_ps_080_100_2023_10_26_21_31_50_SENSITIVITY',
    #     'net_farms_limbless_swimming_cpg_neural_parameters_ps_065_100_2023_10_26_21_31_51_SENSITIVITY',
    #     'net_farms_limbless_swimming_cpg_neural_parameters_ps_050_100_2023_10_26_21_31_47_SENSITIVITY',
    # ]

    # folder_names = [
    #     'net_farms_limbless_swimming_cpg_neural_parameters_ps_110_100_2023_10_30_22_02_05_SENSITIVITY',
    #     'net_farms_limbless_swimming_cpg_neural_parameters_ps_095_100_2023_10_30_22_01_53_SENSITIVITY',
    #     'net_farms_limbless_swimming_cpg_neural_parameters_ps_080_100_2023_10_30_22_02_09_SENSITIVITY',
    #     'net_farms_limbless_swimming_cpg_neural_parameters_ps_065_100_2023_10_30_22_02_10_SENSITIVITY',
    #     'net_farms_limbless_swimming_cpg_neural_parameters_ps_050_100_2023_10_30_22_01_45_SENSITIVITY',
    # ]

    folder_names = [
        'net_farms_limbless_swimming_cpg_neural_parameters_ps_110_100_2023_10_31_23_45_37_SENSITIVITY',
        'net_farms_limbless_swimming_cpg_neural_parameters_ps_095_100_2023_10_31_23_45_46_SENSITIVITY',
        'net_farms_limbless_swimming_cpg_neural_parameters_ps_080_100_2023_10_31_23_45_35_SENSITIVITY',
        'net_farms_limbless_swimming_cpg_neural_parameters_ps_065_100_2023_10_31_23_45_43_SENSITIVITY',
        'net_farms_limbless_swimming_cpg_neural_parameters_ps_050_100_2023_10_31_23_45_31_SENSITIVITY',
    ]

    # PS gains
    ps_gains = [
        1.10,
        0.95,
        0.80,
        0.65,
        0.50,
    ]

    # Metrics of interest
    sensitivity_metrics_keys = [
        'neur_freq_ax',
        'neur_ptcc_ax',
        'neur_ipl_ax_t',
        'neur_ipl_ax_a',
        'mech_speed_fwd',
        'mech_cot',
    ]

    sensitivity_metrics_labels = [
        'FREQUENCY',
        'PTCC',
        'IPL - Trunk',
        'IPL - Axis',
        'SPEED',
        'COT',
    ]

    sensitivity_indices_1_list = []
    sensitivity_indices_t_list = []

    for folder_name in folder_names:

        # Load sensitivity analysis parameters
        sal_analysis_info = snn_sensitivity_analysis_results.load_sensitivity_analysis_parameters(
            folder_name,
            results_path,
        )

        # Load simulation results
        (
            _params_processes_list,
            _params_runs_processes_list,
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

        # Compute indices
        (
            sensitivity_indices_1,
            sensitivity_indices_t,
        ) = snn_sensitivity_analysis_results.compute_sensitivity_indices(
            analysis_info     = sal_analysis_info,
            metrics_processes = metrics_processes_filt,
            metrics_keys      = sensitivity_metrics_keys,
        )

        sensitivity_indices_1_list.append(sensitivity_indices_1)
        sensitivity_indices_t_list.append(sensitivity_indices_t)

    # Organize sensitivity indices
    sensitivity_indices_0 = {}
    sensitivity_indices_T = {}

    for metric_name in sensitivity_metrics_keys:
        sensitivity_indices_0_metric = []
        sensitivity_indices_T_metric = []

        for ps_gain, s0_all, sT_all in zip(ps_gains, sensitivity_indices_1_list, sensitivity_indices_t_list):
            for repetition, (s0_repetition, sT_repetition) in enumerate(zip(s0_all[metric_name], sT_all[metric_name])):
                s0_dict = {
                    'PS_gain'   : f'{1 / ps_gain :.2f}' if ps_gain < 1 else '0.00',
                    'repetition': repetition
                }

                sT_dict = {
                    'PS_gain'   : f'{1 / ps_gain :.2f}' if ps_gain < 1 else '0.00',
                    'repetition': repetition
                }

                for par_name, s0_value, sT_value in zip(sal_analysis_info['names'], s0_repetition, sT_repetition):

                    if par_name in ['tau2', 'delta_w2']:
                        continue

                    s0_dict[par_name] = s0_value  # if s0_value > 0 else 0
                    sT_dict[par_name] = sT_value  # if sT_value > 0 else 0

                sensitivity_indices_0_metric.append(s0_dict)
                sensitivity_indices_T_metric.append(sT_dict)

        sensitivity_indices_0[metric_name] = pd.DataFrame(sensitivity_indices_0_metric)
        sensitivity_indices_T[metric_name] = pd.DataFrame(sensitivity_indices_T_metric)

    # Plot sensitivity indices of every metric across repetitions, grouping them by parameters name and nesting them by PS gain
    for metric_name, metric_label in zip(sensitivity_metrics_keys, sensitivity_metrics_labels):

        s0_vals = sensitivity_indices_0[metric_name]
        sT_vals = sensitivity_indices_T[metric_name]

        plot_sensitivity_indices(
            sens_indices     = s0_vals,
            metric_label     = metric_label,
            sens_indices_tag = 'S0',
        )

        plot_sensitivity_indices(
            sens_indices     = sT_vals,
            metric_label     = metric_label,
            sens_indices_tag = 'ST',
        )

    # User prompt
    save = input(f'Save {folder_name} figures? [Y/n] - ')

    if save in ['y','Y','1']:
        analysis_tag = input('Analysis tag: ')
        analysis_tag = f'_{analysis_tag}' if analysis_tag else ''
        snn_utils.save_all_figures(
            f'{analysis_name}{analysis_tag}',
            results_path = f'{CURRENTDIR}/figures'
        )

    plt.show()


if __name__ == '__main__':
    main()
