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

###############################################################################
################ DATA LOADING #################################################
###############################################################################

def load_sensitivity_indices(
    folder_names            : list[str],
    results_path            : str,
    sensitivity_metrics_keys: list[str],
):
    ''' Get sensitivity indices for a list of folder names '''

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

    return sensitivity_indices_1_list, sensitivity_indices_t_list, sal_analysis_info

def organize_sensitivity_indices(
    sal_analysis_info         : dict,
    sensitivity_indices_1_list: list[dict[str, list[float]]],
    sensitivity_indices_t_list: list[dict[str, list[float]]],
    sensitivity_metrics_keys  : list[str],
    theta_thresholds          : list[float],
    ps_weights                : list[float],
):
    ''' Organize sensitivity indices by THETA_th and PS_weight'''

    # Organize sensitivity indices
    sensitivity_indices_0   = {}
    sensitivity_indices_T   = {}
    sensitivity_indices_all = {}

    # Iterate over metrics
    for metric_name in sensitivity_metrics_keys:
        sensitivity_indices_0_metric   = []
        sensitivity_indices_T_metric   = []
        sensitivity_indices_all_metric = []

        # Iterate over feedback parameters
        sensitivity_params = zip(
            theta_thresholds,
            ps_weights,
            sensitivity_indices_1_list,
            sensitivity_indices_t_list
        )

        for threshold_th, ps_weight, s0_all, sT_all in sensitivity_params:

            # Iterate over repetitions
            metrics_params = zip(
                s0_all[metric_name],
                sT_all[metric_name]
            )

            for repetition, (s0_repetition, sT_repetition) in enumerate(metrics_params):
                s0_dict = {
                    'THETA_th'  : threshold_th,
                    'PS_weight' : ps_weight,
                    'repetition': repetition
                }

                sT_dict = {
                    'THETA_th'  : threshold_th,
                    'PS_weight' : ps_weight,
                    'repetition': repetition
                }

                # Iterate over network parameters
                network_params = zip(
                    sal_analysis_info['names'],
                    s0_repetition,
                    sT_repetition
                )

                for par_name, s0_value, sT_value in network_params:

                    if par_name in ['tau2', 'delta_w2']:
                        continue

                    s0_dict[par_name] = s0_value  # if s0_value > 0 else 0
                    sT_dict[par_name] = sT_value  # if sT_value > 0 else 0

                    s_all_dict = {
                        'THETA_th'  : threshold_th,
                        'PS_weight' : ps_weight,
                        'repetition': repetition,
                        'parameter' : par_name,
                        's0'        : s0_value,
                        'sT'        : sT_value,
                    }
                    sensitivity_indices_all_metric.append(s_all_dict)

                sensitivity_indices_0_metric.append(s0_dict)
                sensitivity_indices_T_metric.append(sT_dict)

        sensitivity_indices_0[metric_name]   = pd.DataFrame(sensitivity_indices_0_metric)
        sensitivity_indices_T[metric_name]   = pd.DataFrame(sensitivity_indices_T_metric)
        sensitivity_indices_all[metric_name] = pd.DataFrame(sensitivity_indices_all_metric)

    return sensitivity_indices_0, sensitivity_indices_T, sensitivity_indices_all

###############################################################################
################ DATA PLOTTING ################################################
###############################################################################

def plot_sensitivity_indices_scatter_for_specific_threshold(
    sens_indices    : pd.DataFrame,
    theta_th        : float,
    metric_label    : str,
    sens_indices_tag: str = '',
    variables_list  : list[str] = None,
):
    ''' Plot sensitivity indices for a specific threshold '''

    # Select the indices for the current THETA_th
    sens_indices = sens_indices[sens_indices['THETA_th'] == theta_th]

    # Select parameters columns
    selected_columns = sens_indices.columns[3:]

    # Reshape for boxplot
    s0_vals_melted = sens_indices.melt(
        id_vars    = ['PS_weight', 'repetition'],
        value_vars = selected_columns,
        var_name   = 'variable',
        value_name = 'value'
    )

    if variables_list is None:
        variables_list = s0_vals_melted['variable'].unique()

    # Compute median for values with the same PS_weight
    s0_vals_melted_median = s0_vals_melted.drop(columns='repetition')
    s0_vals_melted_median = s0_vals_melted_median.groupby(['PS_weight', 'variable'])
    s0_vals_melted_median = s0_vals_melted_median.median().reset_index()

    # Compute IQR for values with the same PS_weight
    s0_vals_melted_iqr = s0_vals_melted.drop(columns='repetition')
    s0_vals_melted_iqr = s0_vals_melted_iqr.groupby(['PS_weight', 'variable'])
    s0_vals_melted_iqr = s0_vals_melted_iqr.quantile([0.25, 0.75]).reset_index()

    # Split into a list of dataframes divided by the 'variable' column
    s0_vals_melted_median_list = [
        s0_vals_melted_median[s0_vals_melted_median['variable'] == var]
        for var in variables_list
    ]

    s0_vals_melted_iqr_list = [
        s0_vals_melted_iqr[s0_vals_melted_iqr['variable'] == var]
        for var in variables_list
    ]

    # Plot as scatter plot whose color is determined by the PS_weight
    n_ps_weights = len(sens_indices['PS_weight'].unique())
    cmap         = plt.cm.get_cmap('RdBu_r', n_ps_weights)
    plt.figure(f'{sens_indices_tag} - {metric_label} - Scatter Plot', figsize=(16, 10))

    n_ps_weights = len(s0_vals_melted_median_list[0])
    dx_ps_weight = np.linspace(-0.3, 0.3, n_ps_weights)

    s0_values = zip(s0_vals_melted_median_list, s0_vals_melted_iqr_list)

    for variable_ind, (s0_vals_median, s0_vals_iqr) in enumerate(s0_values):

        plt.scatter(
            variable_ind * np.ones(n_ps_weights) + dx_ps_weight,
            s0_vals_median['value'],
            label = s0_vals_median['variable'].values[0],
            c     = s0_vals_median['PS_weight'],
            cmap  = cmap,
            s     = 50,
        )

        # Plot the IQR
        errors = np.abs(
            s0_vals_iqr['value'].values -
            np.repeat( s0_vals_median['value'].values, 2)
        )

        plt.errorbar(
            variable_ind * np.ones(n_ps_weights) + dx_ps_weight,
            s0_vals_median['value'],
            yerr       = errors.reshape(n_ps_weights, 2).T,
            fmt        = 'none',
            ecolor     = 'black',
            capsize    = 5,
            capthick   = 0.5,
            elinewidth = 0.5,
        )

        # Connect the points with dotted lines
        plt.plot(
            variable_ind * np.ones(n_ps_weights) + dx_ps_weight,
            s0_vals_median['value'],
            c         = 'black',
            lw        = 0.5,
            linestyle = '--',
        )

    # Set labels and title
    plt.title(metric_label, fontsize=20)
    plt.xlabel('PS gain', fontsize=15)
    plt.ylabel('Sensitivity Index', fontsize=15)

    plt.xticks(
        range(len(s0_vals_melted_median_list)),
        [var for var in variables_list],
        fontsize = 15,
        rotation = 45,
    )
    plt.ylim(-0.1, 1.5)
    plt.yticks(np.arange(0, 1.6, 0.2), fontsize=15)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

def plot_sensitivity_indices_boxplot_for_specific_threshold(
    sens_indices    : pd.DataFrame,
    theta_th        : float,
    metric_label    : str,
    sens_indices_tag: str = '',
    variables_list  : list[str] = None,
):
    ''' Plot sensitivity indices for a specific threshold '''

    # Select the indices for the current THETA_th
    sens_indices = sens_indices[sens_indices['THETA_th'] == theta_th]

    # Select parameters columns
    selected_columns = sens_indices.columns[3:]

    # Reshape for boxplot
    s0_vals_melted = sens_indices.melt(
        id_vars    = ['PS_weight', 'repetition'],
        value_vars = selected_columns,
        var_name   = 'variable',
        value_name = 'value'
    )

    if variables_list is None:
        variables_list = s0_vals_melted['variable'].unique()

    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a boxplot for each variable grouped by PS_weight
    colormap = sns.color_palette("RdBu_r", n_colors=len(sens_indices['PS_weight'].unique()))

    plt.figure(f'{sens_indices_tag} - {metric_label} - Boxplot', figsize=(16, 10))
    sns.boxplot(
        data       = s0_vals_melted,
        x          = 'variable',
        y          = 'value',
        hue        = 'PS_weight',
        palette    = colormap,
        width      = 0.7,
        fliersize  = 0,
        linewidth  = 0.5,
        showfliers = False,
        order      = variables_list,
    )

    # Set labels and title
    plt.title(metric_label, fontsize=20)
    plt.xlabel('')
    plt.ylabel('Sensitivity Index', fontsize=15)

    # Force legend position just outside the plot
    plt.legend(
        title          = 'PS gain',
        bbox_to_anchor = (1.01, 1),
        loc            = 'upper left',
        borderaxespad  = 0.,
        fontsize       = 15,
    )

    plt.ylim(-0.1, 1.5)
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(np.arange(0, 1.6, 0.2), fontsize=15)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

def plot_sensitivity_indices_violin_for_specific_threshold(
    sens_indices    : pd.DataFrame,
    theta_th        : float,
    metric_label    : str,
    sens_indices_tag: str = '',
    variables_list  : list[str] = None,
):
    ''' Plot sensitivity indices for a specific threshold '''

    sens_indices = sens_indices[sens_indices['THETA_th'] == theta_th]

    if variables_list is None:
        variables_list = sens_indices['parameter'].unique()

    plt.figure(f'{sens_indices_tag} - {metric_label} - Violin plot', figsize=(16, 10))
    axis = plt.axes()
    axis.set_title(metric_label, fontsize= 20, fontweight= 'medium')

    sns.violinplot(
        data    = sens_indices,
        x       = 'parameter',
        y       = 'sT',
        hue     = 'PS_weight',
        legend  = False,
        inner   = None,
        ax      = axis,
        order   = variables_list,
    )
    sns.boxplot(
        data      = sens_indices,
        x         = 'parameter',
        y         = 'sT',
        #width     = 0.15,
        flierprops = dict(marker='o', markersize=2,  markeredgecolor='black'),
        hue       = 'PS_weight',
        legend    = False,
        ax        = axis,
        order     = variables_list,
    )

    # Set labels and title
    plt.title(metric_label, fontsize=20)
    plt.xlabel('')
    plt.ylabel('Sensitivity Index', fontsize=15)

    plt.ylim(-0.1, 1.5)
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(np.arange(0, 1.6, 0.2), fontsize=15)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    return


def plot_sensitivity_indices(
    sens_indices_0  : pd.DataFrame,
    sens_indices_t  : pd.DataFrame,
    sens_indices_all: pd.DataFrame,
    metric_label    : str,
    sens_indices_tag: str = '',
    theta_th        : float = 0.10,
    variables_list  : list[str] = None,
):
    ''' Plot sensitivity indices '''

    # Get unique values of THETA_th
    if isinstance(theta_th, float):
        theta_th_values = [theta_th]
    else:
        theta_th_values = sens_indices_0['THETA_th'].unique()

    for theta_th in theta_th_values:

        metric_label_th = f'{metric_label}_th_{theta_th:.2f}'

        # Scatter plot
        # plot_sensitivity_indices_scatter_for_specific_threshold(
        #     sens_indices    = sens_indices_0,
        #     theta_th        = theta_th,
        #     metric_label    = metric_label_th,
        #     sens_indices_tag= sens_indices_tag,
        #     variables_list  = variables_list,
        # )
        plot_sensitivity_indices_scatter_for_specific_threshold(
            sens_indices    = sens_indices_t,
            theta_th        = theta_th,
            metric_label    = metric_label_th,
            sens_indices_tag= sens_indices_tag,
            variables_list  = variables_list,
        )

        # Boxplot
        # plot_sensitivity_indices_boxplot_for_specific_threshold(
        #     sens_indices    = sens_indices_0,
        #     theta_th        = theta_th,
        #     metric_label    = metric_label_th,
        #     sens_indices_tag= sens_indices_tag,
        #     variables_list  = variables_list,
        # )
        plot_sensitivity_indices_boxplot_for_specific_threshold(
            sens_indices    = sens_indices_t,
            theta_th        = theta_th,
            metric_label    = metric_label_th,
            sens_indices_tag= sens_indices_tag,
            variables_list  = variables_list,
        )

        # Violin plot
        plot_sensitivity_indices_violin_for_specific_threshold(
            sens_indices    = sens_indices_all,
            theta_th        = theta_th,
            metric_label    = metric_label_th,
            sens_indices_tag= sens_indices_tag,
            variables_list  = variables_list,
        )

    return

###############################################################################
###############################################################################
###############################################################################

def main():
    ''' Run sensitivity analysis '''

    results_path = '/data/pazzagli/simulation_results/data/1_sensitivity_analysis/salamander/swimming/closed_loop/varying_ps_weight'

    analysis_name = 'net_farms_limbless_swimming_cpg_neural_parameters_vary_ps_gain_ps_weight_100_2023_11_03_SENSITIVITY'

    folder_names = [
        # Alpha 0.1
        'net_farms_limbless_ps_weight_swimming_cpg_neural_parameters_alpha_010_ps_weight_000_100_2024_02_12_13_36_51_SENSITIVITY',
        'net_farms_limbless_ps_weight_swimming_cpg_neural_parameters_alpha_010_ps_weight_050_100_2024_02_12_13_36_54_SENSITIVITY',
        'net_farms_limbless_ps_weight_swimming_cpg_neural_parameters_alpha_010_ps_weight_100_100_2024_02_12_13_37_19_SENSITIVITY',
        'net_farms_limbless_ps_weight_swimming_cpg_neural_parameters_alpha_010_ps_weight_150_100_2024_02_12_13_37_03_SENSITIVITY',

        # Alpha 0.3
        'net_farms_limbless_ps_weight_swimming_cpg_neural_parameters_alpha_030_ps_weight_000_100_2024_02_12_13_37_25_SENSITIVITY',
        'net_farms_limbless_ps_weight_swimming_cpg_neural_parameters_alpha_030_ps_weight_050_100_2024_02_12_13_37_14_SENSITIVITY',
        'net_farms_limbless_ps_weight_swimming_cpg_neural_parameters_alpha_030_ps_weight_100_100_2024_02_12_13_37_26_SENSITIVITY',
        'net_farms_limbless_ps_weight_swimming_cpg_neural_parameters_alpha_030_ps_weight_150_100_2024_02_13_03_44_33_SENSITIVITY',

        # Alpha 0.5
        'net_farms_limbless_ps_weight_swimming_cpg_neural_parameters_alpha_050_ps_weight_000_100_2024_02_13_03_47_48_SENSITIVITY',
        'net_farms_limbless_ps_weight_swimming_cpg_neural_parameters_alpha_050_ps_weight_050_100_2024_02_13_03_52_01_SENSITIVITY',
        'net_farms_limbless_ps_weight_swimming_cpg_neural_parameters_alpha_050_ps_weight_100_100_2024_02_13_04_37_24_SENSITIVITY',
        'net_farms_limbless_ps_weight_swimming_cpg_neural_parameters_alpha_050_ps_weight_150_100_2024_02_13_04_41_35_SENSITIVITY',
    ]

    # Parameters
    theta_thresholds = [0.10] * 4 + [0.30] * 4 + [0.50] * 4
    ps_weights       = [0.00, 0.50, 1.00, 1.50] * 3

    # Metrics of interest
    sensitivity_metrics_keys = [
        'neur_freq_ax',
        'neur_ptcc_ax',
        'neur_wave_number_a',
        'mech_speed_fwd',
        'mech_cot',
    ]

    sensitivity_metrics_labels = [
        'FREQUENCY',
        'PTCC',
        'WAVE NUMBER',
        'SPEED',
        'COT',
    ]

    # Get sensitivity indices
    (
        sensitivity_indices_1_list,
        sensitivity_indices_t_list,
        sal_analysis_info
    ) = load_sensitivity_indices(
        folder_names            = folder_names,
        results_path            = results_path,
        sensitivity_metrics_keys= sensitivity_metrics_keys,
    )

    # Organize sensitivity indices
    (
        sensitivity_indices_l,
        sensitivity_indices_t,
        sensitivity_indices_all,
    ) = organize_sensitivity_indices(
        sal_analysis_info         = sal_analysis_info,
        sensitivity_indices_1_list= sensitivity_indices_1_list,
        sensitivity_indices_t_list= sensitivity_indices_t_list,
        sensitivity_metrics_keys  = sensitivity_metrics_keys,
        theta_thresholds          = theta_thresholds,
        ps_weights                = ps_weights,
    )

    # for theta_th in np.unique(theta_thresholds):
    #     for metric_name in sensitivity_metrics_keys:

    #         # Get sensitivity indices with the current THETA_th and metric
    #         st_distrib = sensitivity_indices_all[metric_name]
    #         st_distrib = st_distrib[st_distrib['THETA_th'] == theta_th]

    #         plt.figure(f'{metric_name} - THETA_th {theta_th}', figsize=(16, 10))
    #         axis = plt.axes()
    #         axis.set_title(metric_name, fontsize= 20, fontweight= 'medium')

    #         sns.violinplot(
    #             data    = st_distrib,
    #             x       = 'parameter',
    #             y       = 'sT',
    #             hue     = 'PS_weight',
    #             legend  = False,
    #             inner   = None,
    #             ax      = axis,
    #         )
    #         sns.boxplot(
    #             data      = st_distrib,
    #             x         = 'parameter',
    #             y         = 'sT',
    #             #width     = 0.15,
    #             flierprops = dict(marker='o', markersize=2,  markeredgecolor='black'),
    #             hue       = 'PS_weight',
    #             legend    = False,
    #             ax        = axis,
    #         )

    # Plot sensitivity indices
    variables_list = [
        'tau_memb',
        'R_memb',
        'tau1',
        'delta_w1',
        'weight_ampa',
        'weight_nmda',
        'weight_glyc',
        'ps_weight',
    ]

    for metric_name, metric_label in zip(sensitivity_metrics_keys, sensitivity_metrics_labels):

        plot_sensitivity_indices(
            sens_indices_0   = sensitivity_indices_l[metric_name],
            sens_indices_t   = sensitivity_indices_t[metric_name],
            sens_indices_all = sensitivity_indices_all[metric_name],
            metric_label     = metric_label,
            sens_indices_tag = 'ST',
            variables_list   = variables_list,
        )

    # User prompt
    save = input(f'Save {analysis_name} figures? [Y/n] - ')

    if save in ['y','Y','1']:
        analysis_tag = input('Analysis tag: ')
        analysis_tag = f'_{analysis_tag}' if analysis_tag else ''
        snn_utils.save_all_figures(
            f'{analysis_name}{analysis_tag}',
            results_path = f'{CURRENTDIR}/figures'
        )

    # Show
    plt.show(block=False)
    input('Press enter to close all figures')
    plt.close('all')



if __name__ == '__main__':
    main()
