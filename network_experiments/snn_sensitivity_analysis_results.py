''' Module to perform sensitivity analysis '''
import logging
import dill

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from SALib.analyze import sobol

from network_experiments import snn_simulation_results, snn_utils

# PLOTTING PARAMETERS
SMALL_SIZE  = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 17

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.rc(  'font', size      = SMALL_SIZE )  # controls default text sizes
plt.rc(  'axes', titlesize = SMALL_SIZE )  # fontsize of the axes title
plt.rc(  'axes', labelsize = BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc( 'xtick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc( 'ytick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize  = SMALL_SIZE )  # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title

## LOAD RESULTS
def load_sensitivity_analysis_parameters(
    folder_name : str,
    results_path: str,
) -> dict:
    ''' Get parameters for the sensitivity analysis '''

    folder_path = f'{results_path}/{folder_name}'
    data_file_dll = f'{folder_path}/sensitivity_analysis_parameters.dill'

    # Load information
    with open(data_file_dll, 'rb') as infile:
        analysis_info = dill.load(infile)

    return analysis_info

def get_sensitivity_analysis_results(
    folder_name      : str,
    results_data_path: str,
) -> tuple[ list[dict], list[list[dict]], dict[str, list[np.ndarray]] ]:
    ''' Function to retrieve the results from a sensitivity analysis '''

    return snn_simulation_results.get_analysis_results(
        folder_name       = folder_name,
        results_data_path = results_data_path,
    )


## POST-PROCESSING
def compute_sensitivity_indices(
    analysis_info     : dict,
    metrics_processes : dict[str, list[np.ndarray]],
    metrics_keys      : list[str] = None,
) -> dict[str, list[np.ndarray]]:
    ''' Compute resulting sensitivity indices '''

    if metrics_keys is None:
        metrics_keys = list(metrics_processes.keys())

    logging.info('Computing sensitivity indices')

    problem = {
        key : value
        for key, value in analysis_info.items()
        if key in ['num_vars', 'names', 'bounds']
    }

    # Compute first-order sensitivity indices for all the provided metrics
    sensitivity_indices_1 : dict[str, list[float] ] = {}
    sensitivity_indices_t : dict[str, list[float] ] = {}

    for metric_name in metrics_keys:
        sensitivity_indices_1[metric_name] = []
        sensitivity_indices_t[metric_name] = []
        for metric_values_process in metrics_processes[metric_name]:
            sensitivity_inds = sobol.analyze(
                problem,
                metric_values_process,
                False
            )

            sensitivity_indices_1[metric_name].append(sensitivity_inds['S1'])
            sensitivity_indices_t[metric_name].append(sensitivity_inds['ST'])

    return sensitivity_indices_1, sensitivity_indices_t

## PLOT RESULTS
def plot_sensitivity_indices_boxplot(
    si_distrib : np.ndarray,
    pars_names : list[str],
    metric_name: str,
    axis       : plt.Axes = None,
):
    ''' Plot boxplot of sensitivity indices '''
    n_pars = len(pars_names)

    if axis is None:
        axis = plt.axes()
    axis.set_title(metric_name, fontsize= 20, fontweight= 'medium')
    axis.boxplot(si_distrib)
    axis.set_yticks(np.arange(1, n_pars+1))
    axis.set_yticklabels(pars_names)
    axis.tick_params(labelsize= 15)
    axis.set_xlim( -1.00, 1.00 )
    axis.grid()
    axis.set_ylim( 0.5, n_pars + 0.5)
    axis.plot( [0,0], [0.5, n_pars + 0.5], 'k', linewidth= 1)

def plot_sensitivity_indices_violin(
    si_distrib : np.ndarray,
    pars_names : list[str],
    metric_name: str,
    colors     : list[str] = None,
    axis       : plt.Axes = None,
):
    ''' Plot violin plot of sensitivity indices '''
    if axis is None:
        axis = plt.axes()

    axis.set_title(metric_name, fontsize= 20, fontweight= 'medium')

    # Lighter colors
    colors_light = [ snn_utils.modify_color(color, 1.5) for color in colors ]

    x_vals = pars_names * si_distrib.shape[0]
    y_vals = si_distrib.reshape(-1)
    sns.violinplot(
        x       = x_vals,
        y       = y_vals,
        palette = colors,
        hue     = x_vals,
        legend  = False,
        inner   = None,
        ax      = axis,
    )
    sns.boxplot(
        x         = x_vals,
        y         = y_vals,
        width     = 0.15,
        flierprops = dict(marker='o', markersize=2,  markeredgecolor='black'),
        palette   = colors_light,
        hue       = x_vals,
        legend    = False,
        ax        = axis,
    )

def plot_sensitivity_indices_distribution(
    analysis_info      : dict,
    sensitivity_indices: dict[str, list[np.ndarray] ],
    figure_tag         : str = '',
    excluded_pars      : list[str] = None,
) -> None:
    ''' Plot the distribution of the sensitivity indices '''

    # Divide parameters for NEURONAL, GLYC, AMPA, NMDA
    # NOTE: ID was previously prepended (ex: 0_dg_ampa)

    if excluded_pars is None:
        excluded_pars = []

    n_pars = analysis_info['num_vars']
    names  = analysis_info['names']

    iexcl = [ i for i in range(n_pars) if names[i] in excluded_pars ]
    ipsfb = [ i for i in range(n_pars) if'ps_'  in names[i] and i not in iexcl]
    inmda = [ i for i in range(n_pars) if'nmda' in names[i] and i not in iexcl]
    iampa = [ i for i in range(n_pars) if'ampa' in names[i] and i not in iexcl]
    iglyc = [ i for i in range(n_pars) if'glyc' in names[i] and i not in iexcl]
    ineur = [ i for i in range(n_pars) if i not in ipsfb + inmda + iampa + iglyc + iexcl ]

    imap = np.array( ineur + iampa + inmda + iglyc + ipsfb )

    # Reordered names
    names = [ names[i] for i in imap ]

    # Plot sensitivity indices distribution for every metric
    sns.set_style('darkgrid')
    colors       = (
        len(ineur) * ['mediumseagreen'] +
        len(iampa) * ['darkorange'] +
        len(inmda) * ['darkorange'] +
        len(iglyc) * ['cornflowerblue'] +
        len(ipsfb) * ['darkorchid']
    )

    for metric_name, metric_si_distrib in sensitivity_indices.items():
        metric_name = metric_name.upper()
        figname = (
            f'Sensitivity index - {figure_tag} - {metric_name}'
            if figure_tag != ''
            else f'Sensitivity index - {metric_name}'
        )

        # Reordered sensitivity indices
        metric_si_distrib = np.array( metric_si_distrib )[:, imap]

        # Plot distribution
        plt.figure(figname, figsize= (10, 10))
        axis = plt.axes()
        plot_sensitivity_indices_violin(
            si_distrib  = metric_si_distrib,
            pars_names  = names,
            metric_name = metric_name,
            colors      = colors,
            axis        = axis,
        )

