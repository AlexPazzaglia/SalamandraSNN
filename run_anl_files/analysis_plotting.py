import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.ndimage import gaussian_filter
from matplotlib.cm import ScalarMappable

# FIGURE PARAMETERS
SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.rc(  'font', size      = SMALL_SIZE )  # controls default text sizes
plt.rc(  'axes', titlesize = SMALL_SIZE )  # fontsize of the axes title
plt.rc(  'axes', labelsize = BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc( 'xtick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc( 'ytick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize  = SMALL_SIZE )  # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title

def plot_1d_grid_search_in_1d(
    metric_name     : str,
    metric_label    : str,
    metric_vals     : np.ndarray,
    metric_mean     : np.ndarray,
    metric_std      : np.ndarray,
    metric_limits   : list[float],
    par_0_vals      : np.ndarray,
    par_0_name      : str,
    par_0_label     : str,
    tag             : str  = '',
    log_scale_metric: bool = False,
    log_scale_p0    : bool = False,
    **kwargs,
):
    ''' Plot the results of the 1D grid search in a 1D plot '''

    tag = f' - {tag}' if tag != '' else ''
    fig = plt.figure(f'{metric_name} - plot_1D{tag}', figsize=(10,5))

    for metric_v in metric_vals:
        plt.plot(
            par_0_vals,
            metric_v,
            color  = 'gray',
            alpha  = 0.2,
            marker = '.',
        )

    plt.plot(
        par_0_vals,
        metric_mean,
    )

    plt.fill_between(
        par_0_vals,
        metric_mean - metric_std,
        metric_mean + metric_std,
        alpha = 0.05,
    )

    # Increase size of text
    ax : plt.Axes = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

    # Decorate
    plt.xlim( par_0_vals.min(), par_0_vals.max() )
    plt.ylim( metric_limits )
    plt.xlabel(par_0_label)
    plt.ylabel(metric_label)

    if log_scale_p0:
        plt.xscale('log')
    if log_scale_metric:
        plt.yscale('log')

    if kwargs.get('grid', True):
        plt.grid()
    else:
        plt.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    return fig, ax

def plot_1d_grid_search(
    metric_name     : str,
    metric_label    : str,
    metric_vals     : np.ndarray,
    metric_mean     : np.ndarray,
    metric_std      : np.ndarray,
    metric_limits   : list[float],
    par_0_vals      : np.ndarray,
    par_0_name      : str,
    par_0_label     : str,
    tag             : str  = '',
    log_scale_metric: bool = False,
    log_scale_p0    : bool = False,
    filter_metric   : bool = False,
    metric_mask     : np.ndarray = None,
    **kwargs,
):
    ''' Plot the results of the 1D grid search '''

    tag = f'{tag} - ' if tag != '' else ''

    if filter_metric:
        metric_mean = gaussian_filter(metric_mean, sigma=1.0)
        metric_std  = gaussian_filter(metric_std, sigma=1.0)
        tag         = f'{tag}filtered '
    else:
        tag         = f'{tag}unfiltered '

    metric_mean = copy.deepcopy(metric_mean)
    metric_mean[metric_mean < metric_limits[0] ] = np.nan
    metric_mean[metric_mean > metric_limits[1] ] = np.nan

    if metric_mask is not None:
        metric_mean[~metric_mask] = np.nan

    fig_1d, ax_1d = plot_1d_grid_search_in_1d(
        metric_name      = metric_name,
        metric_label     = metric_label,
        metric_vals      = metric_vals,
        metric_mean      = metric_mean,
        metric_std       = metric_std,
        metric_limits    = metric_limits,
        par_0_vals       = par_0_vals,
        par_0_name       = par_0_name,
        par_0_label      = par_0_label,
        tag              = tag,
        log_scale_metric = log_scale_metric,
        log_scale_p0     = log_scale_p0,
        **kwargs,
    )

    return fig_1d, ax_1d

def plot_2d_grid_search_in_1d(
    metric_name     : str,
    metric_label    : str,
    metric_mean     : np.ndarray,
    metric_std      : np.ndarray,
    metric_limits   : list[float],
    par_0_vals      : np.ndarray,
    par_1_vals      : np.ndarray,
    par_0_name      : str,
    par_0_label     : str,
    par_1_name      : str,
    par_1_label     : str,
    tag             : str  = '',
    log_scale_metric: bool = False,
    log_scale_p0    : bool = False,
    log_scale_p1    : bool = False,
    invert_colors   : bool = False,
    **kwargs,
):
    ''' Plot the results of the 2D grid search in a 1D plot '''

    tag = f' - {tag}' if tag != '' else ''
    fig = plt.figure(f'{metric_name} - plot_1D{tag}', figsize=(10,5))

    colors = pl.cm.jet(np.linspace(0,1, len(par_1_vals)))
    if invert_colors:
        colors = colors[::-1]

    max_labels = 15
    interval_labels = max(1, round(len(par_1_vals)/max_labels))

    for par1_i, par1_v in enumerate(par_1_vals):

        label = (
            f'{par_1_name} = {par1_v:.4f}'
            if par1_i % interval_labels == 0
            else
            None
        )

        plt.plot(
            par_0_vals,
            metric_mean[:, par1_i],
            label = label,
            color = colors[par1_i],
        )

        plt.fill_between(
            par_0_vals,
            metric_mean[:, par1_i] - metric_std[:, par1_i],
            metric_mean[:, par1_i] + metric_std[:, par1_i],
            alpha = 0.05,
            color = colors[par1_i],
        )

    # Increase size of text
    ax : plt.Axes = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

    # Decorate
    plt.xlim( par_0_vals.min(), par_0_vals.max() )
    plt.ylim( metric_limits )
    plt.xlabel(par_0_label)
    plt.ylabel(metric_label)

    if log_scale_p0:
        plt.xscale('log')
    if log_scale_metric:
        plt.yscale('log')

    plt.legend(
        loc            = 'center left',
        bbox_to_anchor = (1, 0.5),
        fontsize       = 10,
    )

    if kwargs.get('grid', True):
        plt.grid()
    else:
        plt.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    return fig, ax

def plot_2d_grid_search_in_2d(
    metric_name     : str,
    metric_label    : str,
    metric_mean     : np.ndarray,
    metric_std      : np.ndarray,
    metric_limits   : list[float],
    par_0_vals      : np.ndarray,
    par_1_vals      : np.ndarray,
    par_0_name      : str,
    par_0_label     : str,
    par_1_name      : str,
    par_1_label     : str,
    tag             : str  = '',
    log_scale_metric: bool = False,
    log_scale_p0    : bool = False,
    log_scale_p1    : bool = False,
    **kwargs,
):
    ''' Plot the results of the 2D grid search in a 2D plot '''

    tag = f' - {tag}' if tag != '' else ''
    fig = plt.figure(f'{metric_name} - plot_2D{tag}', figsize=(10,5))
    ax  = plt.axes()

    X, Y = np.meshgrid(par_0_vals, par_1_vals)

    # Contour plot
    contours = ax.contour(
        X,
        Y,
        metric_mean.T,
        3,
        colors='black',
    )
    ax.clabel(contours, inline= True, fontsize=8)

    # Heatmap
    im = ax.contourf(
        X,
        Y,
        metric_mean.T,
        20,
        cmap='RdBu_r',
        alpha= 0.5,
        vmin = metric_limits[0],
        vmax = metric_limits[1],
    )

    # Colorbar
    cbar = fig.colorbar(
        # im
        ScalarMappable(
            norm=im.norm,
            cmap=im.cmap,
        ),
    )
    cbar.ax.set_ylabel(metric_label, rotation=90, labelpad=15)

    # Decorate
    ax.set_xlabel(par_0_label)
    ax.set_ylabel(par_1_label)

    if log_scale_p0:
        ax.set_xscale('log')
    if log_scale_p1:
        ax.set_yscale('log')

    plt.tight_layout()

    return fig, ax

def plot_2d_grid_search(
    metric_name     : str,
    metric_label    : str,
    metric_mean     : np.ndarray,
    metric_std      : np.ndarray,
    metric_limits   : list[float],
    par_0_vals      : np.ndarray,
    par_1_vals      : np.ndarray,
    par_0_name      : str,
    par_0_label     : str,
    par_1_name      : str,
    par_1_label     : str,
    tag             : str  = '',
    log_scale_metric: bool = False,
    log_scale_p0    : bool = False,
    log_scale_p1    : bool = False,
    filter_metric   : bool = False,
    metric_mask     : np.ndarray = None,
    **kwargs,
):
    ''' Plot the results of the 2D grid search '''

    tag = f'{tag} - ' if tag != '' else ''

    if filter_metric:
        metric_mean = gaussian_filter(metric_mean, sigma=1.0)
        metric_std  = gaussian_filter(metric_std, sigma=1.0)
        tag         = f'{tag}filtered '
    else:
        tag         = f'{tag}unfiltered '

    metric_mean = copy.deepcopy(metric_mean)
    metric_mean[metric_mean < metric_limits[0] ] = np.nan
    metric_mean[metric_mean > metric_limits[1] ] = np.nan

    if metric_mask is not None:
        metric_mean[~metric_mask] = np.nan

    fig_1d, ax_1d = plot_2d_grid_search_in_1d(
        metric_name      = metric_name,
        metric_label     = metric_label,
        metric_mean      = metric_mean,
        metric_std       = metric_std,
        metric_limits    = metric_limits,
        par_0_vals       = par_0_vals,
        par_1_vals       = par_1_vals,
        par_0_name       = par_0_name,
        par_0_label      = par_0_label,
        par_1_name       = par_1_name,
        par_1_label      = par_1_label,
        tag              = tag,
        log_scale_metric = log_scale_metric,
        log_scale_p0     = log_scale_p0,
        log_scale_p1     = log_scale_p1,
        **kwargs,
    )

    fig_2d, ax_2d = plot_2d_grid_search_in_2d(
        metric_name      = metric_name,
        metric_label     = metric_label,
        metric_mean      = metric_mean,
        metric_std       = metric_std,
        metric_limits    = metric_limits,
        par_0_vals       = par_0_vals,
        par_1_vals       = par_1_vals,
        par_0_name       = par_0_name,
        par_0_label      = par_0_label,
        par_1_name       = par_1_name,
        par_1_label      = par_1_label,
        tag              = tag,
        log_scale_metric = log_scale_metric,
        log_scale_p0     = log_scale_p0,
        log_scale_p1     = log_scale_p1,
        **kwargs,
    )

    return fig_1d, ax_1d, fig_2d, ax_2d