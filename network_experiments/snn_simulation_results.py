''' Run the spinal cord model together with the mechanical simulator '''

import os
import dill
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from scipy.interpolate import griddata
from matplotlib.colors import LogNorm

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


## Data retrieval
def get_inds_processes(folder_path: str):
    ''' Get number and index of processes folders in the specified path '''

    inds_processes = []

    for file in os.listdir(folder_path):
        if os.path.isdir(f'{folder_path}/{file}') and file.startswith('process_'):
            inds_processes.append(int(file.split('_')[-1]))

    return sorted(inds_processes)

def get_analysis_results(
    folder_name      : str,
    results_data_path: str,
    inds_processes   : list[int] = None,
) -> tuple[ list[dict], list[list[dict]], dict[str, list[np.ndarray]] ]:
    ''' Function to retrieve the results from an analysis '''

    logging.info('Loading analysis results')

    params_processes_list      : list[dict]                  = []
    params_runs_processes_list : list[list[dict]]            = []
    metrics_processes_list     : list[dict[str, np.ndarray]] = []

    # Get all found processes
    folder_path    = f'{results_data_path}/{folder_name}'

    if inds_processes is None:
        inds_processes = get_inds_processes(folder_path)

    for process_ind in inds_processes:

        folder_path  = f'{results_data_path}/{folder_name}'
        process_path = f'{folder_path}/process_{process_ind}'

        # Parameters
        data_file = f'{process_path}/snn_parameters_process.dill'
        logging.info('Loading snn_parameters_process data from %s', data_file)

        with open(data_file, "rb") as infile:
            params_process : dict       = dill.load(infile)
            params_runs    : list[dict] = dill.load(infile)

        # Performance
        data_file = f'{process_path}/snn_performance_process.dill'
        logging.info('Loading snn_performance_process data from %s', data_file)

        with open(data_file, "rb") as infile:
            metrics_runs : dict[str, np.ndarray] = dill.load(infile)

        # Collect from all processes
        params_processes_list.append(params_process)
        params_runs_processes_list.append(params_runs)
        metrics_processes_list.append(metrics_runs)

    metrics_processes = {
        key : [ metrics_process[key] for metrics_process in metrics_processes_list ]
        for key in metrics_processes_list[0].keys()
    }

    return  params_processes_list, params_runs_processes_list, metrics_processes

## Utils
def get_metrics_for_desired_processes(
    metrics_processes: dict[str, list[np.ndarray]],
    processes_inds   : list[int],
) -> dict[str, list[np.ndarray]] :
    ''' Get metrics values only for the selected processes indices '''
    return {
        metric_name : [ metric_values[process_ind] for process_ind in processes_inds ]
        for metric_name, metric_values in metrics_processes.items()
    }

## Fitting
def get_polynomial_inputs(
    inputs: pd.DataFrame,
    deg   : int
):
    ''' Get polynomial features '''

    polynomial_converter = PolynomialFeatures(
        degree       = deg,
        include_bias = True
    )
    inputs_poly = polynomial_converter.fit_transform(
        X = inputs,
    )
    return inputs_poly

def train_nlf_regression_model(
    inputs: pd.DataFrame,
    output: pd.DataFrame,
):
    ''' Train polynomial regression model'''
    output = output.fillna(output.mean())
    model = LinearRegression(fit_intercept=True)
    model.fit(inputs, output)
    return model

def train_mlp_regression_model(
    inputs         : pd.DataFrame,
    output         : pd.DataFrame,
    n_hidden_layers: int = 1,
):
    ''' Train MLP regression model'''
    output = output.fillna(output.mean())
    n_neurons = round( np.sqrt(inputs.shape[1]) )
    model = MLPRegressor(
        hidden_layer_sizes = (n_neurons)*n_hidden_layers,
        random_state       = 100,
        verbose            = False,
        n_iter_no_change   = 100,
    )
    model.fit(inputs, output)
    return model

def evaluate_nlf_regression_model(
    inputs: pd.DataFrame,
    output: pd.DataFrame,
    deg   : int
):
    ''' Run polynomial regression model of degree d '''

    # Handle NAN values
    output = output.fillna(output.mean())

    # Get polynomial features
    inputs_poly = get_polynomial_inputs(
        inputs = inputs,
        deg    = deg
    )

    # Split training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        inputs_poly,
        output,
        test_size    = 0.3,
        random_state = 101
    )

    # Train model
    model = train_nlf_regression_model(
        inputs = X_train,
        output = y_train
    )

    # Predict on both train and test
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Errors on Train Set
    train_rmse = np.sqrt(mean_squared_error(y_train,train_pred))
    train_r2   = r2_score(y_train,train_pred)

    # Errors on Test Set
    test_rmse = np.sqrt(mean_squared_error(y_test,test_pred))
    test_r2   = r2_score(y_test,test_pred)

    return train_rmse, train_r2, test_rmse, test_r2

def evaluate_mlp_regression_model(
    inputs         : pd.DataFrame,
    output         : pd.DataFrame,
    deg            : int = 1,
    n_hidden_layers: int = 1,
):
    ''' Run MLP regression model '''

    # Handle NAN values
    output = output.fillna(output.mean())

    # Get polynomial features
    inputs_poly = get_polynomial_inputs(
        inputs = inputs,
        deg    = deg
    )

    # Split training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        inputs_poly,
        output,
        test_size    = 0.3,
        random_state = 101
    )

    # Train model
    model = train_mlp_regression_model(
        inputs          = X_train,
        output          = y_train,
        n_hidden_layers = n_hidden_layers
    )

    # Make prediction on test dataset
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Errors on Train Set
    train_rmse = np.sqrt(mean_squared_error(y_train,train_pred))
    train_r2 = r2_score(y_train,train_pred)

    # Errors on Test Set
    test_rmse = np.sqrt(mean_squared_error(y_test,test_pred))
    test_r2 = r2_score(y_test,test_pred)

    return train_rmse, train_r2, test_rmse, test_r2

def evaluate_multiple_nlf_regression_models(
    inputs     : pd.DataFrame,
    outputs    : pd.DataFrame,
    output_keys: list[str],
    degs       : list[int],
    plot       : bool = True,
):
    ''' Train multiple polynomial regression models of different degrees '''

    for output_key in output_keys:

        train_rmse_errors = []
        test_rmse_errors  = []
        train_r2_scores   = []
        test_r2_scores    = []

        for deg in degs:
            logging.info(f'Fitting {outputs[output_key].name} - NLF of degree {deg}')
            train_rmse, train_r2, test_rmse, test_r2 = evaluate_nlf_regression_model(
                inputs = inputs,
                output = outputs[output_key],
                deg    = deg
            )
            logging.info(f'Train RMSE: {train_rmse:.3f} | Train R2: {train_r2:.3f}')
            logging.info(f'Test RMSE: {test_rmse:.3f} | Test R2: {test_r2:.3f}')
            train_rmse_errors.append(train_rmse)
            train_r2_scores.append(train_r2)
            test_rmse_errors.append(test_rmse)
            test_r2_scores.append(test_r2)

        if plot:
            plot_polynomial_fitting_results(
                target_key        = f'{outputs[output_key].name} - NLF',
                tested_degrees    = degs,
                train_rmse_errors = train_rmse_errors,
                train_r2_scores   = train_r2_scores,
                test_rmse_errors  = test_rmse_errors,
                test_r2_scores    = test_r2_scores,
            )
    return

def evaluate_multiple_mlp_regression_models(
    inputs     : pd.DataFrame,
    outputs    : pd.DataFrame,
    output_keys: list[str],
    degs       : list[int],
    plot       : bool = True,
):
    ''' Train multiple polynomial regression models of different degrees '''

    for output_key in output_keys:

        train_rmse_errors = []
        test_rmse_errors  = []
        train_r2_scores   = []
        test_r2_scores    = []

        for deg in degs:
            logging.info(f'Fitting {outputs[output_key].name} - MLP of degree {deg}')
            train_rmse, train_r2, test_rmse, test_r2 = evaluate_mlp_regression_model(
                inputs = inputs,
                output = outputs[output_key],
                deg    = deg
            )
            logging.info(f'Train RMSE: {train_rmse:.3f} | Train R2: {train_r2:.3f}')
            logging.info(f'Test RMSE: {test_rmse:.3f} | Test R2: {test_r2:.3f}')

            train_rmse_errors.append(train_rmse)
            test_rmse_errors.append(test_rmse)
            train_r2_scores.append(train_r2)
            test_r2_scores.append(test_r2)

        if plot:
            plot_polynomial_fitting_results(
                target_key        = f'{outputs[output_key].name} - MLP',
                tested_degrees    = degs,
                train_rmse_errors = train_rmse_errors,
                train_r2_scores   = train_r2_scores,
                test_rmse_errors  = test_rmse_errors,
                test_r2_scores    = test_r2_scores,
            )
    return

## Plotting
def plot_1d_distribution(
    xvals,
    results,
    stdvals,
    labels,
    figname   : str,
    create_fig: bool = True,
    label     : str= '')   :
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 2].

    labels - The labels should be a list of two string for the xlabel and the
    ylabel (in that order).
    """
    fig = plt.figure(figname) if create_fig else None

    # Plot
    plt.plot(
        xvals,
        results,
        #color= 'b',
        label= label
    )
    plt.errorbar(
        xvals,
        results,
        yerr       = stdvals,
        fmt        = 'o',
        #color     = 'black',
        ecolor     = 'black',
        elinewidth = 3,
        capsize    = 6
    )

    # Decoration
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(figname.upper())
    plt.legend(fontsize= 8)
    plt.grid(True)
    return fig

def plot_2d_distribution(
        xvals,
        yvals,
        results,
        labels,
        figname   : str,
        n_data    : int  = 300,
        create_fig: bool = True,
        axis      : plt.Axes = None,
        clim      : tuple = None,
    ):
    """Plot result

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    log = False
    cmap = 'nipy_spectral'

    if axis is None:
        fig  = plt.figure(figname) if create_fig else None
        axis = fig.add_subplot(111)

    # Grid
    xnew = np.linspace(min(xvals), max(xvals), n_data)
    ynew = np.linspace(min(yvals), max(yvals), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)

    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )

    # Interpolation
    results_interp = griddata(
        (xvals, yvals), results,
        (grid_x, grid_y),
        method='linear',  # nearest, cubic
    )

    # Plot
    axis.plot(xvals, yvals, 'r.')

    imgplot = axis.imshow(
        results_interp,
        extent        = extent,
        aspect        = 'equal',
        origin        = 'lower',
        interpolation = 'none',
        norm          = LogNorm() if log else None
    )
    if clim is not None:
        imgplot.set_clim(clim)
    if cmap is not None:
        imgplot.set_cmap(cmap)


    # Decoration
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(figname.upper())
    cbar = plt.colorbar(imgplot)
    cbar.set_label(labels[2])
    plt.tight_layout()
    return

def plot_3d_scatter(
        xvals,
        yvals,
        zvals,
        results,
        labels,
        figname   : str,
):
    """Plot result

    labels - The labels should be a list of four string for the xlabel, the
    ylabel, zlabel and result (in that order).
    """
    fig = plt.figure(figname)
    ax  = fig.add_subplot(projection='3d')

    colors = pl.cm.jet( np.linspace( 0, 1, len(zvals)) )

    for z_ind, z_val in enumerate(zvals):
        ax.scatter(
            xvals[z_ind],
            yvals[z_ind],
            results[z_ind],
            c = colors[z_ind],
            label = f'{labels[2]} = {z_val:.2f}',
        )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[3])
    ax.legend(
        bbox_to_anchor=(1.3, 1.0),
        loc='upper left',
        borderaxespad=0.
    )

    plt.tight_layout()
    return

def plot_polynomial_fitting_results(
        target_key       : str,
        tested_degrees   : list,
        train_rmse_errors: list,
        train_r2_scores  : list,
        test_rmse_errors : list,
        test_r2_scores   : list,
    ):
    ''' Plot the results for a given metric '''

    plt.figure(f'{target_key} - RMSE', figsize=(10,6))
    plt.plot(
        tested_degrees,
        train_rmse_errors,
        label      = 'TRAIN',
        marker     = 'o',
        markersize = 4.0,
        linestyle  = '--',
        linewidth  = 1.0,
    )
    plt.plot(
        tested_degrees,
        test_rmse_errors,
        label      = 'TEST',
        marker     = 'D',
        markersize = 4.0,
        linestyle  = '-.',
        linewidth  = 1.0,
    )
    plt.xlim(tested_degrees[0], tested_degrees[-1])
    plt.xlabel("Polynomial Complexity")
    plt.ylabel("RMSE")
    plt.title("RMSE")
    plt.grid(True)
    plt.legend()

    plt.figure(f'{target_key} - R2 SCORE', figsize=(10,6))
    plt.plot(
        tested_degrees,
        train_r2_scores,
        label      = 'TRAIN',
        marker     = 'o',
        markersize = 4.0,
        linestyle  = '--',
        linewidth  = 1.0,
    )
    plt.plot(
        tested_degrees,
        test_r2_scores,
        label      = 'TEST',
        marker     = 'D',
        markersize = 4.0,
        linestyle  = '-.',
        linewidth  = 1.0,
    )
    plt.xlim(tested_degrees[0], tested_degrees[-1])
    plt.xlabel("Polynomial Complexity")
    plt.ylabel("R2 SCORE")
    plt.title("R2 SCORE")
    plt.grid(True)
    plt.legend()

    return

def plot_polynomial_regression_results_2D(
        df_inputs      : pd.DataFrame,
        df_outputs     : pd.DataFrame,
        target_metrics : list[str],
        prediction_data: list[list[float]],
        columns_names  : list[str],
        poly_deg       : int,
        labels_xy      : list[str],
):
    ''' Plot the regression results when evaluating with a 2D input '''

    # Polynomial Fit for different degrees
    evaluate_multiple_nlf_regression_models(
        inputs      = df_inputs,
        outputs     = df_outputs,
        output_keys = [ key for key, _clim in target_metrics],
        degs        = [1,2,3,4],
    )

    # # MLP Fit for different degrees
    # evaluate_multiple_mlp_regression_models(
    #     inputs      = df_inputs,
    #     outputs     = df_outputs,
    #     output_keys = [ key for key, _clim in target_metrics],
    #     degs        = [1,2],
    # )

    # POLYNOMIAL TRAIN INPUT
    dataframe_inputs_poly = get_polynomial_inputs(
        inputs = df_inputs,
        deg    = poly_deg,
    )

    # POLYNOMIAL TEST INPUT
    X_prediction = pd.DataFrame(
        data    = np.array(prediction_data).T,
        columns = columns_names
    )

    X_prediction_in_poly = get_polynomial_inputs(
        inputs = X_prediction,
        deg    = poly_deg,
    )

    for target_key, target_limits in target_metrics:

        # TRAIN MODEL
        nlf_model = train_nlf_regression_model(
            inputs = dataframe_inputs_poly,
            output = df_outputs[target_key],
        )

        # PREDICT
        y_prediction_in = nlf_model.predict(X_prediction_in_poly)

        # PLOT
        plot_2d_distribution(
            xvals   = X_prediction[labels_xy[0]].values,
            yvals   = X_prediction[labels_xy[1]].values,
            results = y_prediction_in,
            labels  = labels_xy + [target_key ],
            figname = '_'.join([target_key] + labels_xy),
            clim    = target_limits,
        )

## Saving
def _save_figure(
    figure      : str,
    filetag     : str,
    results_path: str,
    name        : str= None,
    **kwargs
):
    """ Save figure """

    for extension in kwargs.pop('extensions', ['pdf']):
        fig = figure.replace(' ', '_').replace('.', 'dot')

        path = f'{results_path}/images/{filetag}'
        name = f'{path}/{fig}.{extension}'
        os.makedirs(path, exist_ok=True)

        fig = plt.figure(figure)
        size = plt.rcParams.get('figure.figsize')
        fig.set_size_inches(0.7*size[0], 0.7*size[1], forward=True)
        plt.tight_layout()
        plt.savefig(name, bbox_inches='tight')
        logging.info('Saving figure %s...', name)
        fig.set_size_inches(size[0], size[1], forward=True)

def save_all_figures(
    filetag     : str,
    results_path: str,
    **kwargs
):
    """Save_figures"""
    figures = [str(figure) for figure in plt.get_figlabels()]
    logging.info('Other files:\n    - %s', '\n    - '.join(figures))
    for figname in figures:
        _save_figure(
            figure       = figname,
            filetag      = filetag,
            results_path = results_path,
            extensions   = kwargs.pop('extensions', ['pdf']),
        )

## User prompt
def user_prompt(
    folder_name  : str,
    results_path : str,
):
    """User prompt"""
    save = input('Save figures? [Y/n] - ')

    if save in ['y','Y','1']:
        folder_tag  = input('Analysis tag: ')
        save_all_figures(
            f'{folder_name}_{folder_tag}',
            results_path = results_path
        )
