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
    folder_name  = 'net_farms_limbless_feedback_topology_strength_swimming_2dim_mn_100_2023_10_20_03_22_49_ANALYSIS'

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
    ps_topology = [
        [ex_up, ex_dw, 0, 0]
        for ex_up in np.linspace(0, 5, 21)
        for ex_dw in np.linspace(0, 5, 21)
        for rep in range(5)
    ] + [
        [0, 0, in_up, in_dw]
        for in_up in np.linspace(0, 5, 21)
        for in_dw in np.linspace(0, 5, 21)
        for rep in range(5)
    ]
    ex_up_list = [ps_top[0] for ps_top in ps_topology]
    ex_dw_list = [ps_top[1] for ps_top in ps_topology]
    in_up_list = [ps_top[2] for ps_top in ps_topology]
    in_dw_list = [ps_top[3] for ps_top in ps_topology]

    metrics = {
        metric_key: [
            metric_val[0] for metric_val in metrics_processes[metric_key]
        ]
        for metric_key in [
            'neur_ptcc_ax',
            'neur_freq_ax',
            'neur_ipl_ax_a',
            'neur_ipl_ax_t',
            'mech_speed_fwd',
            'mech_energy',
            'mech_cot',
        ]
    }

    # Create dataframe of parameters
    data = [
        ex_up_list,
        ex_dw_list,
        in_up_list,
        in_dw_list,
    ]

    columns_names = [
        'ex_up',
        'ex_dw',
        'in_up',
        'in_dw',
    ]

    dataframe_inputs = pd.DataFrame(
        data    = np.array(data).T,
        columns = columns_names
    )

    dataframe_outputs = pd.DataFrame(
        data = metrics
    )

    # NONLINEAR REGRESSION

    target_metrics = [
        ( 'neur_ptcc_ax',   None), # (  1.6,   2.0)
        ( 'neur_freq_ax',   None), # (  2.0,   2.7)
        ( 'neur_ipl_ax_t',  None),
        # ( 'neur_ipl_ax_a',  None), # (0.004, 0.011)
        # ( 'mech_speed_fwd', None), # (  0.0,   0.5)
        # ( 'mech_energy',    None),
        # ( 'mech_cot',       None), # (  0.0,   0.5)
    ]

    # DEPENDENCY ON INHIBITION
    # Select dataframe inputs for which ex_up = 0 and ex_dw = 0
    dataframe_inputs_in = dataframe_inputs[
        (dataframe_inputs['ex_up'] == 0.0) &
        (dataframe_inputs['ex_dw'] == 0.0)
    ]
    dataframe_inputs_in = dataframe_inputs_in.drop(
        columns = ['ex_up', 'ex_dw']
    )
    dataframe_outputs_in = dataframe_outputs.loc[
        dataframe_inputs_in.index,
        [ key for key, _clim in target_metrics]
    ]

    prediction_data_in = [
        [
            [in_up, in_dw][target_ind]
            for in_up in np.linspace(0.0, 2.0, 10)
            for in_dw in np.linspace(0.0, 2.0, 10)
        ]
        for target_ind in range(2)
    ]

    snn_simulation_results.plot_polynomial_regression_results_2D(
        df_inputs       = dataframe_inputs_in,
        df_outputs      = dataframe_outputs_in,
        target_metrics  = target_metrics,
        prediction_data = prediction_data_in,
        columns_names   = ['in_up', 'in_dw'],
        labels_xy       = ['in_up', 'in_dw'],
        poly_deg        = 1
    )

    # # # CONTOUR PLOT
    # in_up = np.linspace(0, 5, 21)
    # in_dw = np.linspace(0, 5, 21)
    # X, Y  = np.meshgrid(in_up, in_dw)
    # freq  = np.array( dataframe_outputs_in['neur_freq_ax'][5:]).reshape(21, 5, 21)
    # freq  = np.mean(freq, axis=1)

    # fig, ax = plt.subplots()
    # cp = ax.contourf(X, Y, freq)
    # fig.colorbar(cp)

    # DEPENDENCY ON EXCITATION
    # Select dataframe inputs and outputs for which in_up = 0 and in_dw = 0
    dataframe_inputs_ex = dataframe_inputs[
        (dataframe_inputs['in_up'] == 0.0) &
        (dataframe_inputs['in_dw'] == 0.0)
    ]
    dataframe_inputs_ex = dataframe_inputs_ex.drop(
        columns = ['in_up', 'in_dw']
    )
    dataframe_outputs_ex = dataframe_outputs.loc[
        dataframe_inputs_ex.index,
        [ key for key, _clim in target_metrics]
    ]

    prediction_data_ex = [
        [
            [ex_up, ex_dw][target_ind]
            for ex_up in np.linspace(0.0, 2.0, 10)
            for ex_dw in np.linspace(0.0, 2.0, 10)
        ]
        for target_ind in range(2)
    ]

    snn_simulation_results.plot_polynomial_regression_results_2D(
        df_inputs       = dataframe_inputs_ex,
        df_outputs      = dataframe_outputs_ex,
        target_metrics  = target_metrics,
        prediction_data = prediction_data_ex,
        columns_names   = ['ex_up', 'ex_dw'],
        labels_xy       = ['ex_up', 'ex_dw'],
        poly_deg        = 1
    )

    # User prompt
    snn_simulation_results.user_prompt(
        folder_name  = folder_name,
        results_path = results_path,
    )

    # Display
    plt.show()


if __name__ == '__main__':
    main()
