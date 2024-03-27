'''
Simulation Framework for Spinal Cord Network Models and Mechanical Simulators

For detailed information on each function, consult the respective function's docstring.

'''

import numpy as np
import matplotlib.pyplot as plt

from typing import Union

from network_modules.plotting.mechanics_plotting import MechPlotting
from network_modules.experiment.network_experiment import SnnExperiment

## Post processing
def _post_processing_single_net_single_run(
    control_type: str,
    snn_sim     : SnnExperiment,
    mech_sim    : MechPlotting,
    plot_figures: bool,
    save_prompt : bool,
) -> dict[str, Union[float, np.ndarray[float]]]:
    '''
    Perform post-processing for a network model simulation.

    Args:
        snn_sim (SnnExperiment): Instance of the SnnExperiment class.
        mech_sim (MechPlotting): Instance of the MechPlotting class.
        plot_figures (bool): Whether to plot figures.
        save_prompt (bool): Whether to save prompts.

    Returns:
        dict[str, Union[float, np.ndarray[float]]]: Dictionary containing computed metrics.
    '''

    metrics = {}
    if control_type == 'closed_loop':
        snn_res = True
        mec_res = True
        fig_dict = None
    elif control_type == 'open_loop':
        snn_res = True
        mec_res = False
        fig_dict = None
    elif control_type == 'signal_driven':
        snn_res = False
        mec_res = True
        fig_dict = {}
    elif control_type == 'position_control':
        snn_res = False
        mec_res = True
        fig_dict = {}
    elif control_type == 'hybrid_position_control':
        snn_res = True
        mec_res = True
        fig_dict = {}
    else:
        raise ValueError(f'Unknown control type {control_type}')

    metrics = {}
    if snn_res:
        metrics = metrics | snn_sim.simulation_post_processing()
    if mec_res:
        metrics = metrics | mech_sim.simulation_post_processing()

    # Plot and save
    if plot_figures:

        if snn_res:
            snn_sim.simulation_plots()
        if mec_res:
            mech_sim.simulation_plots(fig_dict)

        plt.show(block=False)
        plt.pause(0.001)
        input("Enter to close plots and continue...")
        plt.close('all')
        if save_prompt:
            snn_sim.save_prompt(fig_dict)

    return metrics

def post_processing_single_net_single_run_open_loop(
        snn_sim     : SnnExperiment,
        plot_figures: bool,
        save_prompt : bool,
    ) -> dict[str, Union[float, np.ndarray[float]]]:
    '''
    Perform post-processing for a network model simulation in open loop.

    Args:
        snn_sim (SnnExperiment): Instance of the SnnExperiment class.
        plot_figures (bool): Whether to plot figures.
        save_prompt (bool): Whether to save prompts.

    Returns:
        dict[str, Union[float, np.ndarray[float]]]: Dictionary containing computed metrics.
    '''
    return _post_processing_single_net_single_run(
        control_type = 'open_loop',
        snn_sim      = snn_sim,
        mech_sim     = None,
        plot_figures = plot_figures,
        save_prompt  = save_prompt,
    )

def post_processing_single_net_single_run_closed_loop(
        snn_sim     : SnnExperiment,
        mech_sim    : MechPlotting,
        plot_figures: bool,
        save_prompt : bool,
    ) -> dict[str, Union[float, np.ndarray[float]]]:
    '''
    Perform post-processing for a network model simulation in closed loop.

    Args:
        snn_sim (SnnExperiment): Instance of the SnnExperiment class.
        mech_sim (MechPlotting): Instance of the MechPlotting class.
        plot_figures (bool): Whether to plot figures.
        save_prompt (bool): Whether to save prompts.

    Returns:
        dict[str, Union[float, np.ndarray[float]]]: Dictionary containing computed metrics.
    '''
    return _post_processing_single_net_single_run(
        control_type = 'closed_loop',
        snn_sim      = snn_sim,
        mech_sim     = mech_sim,
        plot_figures = plot_figures,
        save_prompt  = save_prompt,
    )

def post_processing_single_net_single_run_signal_driven(
        snn_sim     : SnnExperiment,
        mech_sim    : MechPlotting,
        plot_figures: bool,
        save_prompt : bool,
    ) -> dict[str, Union[float, np.ndarray[float]]]:
    '''
    Perform post-processing for a network model simulation driven by a signal.

    Args:
        snn_sim (SnnExperiment): Instance of the SnnExperiment class.
        mech_sim (MechPlotting): Instance of the MechPlotting class.
        plot_figures (bool): Whether to plot figures.
        save_prompt (bool): Whether to save prompts.

    Returns:
        dict[str, Union[float, np.ndarray[float]]]: Dictionary containing computed metrics.
    '''
    return _post_processing_single_net_single_run(
        control_type = 'signal_driven',
        snn_sim      = snn_sim,
        mech_sim     = mech_sim,
        plot_figures = plot_figures,
        save_prompt  = save_prompt,
    )

def post_processing_single_net_single_run_position_control(
        snn_sim     : SnnExperiment,
        mech_sim    : MechPlotting,
        plot_figures: bool,
        save_prompt : bool,
    ) -> dict[str, Union[float, np.ndarray[float]]]:
    '''
    Perform post-processing for a network model simulation in position_control.

    Args:
        snn_sim (SnnExperiment): Instance of the SnnExperiment class.
        mech_sim (MechPlotting): Instance of the MechPlotting class.
        plot_figures (bool): Whether to plot figures.
        save_prompt (bool): Whether to save prompts.

    Returns:
        dict[str, Union[float, np.ndarray[float]]]: Dictionary containing computed metrics.
    '''
    return _post_processing_single_net_single_run(
        control_type = 'position_control',
        snn_sim      = snn_sim,
        mech_sim     = mech_sim,
        plot_figures = plot_figures,
        save_prompt  = save_prompt,
    )

def post_processing_single_net_single_run_hybrid_position_control(
        snn_sim     : SnnExperiment,
        mech_sim    : MechPlotting,
        plot_figures: bool,
        save_prompt : bool,
    ) -> dict[str, Union[float, np.ndarray[float]]]:
    '''
    Perform post-processing for a network model simulation in hybrid position control.

    Args:
        snn_sim (SnnExperiment): Instance of the SnnExperiment class.
        mech_sim (MechPlotting): Instance of the MechPlotting class.
        plot_figures (bool): Whether to plot figures.
        save_prompt (bool): Whether to save prompts.

    Returns:
        dict[str, Union[float, np.ndarray[float]]]: Dictionary containing computed metrics.
    '''
    return _post_processing_single_net_single_run(
        control_type = 'hybrid_position_control',
        snn_sim      = snn_sim,
        mech_sim     = mech_sim,
        plot_figures = plot_figures,
        save_prompt  = save_prompt,
    )

