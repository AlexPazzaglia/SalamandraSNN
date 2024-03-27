''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from network_experiments.snn_analysis_parser import parse_arguments
from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_utils import prepend_date_to_tag
from network_experiments.snn_simulation import simulate_multi_net_multi_run_closed_loop
import network_experiments.default_parameters.salamander.swimming.closed_loop.default as default

def run_experiment(
    ps_gain_axial           : float,
    simulation_data_file_tag: str,
    stim_a_off_array        : np.ndarray = np.linspace(-2, 5, 41),
    ps_weight_array         : np.ndarray = np.linspace(0.0, 3.0, 31),
    ind_run_start           : int = None,
    ind_run_finish          : int = None,
):
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Default parameters
    default_params = default.get_default_parameters()

    # Script arguments
    args         = parse_arguments()
    results_path = args['results_path']

    # Process parameters
    params_processes_shared = default_params['params_process'] | {

        'simulation_data_file_tag': simulation_data_file_tag,
        'ps_gain_axial'           : default.get_scaled_ps_gains(ps_gain_axial),
    }

    params_processes_variable = [
        {
        }
    ]

    # Params runs
    params_runs = [
        {
            'stim_a_off': stim_a_off,
            'ps_weight' : ps_weight,
        }
        for stim_a_off in stim_a_off_array
        for ps_weight in ps_weight_array
    ]

    ind_run_start  =                0 if ind_run_start  is None else ind_run_start
    ind_run_finish = len(params_runs) if ind_run_finish is None else ind_run_finish

    params_runs = params_runs[ind_run_start:ind_run_finish]

    # Get parameters of all processes
    n_network_replicas = 20

    (
        params_processes,
        params_processes_batches
    ) = get_params_processes(
        params_processes_shared,
        params_processes_variable,
        n_network_replicas,
        np_random_seed    = args['np_random_seed'],
        start_index       = args['index_start'],
        finish_index      = args['index_finish'],
        n_processes_batch = args['n_processes_batch'],
    )

    # Simulate
    tag_folder = prepend_date_to_tag('ANALYSIS')
    for params_processes_batch in params_processes_batches:
        simulate_multi_net_multi_run_closed_loop(
            modname             = f'{CURRENTDIR}/net_farms_limbless_ps_weight.py',
            parsname            = default_params['parsname'],
            params_processes    = params_processes_batch,
            params_runs         = params_runs,
            tag_folder          = tag_folder,
            results_path        = results_path,
            delete_connectivity = True,
        )
