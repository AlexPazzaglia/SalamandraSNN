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
import ps_connections

MAX_PS_RANGE = 0.025

def run_experiment(
    ps_gain_axial  : float,
    ps_type        : str,
    ps_weight_array: np.ndarray = np.linspace(0.0, 3, 31),
    ps_range_array : np.ndarray = np.linspace(0.0, MAX_PS_RANGE, 21),
    ind_run_start  : int = None,
    ind_run_finish : int = None,
):
    ''' Run the spinal cord model together with the mechanical simulator '''

    simulation_data_file_tag = (
        'feedback_topology_strength_swimming_4dim_'
        f'{ps_type}_'
        f'alpha_fraction_th_{round(ps_gain_axial * 100) :03d}'
    )

    # Default parameters
    default_params = default.get_default_parameters()

    # Script arguments
    args         = parse_arguments()
    results_path = args['results_path']

    # Process parameters
    params_processes_shared = default_params['params_process'] | {
        'simulation_data_file_tag': simulation_data_file_tag,
        'ps_gain_axial'           : default.get_scaled_ps_gains(ps_gain_axial),
        'connectivity_axial_newpars' : {
            'ps2ax': [
                ps_connections.get_ps_to_ax_in(amp= 0.5, max_ps_range = MAX_PS_RANGE),
                ps_connections.get_ps_to_mn_ex(amp= 0.5, max_ps_range = MAX_PS_RANGE),
                ps_connections.get_ps_to_ax_ex(amp= 0.5, max_ps_range = MAX_PS_RANGE),
                ps_connections.get_ps_to_mn_in(amp= 0.5, max_ps_range = MAX_PS_RANGE),
            ]
        }
    }

    params_processes_variable = [
        {
        }
    ]

    # Params runs
    params_runs = (
        [
            {
                'sig_ex_w'  : sig_w if ps_type in ['ex_up', 'ex_dw'] else 0,
                'sig_ex_up' : sig_r if ps_type in ['ex_up'] else 0,
                'sig_ex_dw' : sig_r if ps_type in ['ex_dw'] else 0,
                'sig_in_w'  : sig_w if ps_type in ['in_up', 'in_dw'] else 0,
                'sig_in_up' : sig_r if ps_type in ['in_up'] else 0,
                'sig_in_dw' : sig_r if ps_type in ['in_dw'] else 0,
            }
            for sig_w  in ps_weight_array
            for sig_r in ps_range_array
        ]
    )

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
            modname             = f'{CURRENTDIR}/net_farms_limbless_feedback_topology.py',
            parsname            = default_params['parsname'],
            params_processes    = params_processes_batch,
            params_runs         = params_runs,
            tag_folder          = tag_folder,
            results_path        = results_path,
            delete_connectivity = True,
        )

