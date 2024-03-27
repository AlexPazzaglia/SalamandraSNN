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

from network_modules.parameters.network_parameters import SnnParameters
import network_experiments.default_parameters.salamander.swimming.closed_loop.default as default

def get_run_params(
    params_name      : str,
    results_path     : str,
    noise_level_array: np.ndarray,
    ps_weight_array  : np.ndarray,
):
    ''' Get run parameters for varying noise level '''

    default_snn_pars = SnnParameters(
        parsname     = params_name,
        results_path = results_path,
    )

    indices_cpg = default_snn_pars.topology.network_modules['cpg']['axial'].indices

    # Replace neural variables
    params_run = [
        {
            'scalings' : {
                'neural_params' : [
                    {
                        'neuron_group_ind': 0,
                        'var_name'        : 'sigma',
                        'indices'         : indices_cpg,
                        'nominal_value'   : [1.0, 'mV'],
                        'scaling'         : noise_level,
                    }
                ],
            },
            'ps_weight': ps_weight,
        }
        for noise_level in noise_level_array
        for ps_weight in ps_weight_array
    ]

    return params_run

def run_experiment(
    ps_gain_axial           : float,
    simulation_data_file_tag: str,
    noise_level_array       : np.ndarray = np.linspace(0, 15, 31),
    ps_weight_array         : np.ndarray = np.linspace(0.0, 10.0, 41),
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
        'noise_term'              : True,
    }

    params_processes_variable = [
        {
        }
    ]

    # Params runs
    params_runs = get_run_params(
        params_name       = default_params['parsname'],
        results_path      = results_path,
        noise_level_array = noise_level_array,
        ps_weight_array   = ps_weight_array,
    )

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
