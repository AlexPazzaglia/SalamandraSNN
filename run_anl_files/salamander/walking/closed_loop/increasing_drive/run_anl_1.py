''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
from brian2.units.allunits import second, msecond

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_utils import (
    prepend_date_to_tag,
    divide_params_in_batches
)
from network_experiments.snn_simulation import simulate_multi_net_multi_run_closed_loop
from network_experiments.snn_utils import MODELS_FARMS

import experimental_data.salamander_kinematics_muscles.muscle_parameters_optimization_load_results as mpl

from network_experiments.snn_analysis_parser import parse_arguments

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Shared by processes
    args = parse_arguments()

    results_path = args['results_path']
    model_name   = '4limb_1dof_unweighted'

    timestep = 1 * msecond
    duration = 10 * second

    simulation_data_file_tag = 'drive_effect'

    # Process parameters
    params_processes_shared = {
        'timestep'  : timestep,
        'duration'  : duration,
        'verboserun': False,
        'set_seed'  : True,

        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': False,

        'gaitflag'  : 1,
    }

    params_processes_variable = [
        {
        }
    ]

    # Get parameters of all processes
    n_network_replicas = 30

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

    # Params runs
    params_runs = [
        {
            'stim_a_mul'   : stim_a_mul,
            'ps_gain_axial': ps_gain,
        }
        for stim_a_mul in np.linspace(-2, 4, 31)
        for ps_gain in np.linspace(0, 200, 21)
    ]

    # Simulate
    tag_folder = prepend_date_to_tag('ANALYSIS')
    for params_processes_batch in params_processes_batches:
        simulate_multi_net_multi_run_closed_loop(
            modname             = MODELS_FARMS[model_name][0],
            parsname            = MODELS_FARMS[model_name][1],
            params_processes    = params_processes_batch,
            params_runs         = params_runs,
            tag_folder          = tag_folder,
            results_path        = results_path,
            delete_connectivity = True,
        )


if __name__ == '__main__':
    main()
