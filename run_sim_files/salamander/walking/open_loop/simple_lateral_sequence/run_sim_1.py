''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from brian2.units.allunits import second, msecond

from network_experiments.snn_simulation import simulate_single_net_multi_run_open_loop_build
from network_experiments.snn_utils import MODELS_OPENLOOP

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path = '/data/pazzagli/simulation_results'
    model_name   = '4limb_1dof_unweighted'

    timestep = 1 * msecond
    duration = 10 * second

    simulation_data_file_tag = 'lateral_sequence'

    # Process parameters
    params_process = {
        'timestep'  : timestep,
        'duration'  : duration,

        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': True,

        'gaitflag'  : 2,
    }

    # Params runs
    params_runs = [
        {
        }
    ]

    # Simulate
    simulate_single_net_multi_run_open_loop_build(
        modname             = default_params['modname'],
        parsname            = default_params['parsname'],
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = True,
        results_path        = results_path,
        delete_files        = True,
        delete_connectivity = False,
    )


if __name__ == '__main__':
    main()
