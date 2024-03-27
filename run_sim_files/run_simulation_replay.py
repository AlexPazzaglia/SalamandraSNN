''' Replay a simulation'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import argparse
from network_experiments import snn_simulation, snn_simulation_replay

def main():
    ''' Replay a simulation'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        action  = 'store',
        type    = str,
        help    = "Results location",
        default = '/data/pazzagli/simulation_results'
    )
    args = vars(parser.parse_args())

    results_path = args['results_path']
    model_name   = '4limb_1dof'
    tag_analysis = 'walking_closedloop_saveall'
    tag_folder   = '2023_03_20_18_17_42_OPTIMIZATION'

    modname     = snn_simulation.MODELS_FARMS[model_name][0]
    parsname    = snn_simulation.MODELS_FARMS[model_name][1]

    run_params = {
        'ps_gain_axial'        : 9.611191500288077,
        'mech_activation_delays': 0.2617739361643794,
        'gains_drives_axis'    : [0.74315218, 1.48906856, 1.06780051, 0.64557257, 1.27971729, 1.19289151],
        'gains_drives_limbs'   : [0.54428265, 0.54428265, 0.54428265, 0.54428265],
        'mc_gain_axial'        : 0.7310460806187812,
        'mc_gain_limbs'        : 0.8073853001140241,
    }
    snn_simulation_replay.replicate_network_simulation(
        control_type = 'closed_loop',
        modname      = modname,
        parsname     = parsname,
        folder_name  = f'{modname}_{tag_analysis}_{tag_folder}',
        tag_folder   = tag_folder,
        tag_process  = '3',
        run_id       = None,
        run_params   = run_params,
        plot_figures = True,
        results_path = results_path,
    )


if __name__ == '__main__':
    main()