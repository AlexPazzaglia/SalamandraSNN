''' Replay previous simulation from an optimization process'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from network_experiments import snn_optimization_results

def main():
    '''  Replay previous simulation from an optimization process '''

    # results_path = '/data/pazzagli/simulation_results/data/optimization/swimming/target_speed'
    # folder_name  = 'net_farms_4limb_1dof_swimming_target_speed_100/ps4_speed3'

    folder_aux_max = '/data/pazzagli/simulation_results/data/optimization/{gait}/{type}_speed'
    folder_aux_trg = '/data/pazzagli/simulation_results/data/optimization/{gait}/{type}_speed/net_farms_4limb_1dof_{gait}_{type}_speed_{seed}'

    optimization_paths = [

        # Max
        # [ folder_aux_max.format(gait='swimming', type='max') , 'net_farms_4limb_1dof_swimming_openloop_max_speed_100', ],
        # [ folder_aux_max.format(gait='swimming', type='max') , 'net_farms_4limb_1dof_swimming_openloop_max_speed_101', ],
        # [ folder_aux_max.format(gait='walking', type='max') , 'net_farms_4limb_1dof_walking_openloop_max_speed_100', ],
        # [ folder_aux_max.format(gait='walking', type='max') , 'net_farms_4limb_1dof_walking_openloop_max_speed_101', ],

        # Target
        # [ folder_aux_trg.format(gait='swimming', type='target', seed=100),  'ps0_speed2', ],
        # [ folder_aux_trg.format(gait='swimming', type='target', seed=100),  'ps4_speed3', ],
        [ folder_aux_trg.format(gait='walking', type='target', seed=100),  'ps0_speed2', ],
        # [ folder_aux_trg.format(gait='walking', type='target', seed=100),  'ps4_speed2', ],
    ]

    process_ind  = 0

    constr_additional = {
        # 'speed_fwd' : [0.2, None],
        # 'ptcc_ax' : [1.5, None],
    }

    metric_keys = [
        'speed_fwd',
        # 'cot',
    ]

    new_pars_prc = {

        'shared_mc_neural_params' : {
            'tau_mc_act'  : [100.0,  'ms'],
            'tau_mc_deact': [100.0,  'ms'],
            'w_ampa'      : [0.60,    ''],
            'w_nmda'      : [0.15,    ''],
            'w_glyc'      : [1.00,    ''],
        }

    }

    # [0.73137806, 0.60700062, 0.84005937, 0.98322616, 1.31282731, 0.65432231]

    # np.linspace(1.1, 0.9, 6)

    new_pars_run = {

        # 'gains_drives_axis': np.ones(6),
        # 'ps_gain_axial'    : 0.0,

        'stim_l_off' : 2.0

    }

    # Model with lowest COT
    for results_path, folder_name in optimization_paths:
        for metric_key in metric_keys:
            snn_optimization_results.run_best_individual(
                results_path      = results_path,
                folder_name       = folder_name,
                inds_processes       = process_ind,
                inds_generations        = -1,
                metric_key        = metric_key,
                tag_run           = f'best_{metric_key}',
                constr_additional = constr_additional,
                load_connectivity = True,
                new_pars_prc      = new_pars_prc,
                new_pars_run      = new_pars_run,
            )

    return

if __name__ == '__main__':
    main()
