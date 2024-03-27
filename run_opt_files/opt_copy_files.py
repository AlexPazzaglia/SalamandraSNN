''' Replay previous simulation from an optimization process'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments import snn_simulation_data

TARGET_SPEED_DIRS = [
    'ps0_speed1',
    'ps0_speed2',
    'ps0_speed3',
    'ps4_speed1',
    'ps4_speed2',
    'ps4_speed3',
]

SEED_VALUES = [
    100,
    101,
]

def main():
    '''  Copy results from previous optimization process '''

    # Source
    results_path = '/data/pazzagli/simulation_results/data/optimization'

    max_speed_str = '{gait}/max_speed/net_farms_4limb_1dof_{gait}_{type}_max_speed_{seed}'
    source_files_max_speed = [
        max_speed_str.format(gait=gait, type=type, seed=seed)
        for gait in ['swimming', 'walking']
        for type in ['closedloop', 'openloop']
        for seed in SEED_VALUES
    ]

    target_speed_str = '{gait}/target_speed/net_farms_4limb_1dof_{gait}_target_speed_{seed}/{ps_speed_combination}'
    source_files_target_speed = [
        target_speed_str.format(gait=gait, seed=seed, ps_speed_combination=ps_speed_combination)
        for gait in ['swimming', 'walking']
        for seed in SEED_VALUES
        for ps_speed_combination in TARGET_SPEED_DIRS
    ]

    source_folders = source_files_max_speed + source_files_target_speed

    # Destination
    target_path = '/home/pazzagli/simulation_results/data'
    os.makedirs(target_path, exist_ok = True)

    # Copy files
    for source_folder in source_folders:
        snn_simulation_data.copy_folder(
            source_folder = f'{results_path}/{source_folder}',
            target_folder = f'{target_path}/{source_folder}',
            exist_ok      = True,

        )

    return

if __name__ == '__main__':
    main()
