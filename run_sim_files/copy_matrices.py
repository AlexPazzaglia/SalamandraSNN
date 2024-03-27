''' Copy connectivity matrices from one folder to another '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments import snn_simulation_data

def main():
    ''' Copy connectivity matrices from one folder to another '''

    # Source
    results_path = '/data/pazzagli/simulation_results/data'
    folder_name  = 'net_farms_4limb_1dof_test_test'
    tag_process  = '1'

    # Destination
    destination_path = '/data/pazzagli/simulation_results/data'

    snn_simulation_data.copy_connectivity_matrices_from_process_data(
        folder_name      = folder_name,
        tag_process      = tag_process,
        results_path     = results_path,
        destination_path = destination_path,
    )

