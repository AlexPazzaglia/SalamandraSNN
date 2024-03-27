''' Replay previous simulation from an optimization process'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments import snn_simulation_data

def main():
    '''  Copy results from previous optimization process '''

    folder_name = f'home/pazzagli/simulation_results'
    snn_simulation_data.delete_folder(folder_name)

    return

if __name__ == '__main__':
    main()
