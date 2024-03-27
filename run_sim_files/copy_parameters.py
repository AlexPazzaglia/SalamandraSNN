''' Copy parameters '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import argparse
import dill

def main():
    ''' Copy parameters'''

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

    # Optimization objectives
    obj_optimization = [
        ['speed_fwd', 'max'],
        [     'ptcc', 'max'],
    ]

    # Optimization constraints
    constr_optimization = {}

    ## OPEN LOOP
    pars_optimization = [
        (    'gains_drives_axis', 0.5, 1.5),
        (        'mc_gain_axial', 0.1, 1.0),
    ]

    folder = 'net_farms_4limb_1dof_swimming_openloop_2obj_2023_03_10_09_04_19_OPTIMIZATION'
    filename = f'{results_path}/data/{folder}/parameters_optimization.dill'
    with open(filename, 'wb') as outfile:
        dill.dump(pars_optimization,   outfile)
        dill.dump(obj_optimization,    outfile)
        dill.dump(constr_optimization, outfile)


    ## CLOSED LOOP
    pars_optimization = [
        (        'ps_gain_axial', 0.0, 3.0),
        (    'gains_drives_axis', 0.5, 1.5),
        (        'mc_gain_axial', 0.1, 1.0),
    ]

    folder = 'net_farms_4limb_1dof_swimming_closedloop_2obj_2023_03_11_03_05_44_OPTIMIZATION_copy'
    filename = f'{results_path}/data/{folder}/parameters_optimization.dill'
    with open(filename, 'wb') as outfile:
        dill.dump(pars_optimization,   outfile)
        dill.dump(obj_optimization,    outfile)
        dill.dump(constr_optimization, outfile)

if __name__ == '__main__':
    main()
