'''
Functions to Parse Optimization Arguments for Neuromechanical Models

This module contains functions to parse command-line arguments for running optimization with neuromechanical models.
It provides a function to parse command-line arguments and return them as a dictionary.

Functions:
    - parse_arguments
'''

import argparse
from network_experiments import snn_optimization_problem

# # NOTE: For testing
# args['results_path']    = '/data/pazzagli/simulation_results_test'
# args['n_processes']     = 1
# args['n_sub_processes'] = 2
# args['n_gen']           = 2
# args['pop_size']        = 4

def parse_arguments(new_args : list[dict] = None):
    '''
    Parse command-line arguments for optimization.

    This function parses command-line arguments using the argparse library and returns them as a dictionary.

    Parameters:
    - new_args (list[dict]): Additional arguments to parse (default: None).

    Returns:
    - dict: Parsed command-line arguments as a dictionary.
    '''

    if new_args is None:
        new_args = []

    parser = argparse.ArgumentParser()

    for arg in new_args:
        parser.add_argument(
            f"--{arg['name']}",
            action  = 'store',
            type    = arg['type'],
            default = arg['default'],
        )

    parser.add_argument(
        "--results_path",
        action  = 'store',
        type    = str,
        help    = "Results location",
        default = '/data/pazzagli/simulation_results'
    )
    parser.add_argument(
        "--n_processes",
        action  = 'store',
        type    = int,
        help    = "Number of processes",
        default = 1
    )

    parser.add_argument(
        "--n_sub_processes",
        action  = 'store',
        type    = int,
        help    = "Number of sub processes",
        default = 6
    )
    parser.add_argument(
        "--pop_size",
        action  = 'store',
        type    = int,
        help    = "Size of a population",
        default = snn_optimization_problem.DEFAULT_PARAMS['pop_size']          # 50
    )
    parser.add_argument(
        "--n_gen",
        action  = 'store',
        type    = int,
        help    = "Number of generations",
        default = snn_optimization_problem.DEFAULT_PARAMS['n_gen']             # 300
    )
    parser.add_argument(
        "--np_random_seed",
        action  = 'store',
        type    = int,
        help    = "Seed of the random number generator",
        default = snn_optimization_problem.DEFAULT_PARAMS['np_random_seed']    # 100
    )

    return vars(parser.parse_args())