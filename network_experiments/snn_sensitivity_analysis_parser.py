'''
Functions to Parse Analysis Arguments for Neuromechanical Models

This module contains functions to parse command-line arguments for running analyses with neuromechanical models.
It provides a function to parse command-line arguments and return them as a dictionary.

Functions:
    - parse_arguments
'''
import argparse

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
        default = 20
    )
    parser.add_argument(
        "--n_processes_batch",
        action  = 'store',
        type    = int,
        help    = "Number of processes in a batch",
        default = 6
    )
    parser.add_argument(
        "--np_random_seed",
        action  = 'store',
        type    = int,
        help    = "Seed of the random number generator",
        default = 100
    )
    parser.add_argument(
        "--n_saltelli",
        action  = 'store',
        type    = int,
        help    = "N for the Saltelli sampler",
        default = 64
    )
    parser.add_argument(
        "--index_start",
        action  = 'store',
        type    = int,
        help    = "Index of the first simulation to run",
        default = None
    )
    parser.add_argument(
        "--index_finish",
        action  = 'store',
        type    = int,
        help    = "Index of the last simulation to run",
        default = None
    )
    return vars(parser.parse_args())