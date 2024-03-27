'''
Utility Functions for logging in Neuromechanical Simulations

This module contains utility functions for setting up logging functionality in neuromechanical simulations.

Functions:
    - define_logging
'''

import os
import copy
import json
import logging

import numpy as np

from typing import Any, Callable

# Logging
def define_logging(
    modname        : str,
    tag_folder     : str,
    tag_process    : str,
    results_path   : str,
    tag_sub_process: str  = None,
    data_file_tag  : str  = None,
    verbose        : bool = True,
) -> None:
    '''
    Define logging functionality for simulations.

    This function sets up personalized logging for neuromechanical simulations. It creates log files based on the provided parameters and configures logging to write messages to these files.

    Parameters:
    - modname (str): Name of the module for which logging is being defined.
    - tag_folder (str): Tag representing the folder for log files.
    - tag_process (str): Tag representing the process for log files.
    - results_path (str): Path where log files and folders will be saved.
    - tag_sub_process (str, optional): Tag representing the sub-process for log files (default: None).
    - data_file_tag (str, optional): Tag representing data file for log files (default: None).
    - verbose (bool, optional): Flag for verbose logging (default: True).
    '''

    # Remove undesired logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Define personalized logging
    log_folder_path = '_'.join(
        [
            str(x) for x in [
                f'{results_path}/logs/{modname}',
                data_file_tag,
                tag_folder,
            ]
            if x is not None
        ]
    )
    os.makedirs(log_folder_path, exist_ok=True)

    process_id = '_'.join(
        [
            str(x) for x in [ tag_process, tag_sub_process ]
            if x is not None
        ]
    )

    log_file_path = f'{log_folder_path}/process_{process_id}.log'

    logging.basicConfig(
        filename = log_file_path,
        filemode = 'w',
        format   = '%(asctime)s - %(levelname)s - %(name)s : %(message)s',
        datefmt  = '%d-%b-%y %H:%M:%S',
        level    = logging.INFO
    )

    if verbose:
        logging.info(f'Process {process_id} logging to {log_file_path}')
        print(f'Process {process_id} logging to {log_file_path}')

    return

# Serialization

def _serialize_dict(dict_non_ser : dict):
    '''
    Make a dictionary serializable by converting numpy arrays to lists.
    '''
    dict_ser : dict = copy.deepcopy(dict_non_ser)

    keys_to_pop = []
    for key, val in dict_ser.items():
        if isinstance(val, np.ndarray):
            dict_ser[key] = val.tolist()
        if isinstance(val, dict):
            dict_ser[key] = _serialize_dict(val)
        if isinstance(val, Callable):
            keys_to_pop.append(key)

    for key in keys_to_pop:
        dict_ser.pop(key)

    return dict_ser

def pretty_string(value: Any):
    ''' Get a string representation of a value for logging '''
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), indent=4)

    if isinstance(value, dict):
        return json.dumps(_serialize_dict(value), indent=4)

    return str(value)

