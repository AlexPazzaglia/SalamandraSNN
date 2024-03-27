''' Utility functions for the neuromechanical simulations '''


import os
import sys
import dill
import shutil
import logging

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]

from network_modules.experiment.network_experiment import SnnExperiment

# ------------ [ DATA SAVING ] ------------
def _save_calling_file(dst_path: str):
    '''
    Save the calling script to the specified path.

    Args:
        dst_path (str): Destination path to save the script.
    '''
    file_src = sys.argv[0]
    file_dst = f'{dst_path}/{os.path.basename(file_src)}'
    logging.info('Copying %s file to %s', file_src,  file_dst)
    shutil.copyfile(file_src, file_dst)

def save_parameters_process(
        snn_sim          : SnnExperiment,
        params_process   : dict,
        params_runs      : list[dict],
    ) -> None:
    '''
    Save the parameters for multiple runs of a network.

    Args:
        snn_sim (SnnExperiment): Instance of SnnExperiment representing the neuromechanical simulation.
        params_process (dict): Parameters for the simulation process.
        params_runs (list[dict]): List of parameters for individual simulation runs.
    '''

    data_folder_path = snn_sim.params.simulation.results_data_folder_process
    os.makedirs(data_folder_path, exist_ok=True)

    # Called script
    _save_calling_file(data_folder_path)

    # Configuration files
    for pars_object in snn_sim.params.parameters_objects_list:
        pars_object.save_yaml_files(data_folder_path)

    # Process and runs parameters
    data_file = f'{data_folder_path}/snn_parameters_process.dill'
    logging.info('Saving snn_parameters_process data to %s', data_file)

    with open(data_file, "wb") as outfile:
        dill.dump(params_process, outfile)
        dill.dump(params_runs, outfile)

    return

def save_data_process(
    snn_sim       : SnnExperiment,
    metrics_runs  : dict[str, NDArrayFloat],
) -> None:
    '''
    Save the data from multiple runs of a network.

    Args:
        snn_sim (SnnExperiment): Instance of SnnExperiment representing the neuromechanical simulation.
        metrics_runs (dict[str, NDArrayFloat]): Metrics data for the simulation runs.
    '''

    logs_folder_path = snn_sim.params.simulation.logging_data_folder_process
    data_folder_path = snn_sim.params.simulation.results_data_folder_process
    logs_file_name   = f'process_{snn_sim.params.simulation.tag_process}.log'
    os.makedirs(data_folder_path, exist_ok=True)

    # Log file
    logs_folder_dest = f'{data_folder_path}/logs'
    logging.info(
        'Copying logs file %s from %s to %s',
        logs_file_name,
        logs_folder_path,
        logs_folder_dest
    )

    os.makedirs(logs_folder_dest, exist_ok=True)
    shutil.copyfile(
        f'{logs_folder_path}/{logs_file_name}',
        f'{logs_folder_dest}/{logs_file_name}',
    )

    # Performance
    data_file = f'{data_folder_path}/snn_performance_process.dill'
    logging.info('Saving snn_performance_process data to %s', data_file)

    with open(data_file, "wb") as outfile:
        dill.dump(metrics_runs, outfile)

    return

# \----------- [ DATA SAVING ] ------------

# ------------ [ DATA DELETING ] ------------
def delete_folder(
    source_folder: str,
):
    '''
    Delete an optimization folder.

    Args:
        source_folder (str): Path to the folder to be deleted.
    '''
    if 'data/pazzagli/simulation_results/data' in source_folder:
        raise ValueError('Cannot programmatically delete files from data folder, do it manually')
    shutil.rmtree(source_folder)
    return

def delete_state_files(
    snn_sim            : SnnExperiment,
    delete_connectivity: bool = False,
) -> None:
    '''
    Delete state and simulation files from the process folder.

    Args:
        snn_sim (SnnExperiment): Instance of SnnExperiment representing the neuromechanical simulation.
        delete_connectivity (bool, optional): Whether to delete connectivity files. Defaults to False.
    '''
    data_folder_path = snn_sim.params.simulation.results_data_folder_process

    files_to_remove = [
        'network_state.brian',
    ]

    if delete_connectivity:
        files_to_remove.append('snn_connectivity_indices.dill')

    for file in files_to_remove:
        file_path = f'{data_folder_path}/{file}'
        if not os.path.isfile(file_path):
            continue

        logging.info('Deleting %s from %s', file, file_path)
        os.remove(file_path)

    folders_to_remove = [
        'run_0'
    ]
    for folder in folders_to_remove:
        folder_path = f'{data_folder_path}/{folder}'
        if not os.path.isdir(folder_path):
            continue

        logging.info('Deleting %s from %s', folder, folder_path)
        shutil.rmtree(folder_path)

    return

# \----------- [ DATA DELETING ] ------------

# ------------ [ DATA LOADING ] ------------
def load_parameters_process(
    folder_name : str,
    tag_process : str,
    results_path: str,
    full_path   : str = None,
) -> tuple[dict, list[dict]]:
    '''
    Load parameters and runs data for a specific process.

    Args:
        folder_name (str): Name of the simulation folder.
        tag_process (str): Tag for the simulation process.
        results_path (str): Path to the results folder.
        full_path (str, optional): Full path to the data file. Defaults to None.

    Returns:
        tuple[dict, list[dict]]: Loaded parameters and runs data.
    '''
    if full_path is None:
        data_folder_path = f'{results_path}/{folder_name}/process_{tag_process}'
        data_file = f'{data_folder_path}/snn_parameters_process.dill'
    else:
        data_file = full_path

    logging.info('Loading snn_parameters_process data from %s', data_file)

    with open(data_file, "rb") as infile:
        params_process: dict       = dill.load(infile)
        params_runs   : list[dict] = dill.load(infile)

    return params_process, params_runs

def load_parameters_processes(
    folder_name   : str,
    tag_processes : list[str],
    results_path  : str,
) -> tuple[list[dict], list[list[dict]]]:
    '''
    Load parameters and runs data for multiple processes.

    Args:
        folder_name (str): Name of the simulation folder.
        tag_processes (list[str]): Tags for the simulation processes.
        results_path (str): Path to the results folder.

    Returns:
        tuple[list[dict], list[list[dict]]]: Loaded parameters and runs data for multiple processes.
    '''
    params_processes      = []
    params_runs_processes = []

    for tag_process in tag_processes:
        params_process, params_runs = load_parameters_process(
            folder_name,
            tag_process,
            results_path,
        )
        params_processes.append(params_process)
        params_runs_processes.append(params_runs)

    return params_processes, params_runs_processes

# \----------- [ DATA LOADING ] ------------

# ------------ [ DATA COPYING ] ------------
def copy_folder(
    source_folder: str,
    target_folder: str,
    exist_ok     : bool = False,
):
    '''
    Copy a folder to a target location.

    Args:
        source_folder (str): Source folder path.
        target_folder (str): Target folder path.
        exist_ok (bool, optional): Whether to copy even if the target folder already exists. Defaults to False.
    '''
    shutil.copytree(source_folder, target_folder, dirs_exist_ok=exist_ok)
    return

def copy_connectivity_matrices_from_process_data(
    folder_name     : str,
    tag_process     : str,
    results_path    : str,
    destination_path: str,
):
    '''
    Copy connectivity matrices from source to target directory.

    Args:
        folder_name (str): Name of the simulation folder.
        tag_process (str): Tag for the simulation process.
        results_path (str): Path to the results folder.
        destination_path (str): Destination path for copying the file.
    '''

    filename = f'{results_path}/{folder_name}/process_{tag_process}/snn_connectivity_indices.dill'
    shutil.copyfile(filename, destination_path)
    return

# \----------- [ DATA COPYING ] ------------
