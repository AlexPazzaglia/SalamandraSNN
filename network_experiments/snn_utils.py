'''
Utility functions for the neuromechanical simulations

This module provides utility functions for handling various aspects of neuromechanical simulations.
It includes functions for file naming, parameter processing, and other useful tasks in the context of conducting simulations involving neuromechanical models.

Functions:
- prepend_date_to_tag(file_tag, date_time_str=None): Prepend the date and time to a filename to avoid overriding issues.
- get_configuration_files_from_folder_name(folder_name, model_name=None, open_loop=False): Get the configuration files from the model name.
- divide_params_in_batches(params_processes, n_processes_batch): Divide process parameters into batches.

'''

import os
import numpy as np

import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mc

from datetime import datetime

MODELS_OPENLOOP = {
          'test'           : [                 'net_openloop_test',                  'pars_simulation_openloop_test'],
    '4limb_1dof'           : [           'net_openloop_4limb_1dof',            'pars_simulation_openloop_4limb_1dof'],
    '4limb_4dof'           : [           'net_openloop_4limb_4dof',            'pars_simulation_openloop_4limb_4dof'],
    '4limb_1dof_unweighted': ['net_openloop_4limb_1dof_unweighted', 'pars_simulation_openloop_4limb_1dof_unweighted'],
    'zebrafish'            : [            'net_openloop_zebrafish',             'pars_simulation_openloop_zebrafish'],
    'zebrafish_simplified' : [ 'net_openloop_zebrafish_simplified',  'pars_simulation_openloop_zebrafish_simplified'],
    'zebrafish_v1'         : [ 'net_openloop_zebrafish_simplified',          'pars_simulation_openloop_zebrafish_v1'],
    'single_segment'       : [       'net_openloop_single_segment',        'pars_simulation_openloop_single_segment'],
}

MODELS_FARMS = {
    'test'                 : [                 'net_farms_test',                  'pars_simulation_farms_test'],
    '4limb_1dof'           : [           'net_farms_4limb_1dof',            'pars_simulation_farms_4limb_1dof'],
    '4limb_1dof_unweighted': ['net_farms_4limb_1dof_unweighted', 'pars_simulation_farms_4limb_1dof_unweighted'],
    '4limb_4dof'           : [           'net_farms_4limb_4dof',            'pars_simulation_farms_4limb_4dof'],
    'salamandra_limbless'  : [             'net_farms_limbless',              'pars_simulation_farms_limbless'],
    'zebrafish'            : [            'net_farms_zebrafish',             'pars_simulation_farms_zebrafish'],
    'zebrafish_simplified' : [ 'net_farms_zebrafish_simplified',  'pars_simulation_farms_zebrafish_simplified'],
}

# ------------ [ FILE NAME ] ------------
def prepend_date_to_tag(file_tag: str, date_time_str: str= None):
    '''
    Prepend the date and time to a filename to avoid overriding issues.

    Parameters:
    - file_tag (str): The original file tag.
    - date_time_str (str, optional): A formatted date and time string (default: current date and time).

    Returns:
    - str: The modified file tag with the date and time prepended.
    '''
    if date_time_str is None:
        date_time = datetime.now()
        date_time_str = date_time.strftime("%Y_%m_%d_%H_%M_%S")

    file_tag = f'{date_time_str}_{file_tag}'

    return file_tag

# \----------- [ FILE NAME ] ------------

# ------------ [ PARAMS PROCESSES ] ------------
def get_configuration_files_from_folder_name(
    folder_name: str,
    model_name : str = None,
    open_loop  : bool = False,
) -> tuple[str, str, str]:
    '''
    Get the configuration files from the model name.

    Parameters:
    - folder_name (str): The name of the folder.
    - model_name (str, optional): The name of the model (default: None).
    - open_loop (bool, optional): Flag indicating open-loop simulation (default: False).

    Returns:
    - tuple[str, str, str]: A tuple containing model name, parameters name, and tag folder.
    '''

    models_list = MODELS_FARMS if not open_loop else MODELS_OPENLOOP

    if model_name is None:
        # Matchin names
        mod_ind_l = [ x for x in models_list if x in folder_name ]
        # Longest matches
        n_longest_match = max( [len(x) for x in mod_ind_l] )
        mod_ind_l       = [ x for x in mod_ind_l if len(x) == n_longest_match]

        if len(mod_ind_l) != 1:
            raise ValueError(
                f'''
                Unable to determine the model and parameters name from the folder name.
                folder_name: {folder_name}
                mod_ind_l: {mod_ind_l}
                '''
            )
        model_name = mod_ind_l[0]

    modname    = models_list[ model_name ][0]
    parsname   = models_list[ model_name ][1]
    tag_folder = folder_name.split(modname)[-1][1:]

    return modname, parsname, tag_folder

def divide_params_in_batches(
    params_processes  : list[dict],
    n_processes_batch : int,
) -> list[list[dict]]:
    '''
    Divide process parameters into batches.

    Parameters:
    - params_processes (list[dict]): List of process parameters.
    - n_processes_batch (int): Number of processes per batch.

    Returns:
    - list[list[dict]]: A list of batches, each containing a list of process parameters.
    '''

    n_processes = len(params_processes)
    batches = n_processes // n_processes_batch + 1

    # Numerosity
    n_processes_batches = np.array(
        [
            n_processes_batch
            for batch in range(batches)
        ]
    )
    n_processes_batches[-1] -= n_processes_batch * batches - n_processes

    if n_processes_batches[-1] == 0:
        n_processes_batches = n_processes_batches[:-1]
        batches -= 1

    # Distribute parameters
    params_processes_batches = [
        params_processes[
            np.sum(n_processes_batches[:batch]) : np.sum(n_processes_batches[:batch+1])
        ]
        for batch in range(batches)
    ]

    return params_processes_batches

# \----------- [ PARAMS PROCESSES ] ------------

# ------------ [ COLORS FOR PLOTS ] ------------
def modify_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# \----------- [ COLORS FOR PLOTS ] ------------

# ------------ [ SAVE FIGURES ] ------------
def save_all_figures(folder_name : str, results_path : str):
    ''' Save all currently defined figures '''
    folder = f'{results_path}/{folder_name}'
    print(f'Saving figures to {folder}')

    os.makedirs( folder, exist_ok=True)
    for i in plt.get_fignums():
        fig     = plt.figure(i)
        figname = fig.canvas.manager.get_window_title().replace(' - ', '_')
        plt.savefig(f'{folder}/{figname}.pdf')

def save_prompt(
    folder_name : str,
    results_path: str,
):
    ''' User pront for saving figures '''

    save = input('Save figures? [Y/n] - ')

    if save in ['y','Y','1']:
        folder_tag  = input('Analysis tag: ')

        os.makedirs(results_path.format('images'), exist_ok=True)
        save_all_figures(
            f'{folder_name}_{folder_tag}',
            results_path = results_path.format('images'),
        )

# \----------- [ SAVE FIGURES ] ------------