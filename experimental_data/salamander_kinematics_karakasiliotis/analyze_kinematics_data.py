''' Read kinematics data '''

import os
import sys
import json
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import csv
import copy
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

from network_modules.plotting import plots_utils
from network_modules.performance import (
    signal_processor_mech,
    signal_processor_mech_plot,
)

FOLDER_PATH = 'experimental_data/salamander_kinematics_karakasiliotis'

def save_dict_to_files(
        dict_metrics : dict[str, np.ndarray],
        file_path,
):
    """
    Save a dictionary containing arrays to both a text file and a pickle file.

    Parameters:
        dictionary (dict): A dictionary where the keys are identifiers and the values are arrays (lists or NumPy arrays).
        txt_file_path (str): The file path where the dictionary will be saved in text format.
        pickle_file_path (str): The file path where the dictionary will be saved in pickle format.

    Returns:
        None
    """
    # Write dictionary to text file
    txt_file_path = f'{file_path}/gait_metrics.txt'
    with open(txt_file_path, mode='w') as txt_file:
        dict_metrics_list = copy.deepcopy(dict_metrics)

        for key, metric in dict_metrics_list.items():
            if not isinstance(metric, np.ndarray):
                continue
            dict_metrics_list[key] = list(metric)

        txt_file.write(json.dumps(dict_metrics_list, indent=4))

    # Save dictionary to pickle file
    pickle_file_path = f'{file_path}/gait_metrics.pickle'
    with open(pickle_file_path, mode='wb') as pickle_file:
        pickle.dump(dict_metrics, pickle_file)

###############################################################################
####### INERTIAL PROPERTIES ###################################################
###############################################################################

def get_inertial_seen_by_joints(plot_results = False, figures_dict: dict[str, list] = {}):
    ''' Get muscle scaling factors '''

    links_ref_frames_x = np.array(
        [
            0.0000,
            0.0150,
            0.0257,
            0.0364,
            0.0471,
            0.0578,
            0.0685,
            0.0792,
            0.0899,
        ]
    )

    joints_x = np.array(
        [
            0.0150,
            0.0257,
            0.0364,
            0.0471,
            0.0578,
            0.0685,
            0.0792,
            0.0899,
        ]
    )

    links_masses = np.array(
        [
            0.0003314,
            0.0005139,
            0.0006302,
            0.0005277,
            0.0002672,
            0.0002091,
            0.0001574,
            0.0001188,
            0.0000754,
        ]
    )

    links_inertias = np.array(
        [
            4.2878e-09,
            7.8835e-09,
            1.1056e-08,
            8.6788e-09,
            3.3960e-09,
            2.3738e-09,
            1.7043e-09,
            1.2165e-09,
            5.0731e-10,
        ]
    )

    n_joints_axis = 8

    inertias_in_joints_pos = np.array(
        [
            [
                links_inertias[link] + links_masses[link] * (joints_x[joint] - links_ref_frames_x[link]) ** 2
                for link in range(n_joints_axis+1)
            ]
            for joint in range(n_joints_axis)
        ]
    )

    inertias_in_joint_pos_left = np.array(
        [
            np.sum( inertias_in_joints_pos[joint, :joint+1] )
            for joint in range(n_joints_axis)
        ]
    )

    inertias_in_joint_pos_right = np.array(
        [
            np.sum( inertias_in_joints_pos[joint, joint+1:] )
            for joint in range(n_joints_axis)
        ]
    )

    inertias_seen_by_joints = np.amin(
        [
            inertias_in_joint_pos_left,
            inertias_in_joint_pos_right
        ],
        axis=0
    )

    if not plot_results:
        return inertias_seen_by_joints

    fig = plt.figure('Inertias seen by joints')
    plt.plot(inertias_seen_by_joints)
    plt.yscale('log')
    plt.xlabel('Joint index [#]')
    plt.ylabel('Inertia [kg m^2]')
    plt.title('Inertias seen by joints')
    plt.grid()

    figures_dict['inertias_seen_by_joints'] = fig

    return inertias_seen_by_joints

def compute_inertial_torques(
    joint_angles: np.ndarray[float],
    plot_results: bool = False,
    figures_dict: dict[str, list] = {},
):
    ''' Compute inertial torques '''

    frequency = 2
    timestep  = 1 / ( frequency * joint_angles.shape[0] )
    times     = np.arange(joint_angles.shape[0]) * timestep

    joint_angles_dt  = np.gradient(joint_angles, times, axis=0)

    links_angles     = np.cumsum(joint_angles, axis= 1)
    links_angles_dt  = np.gradient(links_angles, times, axis=0)
    links_angles_ddt = np.gradient(links_angles_dt, times, axis=0)

    inertias         = get_inertial_seen_by_joints(plot_results, figures_dict)
    inertial_torques = inertias * links_angles_ddt[:, :len(inertias)]

    # Low pas filter the inertial torques
    cutoff_frequency  = frequency + 1
    filter_order      = 4
    cutoff_normalized = 2 * cutoff_frequency * timestep

    b, a = butter(filter_order, cutoff_normalized, btype='low')

    inertial_torques_filtered = filtfilt(b, a, inertial_torques, axis=0)

    if not plot_results:
        return inertial_torques_filtered

    fig = plt.figure('Inertial torques')
    plt.plot(times, inertial_torques_filtered)
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.title('Inertial torques')
    plt.grid()

    figures_dict['inertial_torques'] = fig

    return inertial_torques_filtered

###############################################################################
####### ANALYSIS ##############################################################
###############################################################################

def analyze_joint_data(
    link_lengths: list[float],
    joint_angles: np.ndarray[float],
    results_path: str,
):
    '''
    Analyze joint data.
    Computes the Cartesian coordinates of each joint in a kinematic chain.
    '''

    num_joints   = len(link_lengths)
    joint_angles = np.radians(joint_angles)
    num_steps    = joint_angles.shape[0]

    # Initialize an array to store the Cartesian coordinates of each joint
    joint_coordinates = np.zeros((num_steps, num_joints, 2))

    # Set the position of the first joint as the origin
    joint_coordinates[:, 0] = [0, 0]

    # Iterate over each step in the joint angles array
    for step in range(num_steps):
        # Compute the cumulative angles for each joint at the current step
        cumulative_angles = np.array(
            [
                np.sum( joint_angles[step, :joint] )
                for joint in range(0, num_joints)
            ]
        )

        # Compute the Cartesian coordinates for each joint using the link lengths and cumulative angles
        for i in range(1, num_joints):
            x = joint_coordinates[step, i - 1, 0] + link_lengths[i - 1] * np.cos(cumulative_angles[i - 1])
            y = joint_coordinates[step, i - 1, 1] + link_lengths[i - 1] * np.sin(cumulative_angles[i - 1])
            joint_coordinates[step, i] = [x, y]

    # Compute metrics for the joint angles
    (
        _joints_positions,
        joints_displacements_mean,
        joints_displacements_std,
        joints_displacements_amp,
    ) = signal_processor_mech.compute_joints_displacements_metrics(
        joints_positions= joint_angles,
        n_joints_axis   = num_joints,
    )

    # Compute metrics for the joint coordinates
    (
        _links_displacements,
        links_displacements_mean,
        links_displacements_std,
        links_displacements_amp,
        links_displacements_data,
    ) = signal_processor_mech.compute_links_displacements_metrics(
        points_positions = joint_coordinates,
        n_points_axis    = num_joints,
    )

    # Save metrics to files
    metrics_dict = {
        'joints_displacements_mean' : joints_displacements_mean,
        'joints_displacements_std'  : joints_displacements_std,
        'joints_displacements_amp'  : joints_displacements_amp,
        'links_displacements_mean'  : links_displacements_mean,
        'links_displacements_std'   : links_displacements_std,
        'links_displacements_amp'   : links_displacements_amp,
    }
    save_dict_to_files(
        dict_metrics = metrics_dict,
        file_path    = results_path,
    )

    return joint_coordinates, links_displacements_data

def analyze_gait_data(gait, plot_results = False):
    ''' Analyze gait data '''

    filename_data_input  = f'{FOLDER_PATH}/{gait}/data_karakasiliotis_processed.csv'
    filename_data_output = f'{FOLDER_PATH}/{gait}/results'

    n_limbs        = 4
    n_joints_axial = 8

    figures_dict = {}

    # Load data
    kinematics_data    = pd.read_csv(filename_data_input)
    times              = kinematics_data['time'].to_numpy()
    names_joints_axial = kinematics_data.columns[1:1 + n_joints_axial].to_list()
    names_joints_limbs = kinematics_data.columns[1 + n_joints_axial:].to_list()
    joint_angles       = kinematics_data[names_joints_axial + names_joints_limbs].to_numpy()
    joint_angles_ax    = kinematics_data[names_joints_axial].to_numpy()
    joint_angles_lb    = kinematics_data[names_joints_limbs].to_numpy()

    n_joints_limbs = joint_angles_lb.shape[1]
    n_joints_limb  = joint_angles_lb.shape[1] // n_limbs
    n_joints       = n_joints_axial + n_joints_limbs

    # Compute joint coordinates
    link_lengths = np.diff(
        [
            0.0000,
            0.0150,
            0.0257,
            0.0364,
            0.0471,
            0.0578,
            0.0685,
            0.0792,
            0.0899,
        ]
    )
    joint_coordinates, links_displacements_data = analyze_joint_data(
        link_lengths = link_lengths,
        joint_angles = joint_angles_ax,
        results_path = filename_data_output,
    )

    # Compute inertial properties
    inertial_torques_filtered = compute_inertial_torques(
        joint_angles = joint_angles,
        plot_results = True,
        figures_dict = figures_dict,
    )

     # PLOTTING
    if not plot_results:
        return

    # Plot joint angles
    fig_joints_dict = signal_processor_mech_plot.plot_joints_angles(
        times           = times,
        joints_angles   = joint_angles,
        n_joints        = n_joints,
        n_joints_groups = [n_joints_axial] + [n_joints_limb] * n_limbs,
        names_groups    = ['axial'] + [f'limb {i}' for i in range(n_limbs)],
    )
    figures_dict = figures_dict | fig_joints_dict

    # Plot links displacements
    steps_sampled = links_displacements_data['links_displacements'].shape[0]
    times_sampled = np.linspace(times[0], times[-1], steps_sampled)
    fig_disp_dict = signal_processor_mech_plot.plot_links_displacements(
        times               = times_sampled,
        links_displacements = links_displacements_data['links_displacements'],
        n_links             = n_joints_axial,
        n_links_groups      = [n_joints_axial],
        names_groups        = ['axial'],
    )
    figures_dict = figures_dict | fig_disp_dict

    # Animate joint coordinates
    vect_com             = np.mean(links_displacements_data['links_positions'], axis = 1)
    vect_com_transformed = np.mean(links_displacements_data['links_positions_transformed'], axis = 1)
    direction_fwd        = links_displacements_data['direction_fwd']
    direction_left       = links_displacements_data['direction_left']

    a,b,c = links_displacements_data['quadratic_fit_coefficients']
    x_vals = vect_com_transformed[:, 0]
    y_vals = a * x_vals ** 2 + b * x_vals + c

    vect_fwd = np.diff( np.array( [x_vals, y_vals] ).T, axis = 0 )
    vect_lat = np.cross(np.array([0, 0, 1]), vect_fwd)[:, :2]

    rot_mat = np.array( [direction_fwd, direction_left] ).T
    vect_fwd = np.array( [[0,0]] + [ rot_mat @ vect_fwd_step for vect_fwd_step in vect_fwd ] ) * 100
    vect_lat = np.array( [[0,0]] + [ rot_mat @ vect_lat_step for vect_lat_step in vect_lat ] ) * 100


    fig_joint_a, animation_joint = signal_processor_mech_plot.animate_joint_coordinates_and_body_axis(
        joint_coordinates = links_displacements_data['links_positions'],
        vect_com          = vect_com,
        vect_fwd          = vect_fwd,
        vect_lat          = vect_lat,
    )
    figures_dict['joints_coordinates_animation'] = [fig_joint_a, animation_joint]

    plots_utils.save_prompt(
        figures_dict = figures_dict,
        folder_path  = filename_data_output,
    )

    plt.show()

def main():
    ''' Main function '''

    # 'swimming', 'walking'
    for gait in ['swimming', 'walking']:
        analyze_gait_data(
            gait         = gait,
            plot_results = True,
        )

    return

if __name__ == '__main__':
    main()