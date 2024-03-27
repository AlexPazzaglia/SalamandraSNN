''' Read kinematics data '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

from network_modules.performance import (
    signal_processor_snn,
    signal_processor_mech_plot,
)

from scipy.interpolate import CubicSpline

FOLDER_PATH = 'experimental_data/salamander_kinematics_karakasiliotis'

def _interpolate_signals(times, signals, times_sampled):
    '''
    Interpolates signals using cubic spline interpolation.

    Parameters:
    times (numpy.ndarray): Array of times corresponding to the input signals.
    signals (numpy.ndarray): Array of input signals to be interpolated.
    times_sampled (numpy.ndarray): Array of times to interpolate the signals at.

    Returns:
    numpy.ndarray: Array of interpolated signals.
    '''

    n_signals       = signals.shape[1]
    signals_sampled = np.zeros((len(times_sampled), n_signals))

    for signal_ind in enumerate(range(n_signals)):

        # Cubic interpolation
        func_interpolate     = CubicSpline(times, signals[:, signal_ind])
        interpolated_values  = func_interpolate(times_sampled)
        signals_sampled[:, signal_ind] = interpolated_values

    return signals_sampled

def generate_gait_data(
    gait        : str,
    frequency   : float,
    duration    : float = 10.0,
    timestep    : float = 0.001,
    save_results: bool = False,
    plot_results: bool = False,
):
    ''' Analyze gait data '''

    data_tag = f'{gait}_{int(frequency*1000)}_mHz_{int(duration*1000)}_ms'

    filename_data_input  = f'{FOLDER_PATH}/{gait}/data_karakasiliotis_processed.csv'
    filename_data_output = f'{FOLDER_PATH}/{gait}/generated_kinematics/data_karakasiliotis_{data_tag}.csv'

    n_limbs        = 4
    n_joints_limb  = 4
    n_joints_axial = 8
    n_joints       = n_joints_axial + n_limbs * n_joints_limb

    figures_dict = {}

    # Load data
    kinematics_data    = pd.read_csv(filename_data_input)
    times              = kinematics_data['time'].to_numpy()
    names_joints_axial = kinematics_data.columns[1:1 + n_joints_axial].to_list()
    names_joints_limbs = kinematics_data.columns[1 + n_joints_axial:].to_list()
    joint_angles       = kinematics_data[names_joints_axial + names_joints_limbs].to_numpy()

    # Interpolate axial joint angles for the sampling times
    times_sampled = np.arange(times[0], times[-1], frequency * timestep)

    joint_angles_sampled = _interpolate_signals(
        times         = times,
        signals       = joint_angles,
        times_sampled = times_sampled,
    )

    # Repeat signal for n_cycles
    n_cycles = int( np.ceil(duration * frequency) ) + 1

    times_rep        = np.arange(0, n_cycles / frequency, timestep)
    joint_angles_rep = np.tile(joint_angles_sampled, (n_cycles, 1))

    n_steps_rep = np.amin([len(times_rep), len(joint_angles_rep)])

    times_rep        = times_rep[:n_steps_rep]
    joint_angles_rep = joint_angles_rep[:n_steps_rep]

    # Filter around the frequency
    cutoff_frequency  = frequency * 5.0
    filter_order      = 4
    cutoff_normalized = 2 * cutoff_frequency * timestep

    b, a = butter(filter_order, cutoff_normalized, btype='low')

    joint_angles_rep = filtfilt(b, a, joint_angles_rep, axis=0)

    # # Compute IPLS
    ipls = signal_processor_snn.compute_ipls_corr(
        times        = times_rep,
        signals      = joint_angles_rep.T,
        freqs        = [frequency]*n_joints_axial,
        inds_couples = [[i, i+1] for i in range(n_joints_axial-1)]
    )

    # # Compute FREQS
    # joint_angles_rep_fft = signal_processor_snn.compute_fft_transform(times_rep, joint_angles_rep.T)
    # freqs, inds = signal_processor_snn.compute_frequency_fft(
    #     times=times_rep,
    #     signals_fft=joint_angles_rep_fft,
    # )

    # If there are no limbs, add data for the limbs (folding)
    if len(names_joints_limbs) == 0:

        joint_angles_rep_lb = np.zeros((joint_angles_rep.shape[0], n_limbs * n_joints_limb))
        joint_angles_rep_lb[:,  0] = -70 * (1 - np.exp(- 2 * frequency * times_rep ))
        joint_angles_rep_lb[:,  4] = -70 * (1 - np.exp(- 2 * frequency * times_rep ))
        joint_angles_rep_lb[:,  8] = -70 * (1 - np.exp(- 2 * frequency * times_rep ))
        joint_angles_rep_lb[:, 12] = -70 * (1 - np.exp(- 2 * frequency * times_rep ))

        joint_angles_rep = np.concatenate((joint_angles_rep, joint_angles_rep_lb), axis=1)

    # SAVING
    if save_results:
        joint_angles_generated = np.concatenate((times_rep[:, np.newaxis], joint_angles_rep), axis=1)

        np.savetxt(
            filename_data_output,
            joint_angles_generated,
            delimiter=',',
        )

     # PLOTTING
    if not plot_results:
        return

    # Plot joint angles
    fig_joints_dict = signal_processor_mech_plot.plot_joints_angles(
        times           = times_rep,
        joints_angles   = joint_angles_rep,
        n_joints        = n_joints,
        n_joints_groups = [n_joints_axial] + [n_joints_limb] * n_limbs,
        names_groups    = ['axial'] + [f'limb {i}' for i in range(n_limbs)],
    )
    figures_dict = figures_dict | fig_joints_dict

    if save_results:
        for fig_name, fig in figures_dict.items():
            fig.savefig(
                f'{FOLDER_PATH}/{gait}/generated_kinematics/{fig_name}_{data_tag}.png',
                dpi=300,
            )

    plt.show()

def main():
    ''' Main function '''

    params_gait_list = [
        # 'swimming': {
        #     'frequency': 3.17,
        #     'duration' : 10.0,
        #     'timestep' : 0.001,
        # },
        # 'walking': {
        #     'frequency': 0.88,
        #     'duration' : 10.0,
        #     'timestep' : 0.001,
        # },
        {
            'gait'     : 'swimming',
            'frequency': freq,
            'duration' : 10.0,
            'timestep' : 0.001,
        }
        for freq in [2.50, 2.75, 3.00, 3.25, 3.5, 3.75, 4.0]
    ]

    # 'swimming', 'walking'
    for params_gait in params_gait_list:
        generate_gait_data(
            save_results = True,
            plot_results = False,
            **params_gait,
        )

    return

if __name__ == '__main__':
    main()