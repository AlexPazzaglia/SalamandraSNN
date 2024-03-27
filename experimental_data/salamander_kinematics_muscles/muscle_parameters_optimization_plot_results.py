''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import dill
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import matplotlib.pyplot as plt

N_JOINTS_AXIS = 8

# FIGURE PARAMETERS
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.rc(  'font',      size = SMALL_SIZE )  # controls default text sizes
plt.rc(  'axes', titlesize = SMALL_SIZE )  # fontsize of the axes title
plt.rc(  'axes', labelsize = MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc( 'xtick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc( 'ytick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend',  fontsize = SMALL_SIZE )  # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure',   figsize = (10.0, 5.0)) # size of the figure
plt.rc('lines',  linewidth = 2.0         ) # linewidth of the figure

def decorate_axes(axes_list : list[plt.Axes]):
    ''' Decorate the axes '''


    for ax in axes_list:

        # Increase label size
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)

        # Grid
        ax.xaxis.grid(False, which='major')
        ax.yaxis.grid(False, which='major')

        # Increase tick size
        ax.tick_params(axis='both', which='major', labelsize=15)

        # Increase thickness of x and y axis
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)

        # Remove right and top box lines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    handles, labels = axes_list[0].get_legend_handles_labels()
    plt.figlegend(
        handles,
        labels,
        loc            = 'upper center',
        prop           = {'size': 10},
        ncol           = 8,
        bbox_to_anchor = (0.5, 0.95)
    )

    # plt.tight_layout()

    return

def main(
    factor_alpha: np.ndarray = np.ones(N_JOINTS_AXIS),
    factor_beta : np.ndarray = np.ones(N_JOINTS_AXIS),
    factor_delta: np.ndarray = np.ones(N_JOINTS_AXIS),
):
    ''' Run the spinal cord model together with the mechanical simulator '''


    # Scalings from optimization results
    folder_name = f'experimental_data/salamander_kinematics_muscles/results/{OPTIMIZATION_NAME}'
    file_names  = [
        f'{folder_name}/performance_iteration_{iteration}.dill'
        for iteration in range(STARTING_ITERATION, STARTING_ITERATION + N_ITERATIONS)
    ]

    iterations_performance = []
    for file_name in file_names:
        with open(file_name, 'rb') as file:
            performance = dill.load(file)

        iterations_performance.append(performance)

    # Extract the parameters from the optimization results

    WN_hat = [ performance['WN_hat'] for performance in iterations_performance ]
    WN_trg = [ performance['WN_target'] for performance in iterations_performance ]

    ZC_hat = [ performance['ZC_hat'] for performance in iterations_performance ]
    ZC_trg = [ performance['ZC_target'] for performance in iterations_performance ]

    gains_alpha = [ performance['gains_scalings_alpha'] for performance in iterations_performance ]
    gains_beta  = [ performance['gains_scalings_beta']  for performance in iterations_performance ]
    gains_delta = [ performance['gains_scalings_delta'] for performance in iterations_performance ]

    # Add first iteration
    gains_alpha = [ np.ones_like(gains_alpha[0]) ] + gains_alpha
    gains_beta  = [ np.ones_like(gains_beta[0])  ] + gains_beta
    gains_delta = [ np.ones_like(gains_delta[0]) ] + gains_delta

    # Scale the gains
    gains_alpha = np.array(gains_alpha) * factor_alpha
    gains_beta  = np.array(gains_beta)  * factor_beta
    gains_delta = np.array(gains_delta) * factor_delta

    # PLOTTING
    joint_colors = plt.cm.plasma(np.linspace(0, 0.8, N_JOINTS_AXIS))

    # Plot WN and ZC
    fig, axes = plt.subplots(2, 1, figsize=(20, 20), sharex=True)

    for joint in range(N_JOINTS_AXIS):
        axes[0].plot(
            [ ( WN_hat[it][joint] - WN_trg[it][joint] ) / (2*np.pi) for it in range(N_ITERATIONS) ],
            # linestyle= '--',
            color= joint_colors[joint],
            label= f'Joint {joint}'
        )
        axes[1].plot(
            [ ZC_hat[it][joint] - ZC_trg[it][joint] for it in range(N_ITERATIONS) ],
            # linestyle= '--',
            color= joint_colors[joint],
            label= f'Joint {joint}'
        )

    axes[0].set_xlim([STARTING_ITERATION, STARTING_ITERATION + N_ITERATIONS])
    axes[0].set_ylabel('FN Error')

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('ZC Error')

    decorate_axes(axes)

    # Plot the gains
    fig, axes = plt.subplots(2, 1, figsize=(20, 20), sharex=True)

    for joint in range(N_JOINTS_AXIS):
        # axes[0].plot(
        #     [ gains_alpha[it][joint] for it in range(N_ITERATIONS + 1) ],
        #     linestyle = '--',
        #     color     = joint_colors[joint],
        #     label     = f'Joint {joint}',
        #     linewidth = 2,
        # )
        axes[0].plot(
            [ gains_beta[it][joint] for it in range(N_ITERATIONS + 1) ],
            # linestyle = '--',
            color     = joint_colors[joint],
            label     = f'Joint {joint}',
            linewidth = 2,
        )
        axes[1].plot(
            [ gains_delta[it][joint] for it in range(N_ITERATIONS + 1) ],
            # linestyle = '--',
            color     = joint_colors[joint],
            label     = f'Joint {joint}',
            linewidth = 2,
        )

    axes[0].set_xlim([STARTING_ITERATION, STARTING_ITERATION + N_ITERATIONS])
    # axes[0].set_yscale('log')
    # axes[0].set_ylabel('Alpha')

    axes[0].set_yscale('log')
    axes[0].set_ylabel('Beta')

    axes[1].set_yscale('log')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Delta')

    decorate_axes(axes)

    plt.show()
    return

if __name__ == '__main__':

    # TOPOLOGY
    N_JOINTS_TRUNK = 4
    N_JOINTS_TAIL  = 4
    N_JOINTS_LIMB  = 4
    N_JOINTS_AXIS  = N_JOINTS_TAIL + N_JOINTS_TRUNK

    # OPTIMIZATION
    # OPTIMIZATION_NAME = 'finding_parameters_FN_3000_ZC_800_G0_785'
    # OPTIMIZATION_NAME = 'finding_parameters_FN_3500_ZC_800_G0_785'
    # OPTIMIZATION_NAME = 'finding_parameters_FN_4000_ZC_800_G0_785'
    # OPTIMIZATION_NAME = 'finding_parameters_FN_4500_ZC_800_G0_785'
    # OPTIMIZATION_NAME = 'muscle_parameters_optimization_FN_1500_ZC_1000_G0_785'
    # OPTIMIZATION_NAME = 'muscle_parameters_optimization_FN_8000_ZC_1000_G0_785'
    OPTIMIZATION_NAME = 'muscle_parameters_optimization_FN_15000_ZC_1000_G0_785'

    N_ITERATIONS      = 50

    STARTING_ITERATION = 0

    # PARAMETERS
    ORIGINAL_ALPHA  = 3.14e-4
    ORIGINAL_BETA   = 2.00e-4
    ORIGINAL_DELTA  = 10.0e-6

    INERTIAL_FACTOR_ALPHA = np.array(
        [
            1.06579348e-01,
            3.91827860e-01,
            1.00000000e+00,
            4.94362095e-01,
            2.10746855e-01,
            6.96893434e-02,
            1.39979008e-02,
            6.85692443e-04
        ]
    )

    INERTIAL_FACTOR_BETA = np.array(
        [
            1.06579348e-01,
            3.91827860e-01,
            1.00000000e+00,
            4.94362095e-01,
            2.10746855e-01,
            6.96893434e-02,
            1.39979008e-02,
            6.85692443e-04
        ]
    )

    INERTIAL_FACTOR_DELTA = np.array(
        [
            0.32646493,
            0.62596155,
            1.00000000,
            0.70310888,
            0.45907173,
            0.26398739,
            0.11831272,
            0.02618573
        ]
    )

    main(
        factor_alpha = ORIGINAL_ALPHA,
        factor_beta  = ORIGINAL_BETA,
        factor_delta = ORIGINAL_DELTA,
    )