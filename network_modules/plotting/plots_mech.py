"""Plot results of the mechanical simulation """

from typing import Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from farms_amphibious.data.data import SpikingData, AmphibiousKinematicsData
from farms_core.io.yaml import yaml2pyobject

### -------- [ PLOTS ] --------
def _plot_mech_signals(
    times      : np.ndarray,
    signals    : np.ndarray,
    ind_min    : int,
    ind_max    : int,
    chain_name : str,
    signal_name: str,
) -> Figure:
    ''' Evolution of the mechanical signals '''

    # Get max and min values
    n_joints  = ind_max - ind_min
    max_val   = np.amax( signals[1:-1, ind_min : ind_max] )
    min_val   = np.amin( signals[1:-1, ind_min : ind_max] )
    range_val = max_val - min_val

    fig, axs = plt.subplots(
        n_joints,
        1,
        sharex  = True,
        figsize = (8.0, 10.5),
        dpi     = 300
    )
    fig.canvas.manager.set_window_title(f'fig_{signal_name}_{chain_name}')
    fig.subplots_adjust(hspace=0)
    axs[ 0].set_xlim(times[0], times[-1])
    axs[-1].set_xlabel('Time [s]')
    axs[ 0].set_title(f'{signal_name.capitalize()} - {chain_name.upper()}')
    colors = matplotlib.cm.winter(np.linspace(0.25, 0.75, n_joints))

    for joint_ind in range(n_joints):
        axs[joint_ind].plot(
            times,
            signals[:, joint_ind + ind_min],
            color     = colors[joint_ind],
            linewidth = 2,
        )

        axs[joint_ind].set_ylim(
            ( min_val - 0.1 * range_val ),
            ( max_val + 0.1 * range_val ),
        )
        axs[joint_ind].set_ylabel(f'{signal_name.capitalize()} {joint_ind+1}')
        axs[joint_ind].grid(axis='both', linestyle='--')

    return fig

def plot_joints_angles(
    times        : np.ndarray,
    joints_angles: np.ndarray,
    params       : dict
) -> dict[str, Figure]:
    ''' Evolution of the joint angles '''

    n_joints_ax = params['morphology']['n_joints_body']
    n_dofs_lb   = params['morphology']['n_dof_legs']
    n_legs      = params['morphology']['n_legs']

    figures_joint_angles : dict[str, Figure] = {}

    # AXIS
    figures_joint_angles['fig_ja_axis'] = _plot_mech_signals(
        times       = times,
        signals     = joints_angles,
        ind_min     = 0,
        ind_max     = n_joints_ax,
        chain_name  = 'axis',
        signal_name = 'joint angles',
    )

    # LIMBS
    for limb in range(n_legs):
        ind_min = n_joints_ax + limb * n_dofs_lb
        ind_max = n_joints_ax + (limb + 1) * n_dofs_lb

        figures_joint_angles[f'fig_ja_limb_{limb}'] = _plot_mech_signals(
            times       = times,
            signals     = joints_angles,
            ind_min     = ind_min,
            ind_max     = ind_max,
            chain_name  = f'limb_{limb}',
            signal_name = 'Joint angles',
        )

    return figures_joint_angles

def plot_com_trajectory(
    times        : np.ndarray,
    com_positions: np.ndarray,
) -> dict[str, Figure]:
    ''' Plot 2D trajectory of the link '''

    timestep = times[1] - times[0]
    xvals    = np.array(com_positions[:, 0])
    yvals    = np.array(com_positions[:, 1])

    speeds      = np.linalg.norm( np.diff(com_positions[:, :2], axis=0), axis=1 ) / timestep
    n_speeds    = len(speeds)
    max_speed   = np.amax(speeds)
    speeds_inds = np.asarray( np.round( (n_speeds-1) * speeds / max_speed), dtype= int )
    colors      = plt.cm.jet(np.linspace(0,1,n_speeds))

    figures_trajectory : dict[str, Figure] = {}

    # XY plot
    fig_traj_2d = plt.figure('COM trajectory 2D')
    ax1 = fig_traj_2d.add_subplot(111)

    ax1.plot(xvals[0], yvals[0], 'gx')

    for ind, speed_ind in enumerate(speeds_inds):
        ax1.plot(
            xvals[ind: ind+2],
            yvals[ind: ind+2],
            lw    = 1.0,
            color = colors[speed_ind]
        )

    # Draw colorbar
    vmin   = np.amin(speeds)
    vmax   = np.amax(speeds)
    vrange = vmax - vmin
    cmap   = plt.get_cmap('jet', n_speeds)
    norm   = matplotlib.colors.Normalize(vmin= vmin, vmax= vmax)
    sm     = plt.cm.ScalarMappable(cmap=cmap, norm= norm)
    sm.set_array([])
    plt.colorbar(
        sm,
        ax    = ax1,
        label = 'Speed [m/s]',
        boundaries = np.linspace(
            vmin - 0.05 * vrange,
            vmax + 0.05 * vrange,
            20
        )
    )

    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.set_title('Trajectory')

    figures_trajectory['fig_traj_2d'] = fig_traj_2d

    # Coordinate evolution
    fig_traj_1d = plt.figure('COM trajectory 1D')
    ax2 = fig_traj_1d.add_subplot(111)

    ax2.plot(times, xvals, label= 'X')
    ax2.plot(times, yvals, label= 'Y')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Coordinate [m]')
    ax2.legend()

    figures_trajectory['fig_traj_1d'] = fig_traj_1d

    # Speed evolution
    fig_speed_1d = plt.figure('COM speed 1D')
    ax3 = fig_speed_1d.add_subplot(111)

    ax3.plot(times[:-1], np.diff(xvals)/timestep, label= 'dX/dt')
    ax3.plot(times[:-1], np.diff(yvals)/timestep, label= 'dY/dt')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Speed [m/s]')
    ax3.legend()

    figures_trajectory['fig_speed_1d'] = fig_speed_1d

    return figures_trajectory

### \-------- [ PLOTS ] --------

### -------- [ ANIMATIONS ] --------
def animate_links_trajectory(
    fig            : Figure,
    times          : np.ndarray,
    links_positions: np.ndarray,
    params         : dict
) -> FuncAnimation:
    ''' Animation of the trajectory '''

    joints_ax = params['morphology']['n_joints_body']

    # Define time steps to be plotted
    steps_max = len(times)
    steps_jmp = 10
    steps     = np.arange( 0, steps_max, steps_jmp, dtype= int )

    # Initialize plot
    ax1 = plt.axes()
    ax1.plot( links_positions[0, :joints_ax, 0],
              links_positions[0, :joints_ax, 1], 'k' )
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.set_title('Trajectory')

    time_text = ax1.text(0.02, 0.05, '', transform=ax1.transAxes)

    # Function to call
    def _plot_body_configuration(links_data: np.ndarray, step: int) -> None:
        ''' Plot 2D trajectory of the link '''
        for artist in ax1.lines + ax1.collections:
            artist.remove()
        ax1.plot(links_data[step, :joints_ax, 0],
                 links_data[step, :joints_ax, 1], 'k')
        ax1.plot(links_data[step, 0, 0],
                 links_data[step, 0, 1], 'ko', markersize= 10)
        ax1.plot(links_data[step, joints_ax-1, 0],
                 links_data[step, joints_ax-1, 1], 'k<')

        xrange = np.array( [np.amin(links_data[step, :joints_ax, 0]),
                            np.amax(links_data[step, :joints_ax, 0])] )
        yrange = np.array( [np.amin(links_data[step, :joints_ax, 1]),
                            np.amax(links_data[step, :joints_ax, 1])] )

        xlims = [ xrange[0] - 0.1*(xrange[1]-xrange[0]) - 0.1,
                xrange[1] + 0.1*(xrange[1]-xrange[0]) + 0.1 ]

        ylims = [ yrange[0] - 0.1*(yrange[1]-yrange[0]) - 0.1,
                yrange[1] + 0.1*(yrange[1]-yrange[0]) + 0.1 ]

        ax1.set_xlim( xlims )
        ax1.set_ylim( ylims )
        time_text.set_text(f"time: {100 * step / steps_max :.1f} %")

    # Define animation
    def _animation_step(anim_step: int) -> Tuple[plt.Axes, str]:
        ''' Update animation '''
        _plot_body_configuration(links_positions, anim_step)
        return (ax1, time_text)

    anim = FuncAnimation(
        fig,
        _animation_step,
        frames   = steps,
        interval = steps_jmp,
        blit     = False
    )

    # plt.show(block = False)
    return anim

### \-------- [ ANIMATIONS ] --------
