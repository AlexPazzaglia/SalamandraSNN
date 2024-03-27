import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from network_modules.plotting import plots_mech

# PLOTTING
def plot_joints_angles(
    times          : np.ndarray,
    joints_angles  : np.ndarray,
    n_joints       : int,
    n_joints_groups: list[int] = None,
    names_groups   : list[str] = None,
) -> dict[str, Figure]:
    ''' Plot joint angles '''

    if n_joints_groups is None:
        n_joints_groups = [n_joints]
    if names_groups is None:
        names_groups = ['ALL']
    assert len(n_joints_groups) == len(names_groups), \
        'n_joints_groups and names_groups must have the same length'
    assert sum(n_joints_groups) == n_joints, \
        'Sum of n_joints_groups must be equal to n_joints'

    figures_joint_angles = {}

    for group_ind, (n_group, name_group) in enumerate(zip(n_joints_groups, names_groups)):
        if n_group == 0:
            continue

        ind_min = round( np.sum(n_joints_groups[:group_ind]) )
        ind_max = ind_min + n_group

        figures_joint_angles[f'fig_ja_{name_group}'] = plots_mech._plot_mech_signals(
            times       = times,
            signals     = joints_angles,
            ind_min     = ind_min,
            ind_max     = ind_max,
            chain_name  = name_group,
            signal_name = 'joint angles',
        )

    return figures_joint_angles

def plot_links_displacements(
    times              : np.ndarray,
    links_displacements: np.ndarray,
    n_links            : int,
    n_links_groups     : list[int] = None,
    names_groups       : list[str] = None,
) -> dict[str, Figure]:
    ''' Plot links displacements '''

    if n_links_groups is None:
        n_links_groups = [n_links]
    if names_groups is None:
        names_groups = ['ALL']
    assert len(n_links_groups) == len(names_groups), \
        'n_links_groups and names_groups must have the same length'
    assert sum(n_links_groups) == n_links, \
        'Sum of n_links_groups must be equal to n_links'

    figures_joint_disp = {}

    for group_ind, (n_group, name_group) in enumerate(zip(n_links_groups, names_groups)):
        ind_min = round( np.sum(n_links_groups[:group_ind]) )
        ind_max = ind_min + n_group

        figures_joint_disp[f'fig_ld_{name_group}'] = plots_mech._plot_mech_signals(
            times       = times,
            signals     = links_displacements,
            ind_min     = ind_min,
            ind_max     = ind_max,
            chain_name  = name_group,
            signal_name = 'links_displacements',
        )

    return figures_joint_disp

def plot_trajectory_fit(
    links_pos_xy              : np.ndarray,
    direction_fwd             : np.ndarray,
    direction_left            : np.ndarray,
    quadratic_fit_coefficients: np.ndarray,
):
    ''' Plot trajectory and fitting line '''

    n_steps = links_pos_xy.shape[0]
    n_links = links_pos_xy.shape[1]

    fig, ax = plt.subplots(1, 1, figsize = (8.0, 10.5), dpi = 300)
    fig.canvas.manager.set_window_title(f'fig_traj_fit')

    # Fitting parabola
    # Transformed coordinates
    x_min_parabola_tr  = np.dot(links_pos_xy[0, -1], direction_fwd)
    x_max_parabola_tr  = np.dot(links_pos_xy[-1, 0], direction_fwd)
    x_vals_parabola_tr = np.linspace(x_min_parabola_tr, x_max_parabola_tr, 100)
    y_vals_parabola_tr = np.array(
        [
            np.polyval(quadratic_fit_coefficients, x_val)
            for x_val in x_vals_parabola_tr
        ]
    )

    # Global coordinates
    xy_vals_parabola = np.zeros((100, 2))
    for step, (x_tr, y_tr) in enumerate(zip(x_vals_parabola_tr, y_vals_parabola_tr)):
        xy_vals_parabola[step] = x_tr * direction_fwd + y_tr * direction_left

    ax.plot(
        xy_vals_parabola[:5, 0],
        xy_vals_parabola[:5, 1],
        linewidth = 1.0,
        color     = 'red',
        linestyle = 'dotted'
    )
    ax.plot(
        xy_vals_parabola[5:-5, 0],
        xy_vals_parabola[5:-5, 1],
        linewidth = 1.0,
        color     = 'red',
    )
    ax.plot(
        xy_vals_parabola[-5:, 0],
        xy_vals_parabola[-5:, 1],
        linewidth = 1.0,
        color     = 'red',
        linestyle = 'dotted'
    )

    # Body coordinates
    n_jump_steps = 10
    n_jump_body  = 20
    for step, links_pos_xy_step in enumerate(links_pos_xy[::n_jump_steps]):
        ax.scatter(
            links_pos_xy_step[:, 0],
            links_pos_xy_step[:, 1],
            marker = '.',
            color  = 'black',
            s      = 1.0,
            alpha  = 0.5,
        )

        # Body configuration
        if step % n_jump_body != 0:
            continue

        for link in range(n_links):
            ax.plot(
                [ links_pos_xy_step[link, 0] ],
                [ links_pos_xy_step[link, 1] ],
                marker     = 'o',
                markersize = 2.0 - 1.0 * link / len(links_pos_xy_step),
                color      = 'blue',
            )
            if link == 0:
                continue

            ax.plot(
                links_pos_xy_step[link-1 : link+1, 0],
                links_pos_xy_step[link-1 : link+1, 1],
                linestyle  = '-',
                linewidth  = 2.0 - 1.0 * link / len(links_pos_xy_step),
                color      = 'blue',
            )

    x_min = np.amin(xy_vals_parabola[:, 0])
    x_max = np.amax(xy_vals_parabola[:, 0])
    ax.set_xlim((x_min, x_max))
    ax.axis('equal')

    return fig

def plot_joint_coordinates_and_body_axis(
    links_positions: np.ndarray[float],
    vect_com       : np.ndarray[float],
    vect_fwd       : np.ndarray[float],
    vect_lat       : np.ndarray[float],
    ind            : int,
):
    ''' Plot joint coordinates and body axis '''

    fig = plt.figure(f'Joint coordinates and body axis at step {ind}')

    plt.plot(links_positions[ind][:,0], links_positions[ind][:, 1], '-o')
    plt.plot(vect_com[ind][0], vect_com[ind][1], 'ro')

    scaling_fwd = 2
    scaling_lat = 1

    plt.plot(
        [  vect_com[ind, 0]  ],
        [  vect_com[ind, 1]  ],
        'ro'
    )
    plt.plot(
        [  vect_com[ind][0], (vect_com[ind] + scaling_fwd * vect_fwd[ind])[0]  ],
        [  vect_com[ind][1], (vect_com[ind] + scaling_fwd * vect_fwd[ind])[1]  ],
        'r',
        marker = 'x'
    )
    plt.plot(
        [  vect_com[ind][0], (vect_com[ind] + scaling_lat * vect_lat[ind])[0]  ],
        [  vect_com[ind][1], (vect_com[ind] + scaling_lat * vect_lat[ind])[1]  ],
        'r',
        marker = 'x'
    )

    return fig

def animate_joint_coordinated(
    joint_coordinates: np.ndarray[float],
):
    ''' Animate joint coordinates '''

    num_steps  = joint_coordinates.shape[0]
    link_lengths = np.array(
        [
            np.linalg.norm(joint_coordinates[0, ind] - joint_coordinates[0, ind+1])
            for ind in range(joint_coordinates.shape[1] - 1)
        ]
    )

    # Create the figure and axis for the animation
    fig, ax = plt.subplots(figsize=(18.5, 10.5))
    ax.set_xlim([- 1, sum(link_lengths) + 1])
    ax.set_ylim([-sum(link_lengths)//2, sum(link_lengths)//2])

    # Create the line object for the kinematic chain
    line, = ax.plot([], [], '-o')

    # Define the update function for the animation
    def _update_plot(frame, line, joint_coordinates):
        ''' Update the line data with the coordinates for the current frame '''
        line.set_data(joint_coordinates[frame, :, 0], joint_coordinates[frame, :, 1])
        return line,

    update_func = lambda frame: _update_plot(frame, line, joint_coordinates)

    # Create the animation
    animation = FuncAnimation(fig, update_func, frames=num_steps, interval=10, blit=True)

    return fig, animation

def animate_joint_coordinates_and_body_axis(
    joint_coordinates: np.ndarray[float],
    vect_com         : np.ndarray[float],
    vect_fwd         : np.ndarray[float],
    vect_lat         : np.ndarray[float],
):
    ''' Animate joint coordinates '''

    num_steps  = joint_coordinates.shape[0]
    link_lengths = np.array(
        [
            np.linalg.norm(joint_coordinates[0, ind] - joint_coordinates[0, ind+1])
            for ind in range(joint_coordinates.shape[1] - 1)
        ]
    )
    axis_length = np.sum(link_lengths)
    axis_lengt_tol = 0.1 * axis_length

    # Create the figure and axis for the animation
    fig, ax = plt.subplots(figsize=(18.5, 10.5))
    ax.set_xlim( [                - axis_lengt_tol,   axis_length +  axis_lengt_tol ] )
    ax.set_ylim( [ -axis_length/2 - axis_lengt_tol, axis_length/2 +  axis_lengt_tol ] )

    # Create the line object for the kinematic chain
    ax.axis('equal')
    line_joints, = ax.plot([], [], '-o')
    line_com  ,  = ax.plot([], [], 'ro')
    line_fwd,    = ax.plot([], [], 'r')
    line_lat,    = ax.plot([], [], 'r')

    # Define the update function for the animation
    scaling_fwd = 2
    scaling_lat = 1

    def _update_plot(ind):
        ''' Update the line data with the coordinates for the current frame '''
        line_joints.set_data(joint_coordinates[ind, :, 0], joint_coordinates[ind, :, 1])

        line_com.set_data(
            [  vect_com[ind, 0]  ],
            [  vect_com[ind, 1]  ],
        )
        line_fwd.set_data(
            [  vect_com[ind][0], (vect_com[ind] + scaling_fwd * vect_fwd[ind])[0]  ],
            [  vect_com[ind][1], (vect_com[ind] + scaling_fwd * vect_fwd[ind])[1]  ],
        )
        line_lat.set_data(
            [  vect_com[ind][0], (vect_com[ind] + scaling_lat * vect_lat[ind])[0]  ],
            [  vect_com[ind][1], (vect_com[ind] + scaling_lat * vect_lat[ind])[1]  ],
        )

        return line_joints, line_com, line_fwd, line_lat

    update_func = lambda frame: _update_plot(frame)

    # Create the animation
    animation = FuncAnimation(fig, update_func, frames=num_steps, interval=10, blit=True)

    return fig, animation
