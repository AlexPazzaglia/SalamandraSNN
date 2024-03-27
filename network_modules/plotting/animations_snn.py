'''
Module to store the functions used to plot neuronal and synaptic quantities from the simulations.
'''
import numpy as np
import brian2 as b2

import matplotlib.pyplot as plt
from matplotlib.figure      import Figure
from matplotlib.animation   import FuncAnimation
from matplotlib.collections import PatchCollection

import network_modules.plotting.plots_snn as plotting
from   network_modules.equations import utils
from network_modules.parameters.network_module import SnnNetworkModule

### -------- [ ANIMATIONS ] --------

def animation_raster_plot(
    fig                 : Figure,
    pop                 : b2.NeuronGroup,
    spikemon_t          : np.ndarray,
    spikemon_i          : np.ndarray,
    duration            : float,
    timestep            : float,
    network_modules_list: list[SnnNetworkModule],
    plotpars            : dict
) -> None:
    '''
    Raster plot of the recorded neural activity. Limbs are inserted in the plot according to
    their position in the axial network.
    '''

    # PARAMETERS
    n_tot          = len(pop)
    t_start        = float( spikemon_t[0] / b2.msecond )
    duration       = float( duration / b2.msecond )
    timestep       = float( timestep / b2.msecond )
    sampling_ratio = plotpars.get('sampling_ratio', 1.0)

    # EXCITATORY-ONLY CONDITION
    ex_only = plotpars.get('ex_only', False)
    if ex_only:
        network_modules_list = [
            net_mod
            for net_mod in network_modules_list
            if 'in' not in net_mod.name.split('.')
        ]

    # ONE-SIDED CONDITION
    if plotpars.get('one_sided', False):
        target_indices = np.sort(
            np.concatenate(
                [ net_mod.indices_sides[0] for net_mod in network_modules_list ],
            )
        )
    else:
        target_indices = np.sort(
            np.concatenate(
                [ net_mod.indices for net_mod in network_modules_list ],
            )
        )

    # Pools properties
    modules_labels = [ mod.name                   for mod in network_modules_list ]
    modules_colors = [ mod.plotting.get('color')  for mod in network_modules_list ]

    # MAP SPIKES FIRED BY THE SELECTED INDICES
    spikes_t = (spikemon_t - spikemon_t[0]) * 1000
    spikes_i = spikemon_i

    pruned_inds = np.ones(n_tot)
    pruned_inds[target_indices] = 0

    inds_mask = np.array(
        [ np.sum(pruned_inds[:i]) for i in range(n_tot) ],
        dtype= int
    )

    spikes_t_pruned = spikes_t[ np.isin(spikes_i, target_indices) ]
    spikes_i_mapped = spikes_i - inds_mask[spikes_i]
    spikes_i_pruned = spikes_i_mapped[ np.isin(spikes_i, target_indices) ]

    # SAMPLE SPIKES
    n_spikes = len(spikes_i_pruned)
    sampled_spikes_inds = np.random.rand(n_spikes) <= sampling_ratio

    spikes_t_sampled = spikes_t_pruned[sampled_spikes_inds]
    spikes_i_sampled = spikes_i_pruned[sampled_spikes_inds]

    # MAP MODULE LIMITS
    modules_limits_mapped = np.array(
        [
            np.array(mod.indices_limits, dtype= int) - inds_mask[mod.indices_limits]
            for mod in network_modules_list
        ]
    )

    modules_copies_limits_mapped = [
        np.array(
            [
                (
                    np.array( [ copy_ind[0], copy_ind[-1] ], dtype= int)
                    - inds_mask[ [ copy_ind[0], copy_ind[-1] ] ]
                )
                for copy_ind in mod.indices_copies
            ]
        )
        for mod in network_modules_list
    ]

    # PLOT SETUP
    axis = plt.axes(
        xlim = (t_start, t_start + duration),
        ylim = (0, modules_limits_mapped[-1, -1])
    )

    # Separate modules
    plt.hlines(
        y          = modules_limits_mapped[:, 0],
        xmin       = t_start,
        xmax       = t_start + duration,
        linestyles = '-',
        linewidth  = 0.4,
        color      = '0.5'
    )

    for mod_copies_limits in modules_copies_limits_mapped:
        # Separate copies
        plt.hlines(
            y          = mod_copies_limits[1:, 0],
            xmin       = t_start,
            xmax       = t_start + duration,
            linestyles = '--',
            linewidth  = 0.4,
            color      = '0.5'
        )

    # DECORATE
    plt.yticks(np.mean(modules_limits_mapped, axis=1), modules_labels)

    plt.xlabel('Time [ms]')
    plt.ylabel('Neuronal pools')
    plt.title('Neural activation')
    plt.xlim(0,duration)

    # Invert y axis representation
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # DEFINE SCATTER PLOTS
    scatter_plots : list[PatchCollection] = []
    for colorval in modules_colors:
        scatter_plots.append(
            plt.scatter(
                [],
                [],
                s = 0.002,
                c = colorval,
            )
        )

    # DEFINE ANIMATION
    steps_max = round(duration / timestep)
    steps_jmp = round(50 / timestep)
    steps     = np.arange(0, steps_max, steps_jmp, dtype= int )

    time_text = axis.text(0.80, 0.95, '', transform=axis.transAxes)
    patches   = [ time_text ] + scatter_plots

    def _animation_step(anim_step):
        ''' Animation step '''

        spike_inds_interval = spikes_t_sampled <  (anim_step + steps_jmp) * timestep

        spikes_i_interval = spikes_i_sampled[spike_inds_interval]
        spikes_t_interval = spikes_t_sampled[spike_inds_interval]

        # Raster plot
        for mod_ind, mod_limits in enumerate(modules_limits_mapped):
            pool_inds  = range( mod_limits[0], mod_limits[1] )
            spike_inds = np.isin(spikes_i_interval, pool_inds)

            scatter_plots[mod_ind].set_offsets(
                np.array(
                    [
                        spikes_t_interval[spike_inds],
                        spikes_i_interval[spike_inds],
                    ]
                ).T
            )

        time_text.set_text(f"time: {100 * anim_step / steps_max :.1f} %" )

        return patches

    anim = FuncAnimation(
        fig       = fig,
        func      = _animation_step,
        frames    = steps,
        interval  = steps_jmp,
        blit      = True
    )

    return anim


def animation_neuronal_activity(
        fig           : Figure,
        pop           : b2.NeuronGroup,
        statemon_times: np.ndarray,
        statemon_dict : dict[str, np.ndarray],
        net_cpg_module: SnnNetworkModule,
        height        : float,
        width         : float = 0.001,
        limb_positions: list[int] = None
    ) -> FuncAnimation:
    '''
    Map showing the neurons in the network and the evolution of their membrane potential
    '''

    if not limb_positions:
        limb_positions = []

    segments_axial = net_cpg_module['axial'].pools
    segments_limbs = net_cpg_module['limbs'].pools
    w_mm           = width/b2.mmetre
    h_mm           = height/b2.mmetre

    ex_ind_l, ex_ind_r = np.concatenate(
        [
            net_cpg_module['axial']['ex'].indices_pools_sides,
            net_cpg_module['limbs']['ex'].indices_pools_sides,
        ],
        axis= 1,
    )
    in_ind_l, in_ind_r = np.concatenate(
        [
            net_cpg_module['axial']['in'].indices_pools_sides,
            net_cpg_module['limbs']['in'].indices_pools_sides,
        ],
        axis= 1,
    )

    # Define plots to be updated
    ax1 = plt.axes(xlim=(-2*w_mm,2*w_mm), ylim=(0,segments_axial*h_mm))

    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

    v_memb = statemon_dict['v'].T
    vmin = v_memb.min()
    vmax = v_memb.max()
    v_memb = (v_memb-vmin)/(vmax-vmin)

    pools_params = [
        ['ex_l',    ex_ind_l,        0,                        segments_axial, +1.0*w_mm, 'o'],
        ['ex_r',    ex_ind_r,        0,                        segments_axial, -1.0*w_mm, 'o'],
        ['in_l',    in_ind_l,        0,                        segments_axial, +0.5*w_mm, ','],
        ['in_r',    in_ind_r,        0,                        segments_axial, -0.5*w_mm, ','],
        ['ex_l_lb', ex_ind_l, segments_axial, segments_axial + segments_limbs, +2.0*w_mm, 'o'],
        ['ex_r_lb', ex_ind_r, segments_axial, segments_axial + segments_limbs, -2.0*w_mm, 'o'],
        ['in_l_lb', in_ind_l, segments_axial, segments_axial + segments_limbs, +1.5*w_mm, ','],
        ['in_r_lb', in_ind_r, segments_axial, segments_axial + segments_limbs, -1.5*w_mm, ','],
    ]

    # Define scatter plots
    scatter_plots : dict[str, list[PatchCollection]]= {}
    for key, indices, start_ind, end_ind, position, marker in pools_params:
        scatter_plots[key] = [
            plt.scatter(
                position * np.ones( (len(indices[seg_ind]),1) ),
                pop[indices[seg_ind]].y_neur / b2.mmetre,
                c = v_memb[indices[seg_ind],0],
                s=100,
                marker = marker
            )
            for seg_ind in range(start_ind, end_ind)
        ]

    # Things to animate
    patches = [ time_text ] + [
        scat_seg
        for scat_list in scatter_plots.values()
        for scat_seg in scat_list
    ]

    # Decorate plot
    # Axial boxes
    plt.plot([0,0], [0,segments_axial*h_mm], 'k', linewidth= 5)
    for i in range(segments_axial):
        plt.plot([-w_mm,w_mm], [i*h_mm,i*h_mm], 'k', linewidth= 1)

    plt.plot([-w_mm,-w_mm], [0,segments_axial*h_mm], 'k', linewidth= 1)
    plt.plot([+w_mm,+w_mm], [0,segments_axial*h_mm], 'k', linewidth= 1)

    # Limb boxes
    base_box = np.array(
        [
            [+1*w_mm, 0*h_mm],
            [+2*w_mm, 0*h_mm],
            [+2*w_mm, 1*h_mm],
            [+1*w_mm, 1*h_mm],
        ]
    )

    for pos in limb_positions:
        pos_y = pos*h_mm
        plt.plot(+base_box[:,0], base_box[:,1] + pos_y, 'r', lw=1)
        plt.plot(-base_box[:,0], base_box[:,1] + pos_y, 'r', lw=1)

    plt.xlabel('X position [mm]')
    plt.ylabel('Y position [mm]')
    plt.title('Evolution of neuronal activity')

    # Define time steps to be plotted
    steps_max = len(statemon_times)
    steps_jmp = 10
    steps     = np.arange(0, steps_max, steps_jmp, dtype= int )

    # Define animation
    def _animation_step(anim_step):
        ''' Animation step '''

        for key, indices, start_ind, end_ind, _, _ in pools_params:
            for seg_ind in range(start_ind, end_ind):
                scatter_plots[key][seg_ind - start_ind].set_array(v_memb[indices[seg_ind], anim_step])

        time_text.set_text(f"time: {100 * anim_step / steps_max :.1f} %" )
        return patches

    anim = FuncAnimation(
        fig       = fig,
        func      = _animation_step,
        frames    = steps,
        interval  = steps_jmp,
        blit      = True)

    return anim

def animation_smooth_neural_activity(
        fig           : Figure,
        signals       : np.ndarray,
        limb_positions: list[int] = None
    ) -> FuncAnimation:
    '''
    Map showing the pools in the CPG network and the evolution of their membrane potential
    '''

    if limb_positions is None:
        limb_positions = []

    limb_segments = len(limb_positions)
    segments = signals.shape[0]//2 - limb_segments

    # Define plots to be updated
    ax1 = plt.axes(xlim=(-2.5,2.5), ylim=(-1,segments+1))

    # Initialize plot
    time_text = ax1.text(0.02, 0.05, '', transform=ax1.transAxes)

    ax_l_scat = plt.scatter(
        +0.5*np.ones(segments),
        -np.arange(segments)+segments-1,
        c = signals[ 2*np.arange(segments, dtype= int), 0],
        s=500,
        marker = "o"
    )

    ax_r_scat = plt.scatter(
        -0.5*np.ones(segments),
        -np.arange(segments)+segments-1,
        c = signals[ 2*np.arange(segments, dtype= int)+1, 0],
        s=500,
        marker = "o"
    )

    limbs_scat : list[PatchCollection] = []

    for lbind, lbpos in enumerate(limb_positions):
        side = -1 if lbind % 2 else 1
        indices = np.array( [2*(segments+lbind), 2*(segments+lbind)+1] )

        limbs_scat.append(
            plt.scatter(
                np.array( [1.25*side, 1.75*side] ),
                -lbpos * np.ones(2)+segments-1 + 0.5,
                c = signals[ indices, 0],
                s=500,
                marker = "o"
            )
        )

    patches = [ax_l_scat] + [ax_r_scat] + limbs_scat + [ time_text ] #things to animate

    # Decorate plot
    # Axial boxes
    plt.plot([0,0], [0,segments], 'k', linewidth= 5)
    for i in range(segments+1):
        plt.plot([-1,1], [i,i], 'k', linewidth= 1)

    plt.plot([-1,-1], [0,segments], 'k', linewidth= 1)
    plt.plot([+1,+1], [0,segments], 'k', linewidth= 1)

    # Limb boxes
    for lbpos in limb_positions:
        pos = segments-1 - lbpos
        plt.plot([ -1,  -2,    -2,    -1],
                 [pos, pos, pos+1, pos+1],
                 'r',
                 linewidth= 1)

        plt.plot([ +1,  +2,    +2,    +1],
                 [pos, pos, pos+1, pos+1],
                 'r',
                 linewidth= 1)

    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Evolution of smooth neuronal activity')

    # Define time steps to be plotted
    steps_max = signals.shape[1]
    steps_jmp = 10
    steps     = np.arange( 0, steps_max, steps_jmp, dtype= int )

    # Define animation
    def _animation_init() -> list:
        ''' Initialize animation '''
        ax_l_scat.set_array(signals[ 2*np.arange(segments, dtype= int), 0])
        ax_r_scat.set_array(signals[ 2*np.arange(segments, dtype= int)+1, 0])

        for lbind, lbscat in enumerate(limbs_scat):
            aux = np.array( [2*(lbind+ segments), 2*(lbind+ segments)+1] )
            lbscat.set_array( signals[aux, 0])

        time_text.set_text('')

        return patches #return everything that must be updated

    def _animation_step(anim_step: int) -> list:
        ''' Update animation '''

        ax_l_scat.set_array(signals[ 2*np.arange(segments, dtype= int),   anim_step])
        ax_r_scat.set_array(signals[ 2*np.arange(segments, dtype= int)+1, anim_step])

        for lbind, lbscat in enumerate(limbs_scat):
            aux = np.array( [2*(lbind+ segments), 2*(lbind+ segments)+1] )
            lbscat.set_array( signals[aux, anim_step])

        time_text.set_text(f"time: {100 * anim_step / steps_max :.1f} %" )

        return patches #return everything that must be updated

    anim = FuncAnimation(
        fig       = fig,
        func      = _animation_step,
        init_func = _animation_init,
        frames    = steps,
        interval  = steps_jmp,
        blit      = True
    )

    return anim

# \-------- [ ANIMATIONS ] --------
