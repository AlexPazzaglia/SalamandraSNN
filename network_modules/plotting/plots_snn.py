'''
Module to store the functions used to plot neuronal and synaptic quantities from the simulations.
'''
import logging
import random
import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.signal import butter, filtfilt
from network_modules.parameters.network_module import SnnNetworkModule


# -------- [ AUXILIARY FUNCTIONS ] --------
def get_modules_labels(
        network_modules_list: list[SnnNetworkModule]
    ) -> tuple[list[float], list[str]]:
    ''' Creates labels for the different modules of the network '''

    modules_labs   = [module.name for module in network_modules_list]
    modules_limits = [module.indices_limits for module in network_modules_list]

    locs = []
    labs = []
    for label, mod_limits in zip(modules_labs, modules_limits):
        locs.append( np.mean(mod_limits) )
        labs.append( label )

    return locs, labs

def plot_modules_grid(
        network_modules_list: list[SnnNetworkModule],
        vlimits: list[int] = None,
        hlimits: list[int] = None,
        **kwargs
    ) -> None:
    ''' Create grid to separate the modules of the network '''

    lw_mod = kwargs.pop('lw_mod', 0.4)
    lw_cop = kwargs.pop('lw_cop', 0.4)
    lw_sid = kwargs.pop('lw_sid', 0.2)

    # Modules properties
    modules_limits = [module.indices_limits[0] - 0.5 for module in network_modules_list]
    modules_copies_limits = [
        [
            module_copy_inds[0] - 0.5
            for module_copy_inds in module.indices_copies
        ]
        for module in network_modules_list
    ]
    modules_ls = [
        (module.plotting['linestyle'] if module.plotting else '-')
        for module in network_modules_list
    ]

    modules_sides_limits = [
        [
            module_copy_inds[0] + module.n_copy_side * side - 0.5
            for module_copy_inds in module.indices_copies
            for side in range(1, module.sides)
        ]
        for module in network_modules_list
    ]

    # Auxiliary function
    def plot_lines(orientation: int, limits: list[int]):
        if limits is None:
            return

        if orientation == 0:
            plotter_function = plt.hlines
        if orientation == 1:
            plotter_function = plt.vlines

        # Plot
        if limits is not None:
            # Between modules
            plotter_function(
                modules_limits,
                limits[0],
                limits[1],
                linewidth = lw_mod,
                color     = '0.5',
            )
            for copies_limits, sides_limits, linsetyle in zip(
                modules_copies_limits,
                modules_sides_limits,
                modules_ls
            ):
                # Between copies
                plotter_function(
                    copies_limits,
                    limits[0],
                    limits[1],
                    linewidth = lw_cop,
                    color     = '0.5',
                    linestyles = linsetyle,
                )
                # Between sides
                plotter_function(
                    sides_limits,
                    limits[0],
                    limits[1],
                    linewidth = lw_sid,
                    color     = 'g',
                    linestyles = '--',
                )

    plot_lines(0, hlimits)
    plot_lines(1, vlimits)
    return

# \-------- [ AUXILIARY FUNCTIONS ] --------

# -------- [ RASTER PLOTS ] --------
def plot_raster_plot(
    pop                 : b2.NeuronGroup,
    spikemon_t          : np.ndarray,
    spikemon_i          : np.ndarray,
    duration            : float,
    network_modules_list: list[SnnNetworkModule],
    plotpars            : dict,
    duration_ratio      : float = 1.0
) -> None:
    '''
    Raster plot of the recorded neural activity. Limbs are inserted in the plot according to
    their position in the axial network.
    '''

    # PARAMETERS
    n_tot          = len(pop)
    t_start_ms     = float( spikemon_t[0] / b2.msecond )
    duration_ms    = float( duration / b2.msecond )
    sampling_ratio = plotpars.get('sampling_ratio', 1.0)

    t_start_ms  = t_start_ms + (1 - duration_ratio) * duration_ms
    duration_ms = duration_ms * duration_ratio

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

    modules_pools_colors = [
        (
            mod.plotting.get('color_pools')
            if mod.plotting.get('color_pools')
            else
            [ mod.plotting.get('color') ] * mod.pools
        )
        for mod in network_modules_list
    ]

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

    # SELECT TARGET TIME WINDOW
    spikes_window_inds = (
        (spikes_t_pruned >= t_start_ms) &
        (spikes_t_pruned <= t_start_ms + duration_ms)
    )
    spikes_i_pruned = spikes_i_pruned[spikes_window_inds]
    spikes_t_pruned = spikes_t_pruned[spikes_window_inds]

    # SAMPLE SPIKES
    n_spikes = len(spikes_i_pruned)
    sampled_spikes_inds = np.random.rand(n_spikes) <= sampling_ratio

    spikes_t_sampled = spikes_t_pruned[sampled_spikes_inds]
    spikes_i_sampled = spikes_i_pruned[sampled_spikes_inds]

    # MAP MODULE LIMITS
    modules_indices_pools_mapped = [
        [
            np.array(mod_pool_inds, dtype= int) - inds_mask[mod_pool_inds]
            for mod_pool_inds in mod.indices_pools
        ]
        for mod in network_modules_list
    ]

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

    # PLOT
    plt.xlim(t_start_ms, t_start_ms + duration_ms)
    plt.ylim(0, modules_limits_mapped[-1, -1])

    # Separate modules
    plt.hlines(
        y          = modules_limits_mapped[:, 0],
        xmin       = t_start_ms,
        xmax       = t_start_ms + duration_ms,
        linestyles = '-',
        linewidth  = 0.4,
        color      = '0.5'
    )

    for (
        mod_pools_inds,
        mod_copies_limits,
        mod_color,
        mod_pools_colors,
        ) in zip(
        modules_indices_pools_mapped,
        modules_copies_limits_mapped,
        modules_colors,
        modules_pools_colors,
    ):

        if isinstance(mod_color, list):
            if len(mod_color) == 1:
                mod_color = [ float(val) for val in mod_color[0].split(' ') ]
            elif len(mod_color) == 3:
                mod_color = [ float(val) for val in mod_color ]
            elif len(mod_color) == 4:
                mod_color = [ float(val) for val in mod_color ]
            else:
                raise ValueError('Color value not recognized')

            mod_color = mcolors.to_hex(
                [
                    ( val/255 if val > 1 else val )
                    for val in mod_color
                ]
            )

        # Separate copies
        plt.hlines(
            y          = mod_copies_limits[1:, 0],
            xmin       = t_start_ms,
            xmax       = t_start_ms + duration_ms,
            linestyles = '--',
            linewidth  = 0.4,
            color      = '0.5'
        )

        # Raster plot
        # mod_pool_inds = range( mod_limits[0], mod_limits[1] )
        for inds_pool, color_pool in zip( mod_pools_inds, mod_pools_colors):

            plt.scatter(
                spikes_t_sampled[ np.isin(spikes_i_sampled, inds_pool) ],
                spikes_i_sampled[ np.isin(spikes_i_sampled, inds_pool) ],
                color = color_pool,
                marker= '.',
                s     = 1,
            )

    # DECORATE
    plt.yticks(np.mean(modules_limits_mapped, axis=1), modules_labels)

    plt.xlabel('Time [ms]')
    plt.ylabel('Neuronal pools')
    plt.title('Neural activation')
    plt.xlim(t_start_ms, t_start_ms + duration_ms)

    # Invert y axis representation
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return

# \-------- [ RASTER PLOTS ] --------

# -------- [ SINGLE QUANTITY EVOLUTION ] --------
def plot_frequency_evolution(freqs: dict[str, np.ndarray]) -> None:
    '''
    Plots the instantaneous frequency of oscillations, derived from
    the hilbert transforms of the filtered signals.
    '''

    times  = freqs['times']
    f_inst_mean = freqs['mean']
    f_inst_std  = freqs['std']

    siglen = times.size

    # Mean value
    mean_f = np.mean( f_inst_mean[int(0.1*siglen) : int(0.9*siglen)] )

    ax1 = plt.axes()
    ax1.plot(times, f_inst_mean,
                label= 'Instantaneous frequency', color='#1B2ACC')

    ax1.plot( [ times[0], times[-1] ], [mean_f, mean_f], linewidth= 2,
                color= 'k', label= 'Mean frequency')

    ax1.fill_between(times,
                    (f_inst_mean - f_inst_std),
                    (f_inst_mean + f_inst_std),
                    edgecolor='#1B2ACC', facecolor='#089FFF', interpolate= True )

    ax1.set_xlim(times[1], times[-1])
    ax1.set_ylim(0, 5 )
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Instantaneous frequency')
    ax1.set_title('Evolution of mean instantaneous frequency')
    ax1.grid()
    ax1.legend()
    plt.tight_layout()
    return

def plot_ipl_evolution(ipls: np.ndarray,
                       plotpars: dict,
                       limb_pair_positions: list[int]= None ) -> None:
    '''
    Plots the instantaneous mean intersegmental phase lag, derived from
    the hilbert transforms of the filtered signals.
    '''

    trunk_only = plotpars.get('trunk_only', False)
    jump_at_girdles = plotpars.get('jump_at_girdles', False)

    if limb_pair_positions is None or not jump_at_girdles:
        limb_pair_positions = []

    times = ipls['times']
    seg_ipls = ipls['seg_ipls'] * 100
    mean_ipl_evolution = ipls['mean_ipl'] * 100
    std_ipl_evolution = ipls['std_ipl'] * 100

    siglen = times.size

    ax1 = plt.axes()

    # Cross-girdle IPLS
    if len(limb_pair_positions)>0 and jump_at_girdles and not trunk_only:
        for i, seg in enumerate(limb_pair_positions):
            if seg == 0 or seg == len(ipls):
                continue

            plt.plot(times, seg_ipls[seg], color= 'r', label= f'Girdle {i}')

    # Evolution of mean IPL
    ax1.plot(times, mean_ipl_evolution,
                label= 'Instantaneous IPL', color='#1B2ACC')

    ax1.fill_between(times,
                    (mean_ipl_evolution - std_ipl_evolution),
                    (mean_ipl_evolution + std_ipl_evolution),
                    edgecolor='#1B2ACC', facecolor='#089FFF', interpolate= True )

    # Mean IPL
    mean_ipl = np.mean(mean_ipl_evolution[int(0.2*siglen) : int(0.8*siglen)])
    ax1.plot( [ times[0], times[-1] ],
              [ mean_ipl, mean_ipl  ],
                linewidth= 2, color= 'k', label= 'Mean IPL')


    ax1.set_xlim(times[1], times[-1])

    miny = np.min( [np.min(mean_ipl_evolution - std_ipl_evolution)*1.1, 0] )
    maxy = np.max( [np.max(mean_ipl_evolution + std_ipl_evolution)*1.1, 0] )

    ax1.set_ylim( miny, maxy )
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Instantaneous IPL [%]')
    ax1.set_title('Evolution of intersegmental phase lag')
    ax1.grid()
    ax1.legend()
    plt.tight_layout()
    return

def plot_online_activities_lb(activities, timestep):
    ''' Online evolution of limbs' pools activities '''
    sigs = activities.shape[0]
    segs = sigs // 2
    steps = activities.shape[1]
    duration = steps * timestep
    times = np.arange(0, duration, timestep)
    incr = 1.05 * np.amax(activities)
    for sig_ind, activity in enumerate(activities):

        sig_incr = (sigs - 1 - sig_ind)//2 * incr
        plt.plot(
            times,
            sig_incr + activity,
            marker= 'o',
            markersize= 2,
            fillstyle= 'none',
            label= f'flexor_{sig_ind//2}' if sig_ind%2==0 else f'extensor_{sig_ind//2}'
        )

    gridlocs = [
        (segs - i) * incr - j
        for i in range(segs)
        for j in np.arange(0, incr/1.1, 0.5)
    ]
    plt.hlines(gridlocs, 0, duration, linestyles= 'dashed', colors= '0.6')

    ylocs = np.array( [ (segs - i) * incr for i in range(segs)] )
    ylabs = [ f'activity_{i}' for i in range(segs)]
    plt.hlines(ylocs, 0, duration)
    plt.yticks(ylocs - incr/2, ylabs)

    plt.legend()
    plt.xlim(0, duration)
    plt.xlabel('Times [s]')
    plt.ylabel('Activities')

    plt.tight_layout()
    return

def plot_online_periods_lb(periods, timestep):
    ''' Online evolution of limbs' pools periods '''
    sigs = periods.shape[0]
    segs = sigs // 2
    steps = periods.shape[1]
    duration = steps * timestep
    times = np.arange(0, duration, timestep)
    incr = 1.05 * np.amax(periods)
    for sig_ind, period in enumerate(periods):

        sig_incr = (sigs - 1 - sig_ind)//2 * incr
        plt.plot(
            times,
            sig_incr + period,
            marker= 'o',
            markersize= 2,
            fillstyle= 'none',
            label= f'flexor_{sig_ind//2}' if sig_ind%2==0 else f'extensor_{sig_ind//2}'
        )

    gridlocs = [
        (segs - i) * incr - j
        for i in range(segs)
        for j in np.arange(0, incr/1.1, 0.25)
    ]
    plt.hlines(gridlocs, 0, duration, linestyles= 'dashed', colors= '0.6')

    ylocs = np.array( [ i * incr for i in range(segs)] )
    ylabs = [ f'period_{i}' for i in range(segs)]
    plt.hlines(ylocs, 0, duration)
    plt.yticks(ylocs - incr/2, ylabs)

    plt.legend()
    plt.xlim(0, duration)
    plt.xlabel('Times [s]')
    plt.ylabel('Periods')

    plt.tight_layout()
    return

def plot_online_duties_lb(duties, timestep):
    ''' Online evolution of limbs' pools duties '''
    sigs = duties.shape[0]
    segs = sigs // 2
    steps = duties.shape[1]
    duration = steps * timestep
    times = np.arange(0, duration, timestep)
    incr = 1
    for sig_ind, duty in enumerate(duties):

        sig_incr = (sigs - 1 - sig_ind)//2 * incr
        plt.plot(
            times,
            sig_incr + duty,
            marker= 'o',
            markersize= 2,
            fillstyle= 'none',
            label= f'flexor_{sig_ind//2}' if sig_ind%2==0 else f'extensor_{sig_ind//2}'
        )

    gridlocs = [
        (segs - i) * incr - j
        for i in range(segs)
        for j in np.linspace(0, 1, 6)
    ]
    plt.hlines(gridlocs, 0, duration, linestyles= 'dashed', colors= '0.6')

    ylocs = np.array( [ i * incr for i in range(segs)] )
    ylabs = [ f'duty_{i}'    for i in range(segs)]
    plt.hlines(ylocs, 0, duration)
    plt.yticks(ylocs - incr/2, ylabs)

    plt.legend()
    plt.xlim(0, duration)
    plt.xlabel('Times [s]')
    plt.ylabel('Duties')

    plt.tight_layout()
    return

# \------- [ SINGLE QUANTITY EVOLUTION ] --------

# -------- [ MULTIPLE QUANTITIES EVOLUTION ] --------
def plot_temporal_evolutions(
        times           : np.ndarray,
        variables_values: list[np.ndarray],
        variables_names : list[str],
        inds            : list,
        three_dim       : int = False
    ) -> None:
    '''
    Temporal evolution of the recorded neural activity, from the selected indeces.\n
    Statemon_variables and varnames list the recorded quantities and their names for the plots.
    - If three_dim = True --> Values are represented in a 3D space (index, time, quantity)
    - If three_dim = False --> Eache statemon variable is plotted in a different subplot
    '''

    n_ind = min(len(inds), 100)
    n_ind_interval = len(inds) // n_ind

    inds = inds[::n_ind_interval]

    if three_dim:
        ax1 = plt.axes(projection='3d')

        for statemon_var in variables_values:
            red = random.randint(0,255)/255.0
            green = random.randint(0,255)/255.0
            blue = random.randint(0,255)/255.0
            color = (red, green, blue)

            for ind in inds:
                ax1.plot3D(
                    ind*np.ones(len(times)),
                    times,
                    statemon_var[ind],
                    color = color
                )

        ax1.set_xlabel('Inds')
        ax1.set_ylabel('Time (ms)')
        ax1.set_zlabel('Membrane potential')
        plt.title('Temporal evolution of neuronal variables')

    else:
        axs = [
            plt.subplot(len(variables_values), 1, i+1)
            for i in range(len(variables_values))
        ]

        for varnum, statemon_var in enumerate(variables_values):
            red = random.randint(0,255)/255.0
            green = random.randint(0,255)/255.0
            blue = random.randint(0,255)/255.0
            color = (red, green, blue)

            plt.setp(axs[varnum], ylabel=variables_names[varnum])

            # NOTE: Plot the opposite because the y-axis will be inverted
            axs[varnum].plot(
                statemon_var[inds].T,
                color = color,
                lw    = 0.5
            )

            # Invert y axis representation
            axs[varnum].set_xlim([0, len(times)])
            axs[varnum].set_ylim(axs[varnum].get_ylim())

        plt.xlabel('Time (step)')
        plt.setp(axs[0], title = 'Temporal evolution of neuronal variables')

    plt.tight_layout()
    return

def plot_processed_pools_activations(
        signals       : dict[str, np.ndarray],
        points        : dict[str, list[list[float]]],
        seg_axial     : int,
        seg_limbs     : int,
        duration      : float,
        plotpars      : dict,
    ) -> None:
    '''
    Plot the result of the processing of spiking data
    We obtained smooth signals, their onset, offset and com
    '''

    gridon         = plotpars.get('gridon', False)
    sampling_ratio = plotpars.get('sampling_ratio', 1.0)

    # Processed signals
    times_f       = signals['times_f']
    spike_count_f = signals['spike_count_f']

    if points is None or points == {}:
        points = {
            'com_x'   : np.array([]),
            'com_y'   : np.array([]),
            'strt_ind': np.array([]),
            'stop_ind': np.array([]),
        }

    com_x    = points['com_x']
    com_y    = points['com_y']
    strt_ind = points['strt_ind']
    stop_ind = points['stop_ind']

    increment = 1.05 * np.amax(spike_count_f)
    logging.info('Increment for processed pools activations: %.2f', increment)

    # Sample axial activations
    seg_axial_sampled  = 0 if seg_axial == 0 else max(1, round( seg_axial * sampling_ratio ))
    seg_axial_interval = 0 if seg_axial == 0 else seg_axial // seg_axial_sampled

    seg_inds_axial = [
        seg_ind
        for seg_ind in range(seg_axial)
        if  seg_ind % seg_axial_interval == 0
    ]

    seg_inds_limbs = [
        seg_ind
        for seg_ind in range(seg_axial, seg_axial + seg_limbs)
    ]

    # Auxiliary plotting functions
    # NOTE: Plot the opposite because the y-axis will be inverted
    def __plot_activity(seg_ind, seg_incr, sides, do_legend):
        ''' Plot activity of a segment'''
        color = 'tab:blue' if seg_ind % 2 == 0 else 'tab:orange'
        label =     sides[0] if seg_ind % 2 == 0 else sides[1]
        width =        0.5 if seg_ind % 2 == 0 else 0.25
        style =    'solid' if seg_ind % 2 == 0 else 'dashed'
        plt.plot(
            times_f,
            seg_incr + spike_count_f[seg_ind],
            c     = color,
            lw    = width,
            ls    = style,
            label = label if do_legend else None
        )

    def __plot_onsets_and_offsets(seg_ind, seg_incr, sides, do_legend):
        ''' Plot onsets and offsets of a segment '''
        if (
            not len(strt_ind) or
            not len(stop_ind) or
            not strt_ind[seg_ind].size or
            not stop_ind[seg_ind].size
        ):
            return

        color =   'blue' if seg_ind % 2 == 0 else 'tomato'
        label = sides[0] if seg_ind % 2 == 0 else sides[1]
        size  =      2.0 if seg_ind % 2 == 0 else 1.0
        plt.plot(
            times_f[ strt_ind[seg_ind] ],
            seg_incr + spike_count_f[ seg_ind ][ strt_ind[seg_ind] ],
            ls         = 'None',
            lw         = 0.5,
            marker     = '^',
            markersize = size,
            c          = color,
            label      = f'Start {label}' if do_legend else None,
        )
        plt.plot(
            times_f[ stop_ind[seg_ind] ],
            seg_incr + spike_count_f[ seg_ind ][ stop_ind[seg_ind] ],
            ls         = 'None',
            lw         = 0.5,
            marker     = 'v',
            markersize = size,
            c          = color,
            label      = f'Stop {label}' if do_legend else None,
        )

    def __plot_com(seg_ind, seg_incr, sides, do_legend):
        ''' Plot COM position of a segment '''
        if (
            not len(com_x) or
            not len(com_y) or
            not com_x[seg_ind].size or
            not com_x[seg_ind].size
        ):
            return

        label = sides[0] if seg_ind % 2 == 0 else sides[1]
        size  =      2.0 if seg_ind % 2 == 0 else 1.0
        plt.plot(
            com_x[seg_ind],
            seg_incr + com_y[seg_ind],
            ls         = 'None',
            marker     = 'x',
            markersize = size,
            c          = 'k',
            label      = f'COM {label}' if do_legend else None,
        )

    def __plot_all_signals(
            seg_inds : list[int],
            sides    : list[str],
            tick_pos : list[float],
            tick_lab : list[str],
            title_tag: str,
        ):
        ''' Plot all segments in seg_inds '''

        n_seg = len(seg_inds)
        for i, seg in enumerate(seg_inds):
            seg_incr  = (n_seg -1 - i) * increment
            do_legend = (i == 0)

            __plot_activity(2*seg,     seg_incr, sides, do_legend)
            __plot_activity(2*seg + 1, seg_incr, sides, do_legend)
            __plot_onsets_and_offsets(2*seg,     seg_incr, sides, do_legend)
            __plot_onsets_and_offsets(2*seg + 1, seg_incr, sides, do_legend)
            __plot_com(2*seg, seg_incr, sides, do_legend)

        # Grid
        if gridon:
            lines_y = [
                increment * i for i, _seg in enumerate(seg_inds)
            ]
            plt.hlines(lines_y, 0, duration, colors='k', linestyles='dashed', linewidth=0.4)

        plt.xlabel('Time [seconds]')
        plt.ylabel('Filtered spike count')
        plt.title(f'{title_tag} - Processed ativations')
        plt.legend(bbox_to_anchor=(1.01,1), loc="upper left")
        plt.xlim(times_f[0],times_f[-1])

        # Create ticks
        plt.yticks(tick_pos, tick_lab)

        # Restrict time interval
        tmin = float( times_f[0] )
        tmax = float( times_f[-1] )
        xmin = min(tmin + 0.4 * (tmax - tmin), 5)
        xmax = min(tmax - 0.2 * (tmax - tmin), xmin + 4)

        plt.xlim(xmin, xmax)
        plt.tight_layout()

    # PLOTTING
    subplots_n   = int( len(seg_inds_axial) > 0 ) + int( len(seg_inds_limbs) > 0 )
    subplots_ind = 1

    # Axis
    if len(seg_inds_axial) > 0:

        n_ax_ticks = min( seg_axial_sampled, 4 )
        inc_n_ax   = seg_axial_sampled // max( n_ax_ticks, 1)
        inc_ax     = increment * inc_n_ax
        inc_max    = ( len(seg_inds_axial) - 1 ) * increment
        n_ticks    = len(seg_inds_axial[::inc_n_ax])

        ticklab_ax = [f'$Ax_{{{seg}}}$' for seg in seg_inds_axial[::inc_n_ax]]
        tickpos_ax = [ inc_max - i * inc_ax for i in range(n_ticks)]

        plt.subplot(subplots_n, 1, subplots_ind)
        __plot_all_signals(
            seg_inds  = seg_inds_axial,
            sides     = ['Left', 'Right'],
            tick_pos  = tickpos_ax,
            tick_lab  = ticklab_ax,
            title_tag = 'Axial',
        )
        subplots_ind += 1

    # Limbs
    if len(seg_inds_limbs) > 0:

        n_lb_ticks = min( seg_limbs, 4 )
        inc_n_lb   = seg_limbs // max( n_lb_ticks, 1)
        inc_lb     = increment * inc_n_lb
        inc_max    = ( len(seg_inds_limbs) - 1 ) * increment
        n_ticks    = len(seg_inds_limbs[::inc_n_lb])

        ticklab_lb = [ f'$Lb_{{{ seg - seg_axial }}}$' for seg in seg_inds_limbs[::inc_n_lb] ]
        tickpos_lb = [ inc_max - i * inc_lb for i in range(n_ticks)]

        plt.subplot(subplots_n, 1, subplots_ind)
        __plot_all_signals(
            seg_inds  = seg_inds_limbs,
            sides     = ['Flexor', 'Extensor'],
            tick_pos  = tickpos_lb,
            tick_lab  = ticklab_lb,
            title_tag = 'Limbs'
        )

    plt.tight_layout()

    return

def plot_musclecells_evolutions_axial(
        musclemon_times: np.ndarray,
        musclemon_dict : dict[str, np.ndarray],
        module_mc      : SnnNetworkModule,
        plotpars       : dict,
        starting_time  : float= 0
    ) -> None:
    '''
    Temporal evolution of the muscle cells' activity.
    Antagonist muscle cells are represented in the same subplot.
    '''

    filtering      = plotpars.get('filtering', False)
    sampling_ratio = plotpars.get('sampling_ratio', 1.0)

    times       = musclemon_times
    variables   = [ var for var in musclemon_dict.keys() if var not in ['t', 'N' ] ]
    n_variables = len(variables)

    target_inds = times >= float(starting_time)
    times = times[target_inds]
    ntimes = len(times)

    # Sample axial activations
    segments_axial         = module_mc['axial'].pools
    seg_axial_sampled      = round( segments_axial * sampling_ratio )
    seg_axial_interval     = segments_axial // seg_axial_sampled
    seg_inds_axial_sampled = np.arange(0, segments_axial, seg_axial_interval, dtype= int)
    seg_axial_sampled      = len(seg_inds_axial_sampled)

    # Plot
    axs: list[plt.Axes] = [
        plt.subplot(n_variables, 1, i+1)
        for i in range(n_variables)
    ]

    for _, (axi, attr) in enumerate( zip(axs,variables) ):
        values = musclemon_dict.get(attr)[target_inds].T
        vrest  = np.amin(values)
        incr   = 1.05 * (np.amax(values[:, ntimes//10:]) - vrest)
        logging.info('AXIS - Increment for variable %s in muscle cell evolution: %.2f', attr, incr)

        if filtering:
            # LOW PASS BUTTERWORT, ZERO PHASE FILTERING
            dt_sig = times[1]-times[0]
            fnyq   = 0.5 / dt_sig
            fcut   = 10

            num, den = butter(5, fcut/fnyq)
            values   = filtfilt(num, den, values, axis=1)

        # Plotting
        locs = []
        labs = []
        n_ticks_axis = 4
        ticks_axis_interval = seg_axial_sampled // n_ticks_axis

        for inc_ind, seg_ind in enumerate(seg_inds_axial_sampled):
            increment = (seg_axial_sampled - 1 - inc_ind) * incr
            axi.plot(
                [ times[0],times[-1] ],
                [ increment, increment ],
                linewidth= 0.3,
                color = 'k'
            )

            if inc_ind % ticks_axis_interval == 0:
                locs.append( float( increment ) )
                labs.append( f'$AX_{{{seg_ind}}}$')

            # NOTE: Plot the opposite because the y-axis will be inverted
            axi.plot( times, values[seg_ind                 ] + increment, color = 'r', lw= 0.5 )
            axi.plot( times, values[seg_ind + segments_axial] + increment, color = 'b', lw= 0.5 )

        axi.set_title('AXIS - Muscle cells activations - ' + attr)
        axi.set_xlim(0, times[-1])
        axi.set_yticks(locs)
        axi.set_yticklabels(labs)

    plt.xlabel('Time [s]')
    plt.tight_layout()
    return

def plot_musclecells_evolutions_limbs(
        musclemon_times: np.ndarray,
        musclemon_dict : dict[str, np.ndarray],
        module_mc      : SnnNetworkModule,
        plotpars       : dict,
        starting_time  : float= 0
    ) -> None:
    '''
    Temporal evolution of the muscle cells' activity.
    Antagonist muscle cells are represented in the same subplot.
    '''

    filtering = plotpars.get('filtering', False)

    times       = musclemon_times
    variables   = [ var for var in musclemon_dict.keys() if var not in ['t', 'N' ] ]
    n_variables = len(variables)

    target_inds = times >= float(starting_time)
    times = times[target_inds]
    ntimes = len(times)

    # Indices
    n_limbs        = module_mc['limbs'].copies
    segments_limbs = module_mc['limbs'].pools
    seg_inds_limbs = module_mc['limbs'].indices_sides_copies

    # Plot
    axs: list[plt.Axes] = [
        plt.subplot(n_variables, 1, i+1)
        for i in range(n_variables)
    ]

    for _, (axi, attr) in enumerate( zip(axs,variables) ):
        values = musclemon_dict.get(attr)[target_inds].T
        vrest  = np.amin(values)
        incr   = 1.05 * (np.amax(values[:, ntimes//10:]) - vrest)
        logging.info('LIMBS - Increment for variable %s in muscle cell evolution: %.2f', attr, incr)

        if filtering:
            # LOW PASS BUTTERWORT, ZERO PHASE FILTERING
            dt_sig = times[1]-times[0]
            fnyq   = 0.5 / dt_sig
            fcut   = 10

            num, den = butter(5, fcut/fnyq)
            values   = filtfilt(num, den, values, axis=1)

        # Plotting
        locs = []
        labs = []
        n_ticks_limbs        = n_limbs
        ticks_limbs_interval = segments_limbs // n_ticks_limbs

        for limb_id, indices_sides_limb in enumerate(seg_inds_limbs):

            increment = (n_limbs - 1 - limb_id) * incr
            axi.plot(
                [ times[0],times[-1] ],
                [ increment, increment ],
                linewidth= 0.3,
                color = 'k'
            )

            if limb_id % ticks_limbs_interval == 0:
                locs.append( float( increment ) )
                labs.append( f'$LB_{{{limb_id}}}$')

            # NOTE: Plot the opposite because the y-axis will be inverted
            axi.plot( times, values[indices_sides_limb[0]].T + increment, color = 'r', lw= 0.5 )
            axi.plot( times, values[indices_sides_limb[1]].T + increment, color = 'b', lw= 0.5 )

        axi.set_title('LIMBS - Muscle cells activations - ' + attr)
        axi.set_xlim(0, times[-1])
        axi.set_yticks(locs)
        axi.set_yticklabels(labs)

    plt.xlabel('Time [s]')
    plt.tight_layout()
    return

# \------- [ MULTIPLE QUANTITIES EVOLUTION ] --------

#-------- [ CONNECTIVITY PLOTS ] --------
def plot_connectivity_matrix(
    pop_i: b2.NeuronGroup,
    pop_j: b2.NeuronGroup,
    w_syn: np.ndarray,
    network_modules_list_i: list[SnnNetworkModule],
    network_modules_list_j: list[SnnNetworkModule],
) -> None:
    '''
    Connectivity matrix showing the links in the network.
    '''

    if not np.any(w_syn):
        return

    # PARAMETERS
    n_tot_i = len(pop_i)
    n_tot_j = len(pop_j)
    plt.xlim(-0.5, n_tot_i - 0.5)
    plt.ylim(-0.5, n_tot_j - 0.5)

    # ORIGIN
    locs_i, labs_i = get_modules_labels(network_modules_list_i)
    plot_modules_grid(
        network_modules_list = network_modules_list_i,
        vlimits              = [0, n_tot_j],
        lw_mod               = 0.2,
        lw_cop               = 0.2,
        lw_sid               = 0.1,
    )

    # TARGET
    locs_j, labs_j = get_modules_labels(network_modules_list_j)
    plot_modules_grid(
        network_modules_list = network_modules_list_j,
        hlimits              = [0, n_tot_i],
        lw_mod               = 0.2,
        lw_cop               = 0.2,
        lw_sid               = 0.1,
    )

    ## DECORATE PLOT
    plt.xticks(locs_i, labs_i, rotation = 90)
    plt.yticks(locs_j, labs_j, rotation = 0)

    plt.title('Connectivity matrix')
    plt.xlabel('Pre-synaptic nurons')
    plt.ylabel('Post-synaptic neurons')

    # PLOT
    plt.imshow(w_syn.T, cmap = 'seismic') #origin ='upper')
    plt.tight_layout()
    return

def plot_limb_connectivity(
        w_syn           : np.ndarray,
        cpg_limbs_module: SnnNetworkModule,
        plot_label      : str = ''
    ) -> None:
    '''
    Connectivity matrix showing the links between the limbs in the network.
    '''

    if not cpg_limbs_module.include:
        return

    plot_label = plot_label if plot_label == '' else plot_label + ' - '
    plot_label = plot_label.upper()

    # Parameters
    limbs   = cpg_limbs_module.copies
    n_limbs = cpg_limbs_module.n_tot

    # Limits
    pmin, pmax = [-0.5, n_limbs - 0.5]
    plt.xlim(pmin, pmax)
    plt.ylim(pmin, pmax)

    # Consider only inter-limb connectivity
    ind_min, ind_max = cpg_limbs_module.indices_limits
    w_limbs = w_syn[ ind_min:ind_max, ind_min:ind_max]

    ind0      = 0
    tick_labs = []
    tick_pos  = []

    for mod in cpg_limbs_module.sub_parts_list:
        # Separate modules
        plt.hlines( y = [ind0] , xmin= pmin, xmax= pmax, c='k', lw= 0.4 )
        plt.vlines( x = [ind0] , ymin= pmin, ymax= pmax, c='k', lw= 0.4 )

        # Separate limbs
        lines_inds = [ ind0 + lmb_ind * mod.n_copy - 0.5 for lmb_ind in range(1,limbs) ]
        plt.hlines( y = lines_inds, xmin= pmin, xmax= pmax, c='0.5', lw= 0.2, ls= '--' )
        plt.vlines( x = lines_inds, ymin= pmin, ymax= pmax, c='0.5', lw= 0.2, ls= '--' )

        # Ticks
        tick_labs += [f'{mod.type}.lmb_{i}' for i in range(limbs)]
        tick_pos  += [ ind0 + mod.n_copy*i + mod.n_copy//2 for i in range(limbs) ]

        ind0 += mod.n_tot

    # Decorate plot
    plt.xticks(tick_pos, tick_labs)
    plt.yticks(tick_pos, tick_labs)
    plt.title(plot_label + 'Limbs connectivity matrix')
    plt.xlabel('Pre-synaptic nurons')
    plt.ylabel('Post-synaptic neurons')

    plt.imshow(w_limbs.T, cmap = 'seismic') #origin ='upper')
    plt.tight_layout()
    return

# \-------- [ CONNECTIVITY PLOTS ] --------

# \-------- [ INTERNAL PARAMETERS ] --------
def plot_neuronal_identifiers(
        pop                 : b2.NeuronGroup,
        network_modules_list: list[SnnNetworkModule],
        identifiers_list    : list[str],
    ) -> None:
    ''' Plots internal parameters of the neuronal population'''

    figures_dict = {
              'i': [   'index [#]', pop.i       ],
         'ner_id': [  'ner_id [#]', pop.ner_id  ],
        'side_id': [ 'side_id [#]', pop.side_id ],
         'y_neur': ['position [m]', pop.y_neur  ],
        'pool_id': [ 'pool_id [#]', pop.pool_id ],
        'limb_id': [ 'limb_id [#]', pop.limb_id ],
          'I_ext': [   'drive [A]', pop.I_ext   ],
    }

    locs, labs = get_modules_labels(network_modules_list)

    figures_list = [ figures_dict[identifier] for identifier in identifiers_list]
    n_figures = len(figures_list)

    for ind, (title, values) in enumerate(figures_list):
        plt.subplot(n_figures, 1, ind + 1)
        plt.title(title)
        plt.plot(values)
        if ind == n_figures - 1:
            plt.xticks(locs, labs, rotation = 45)
        else:
            plt.xticks(locs, labels='')


        vmin, vmax = np.amin(values), np.amax(values)
        gmin = vmin - 0.1 * (vmax - vmin)
        gmax = vmax + 0.1 * (vmax - vmin)

        plt.xlim(0, len(pop))
        plt.ylim(gmin, gmax)
        plot_modules_grid(
            network_modules_list,
            vlimits = [gmin, gmax]
        )
        plt.grid(axis='y')

    plt.tight_layout()
    return
# \-------- [ INTERNAL PARAMETERS ] --------
