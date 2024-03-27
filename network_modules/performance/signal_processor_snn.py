'''
Module to process neural signals from the simulations
And to compute metrics to evaluate network's performance
'''
import numpy as np

from collections import defaultdict
from bisect import bisect_left
from csaps import csaps
from scipy.signal import butter, filtfilt, find_peaks, fftconvolve, hilbert, medfilt

from scipy import signal
import matplotlib.pyplot as plt

import brian2 as b2

from network_modules.equations import utils
from network_modules.performance import signal_processor_snn_plot
from network_modules.parameters.network_module import SnnNetworkModule

# ---------------[ UTILS ] -----------------
def filter_signals(
    signals   : np.ndarray,
    signal_dt : float,
    fcut_hp   : float = None,
    fcut_lp   : float = None,
    filt_order: int = 5,
):
    ''' Butterwort, zero-phase filtering '''

    # Nyquist frequency
    fnyq = 0.5 / signal_dt

    # Filters
    if fcut_hp is not None:
        num, den = butter(filt_order, fcut_hp/fnyq,  btype= 'highpass')
        signals  = filtfilt(num, den, signals)

    if fcut_lp is not None:
        num, den = butter(filt_order, fcut_lp/fnyq, btype= 'lowpass' )
        signals  = filtfilt(num, den, signals)

    return signals

def remove_signals_offset(signals: np.ndarray):
    ''' Removed offset from the signals '''
    return (signals.T - np.mean(signals, axis = 1)).T

# \--------------[ UTILS ] -----------------

# ---------------[ PROCESS SPIKING DATA ] -----------------
def count_spikes_intervals(
        sorted_spikes_indices: list[int],
        indices_intervals    : list[int]
    ) -> list[int]:
    '''
    Count spikes from all the pools specified by the intervals
    '''
    counts = defaultdict(int)
    indices_intervals.sort()
    for item in sorted_spikes_indices:
        pos = bisect_left(indices_intervals, item)
        if pos == len(indices_intervals):
            counts[None] += 1
        else:
            counts[indices_intervals[pos]] += 1
    return counts

def count_spikes_indices(
        counts        : np.ndarray,
        spikes_indices: list[int],
        indices_list  : list[np.ndarray]
    ) -> list[int]:
    '''
    Count spikes from all the pools specified by the intervals
    '''

    if not len(spikes_indices):
        return counts

    for i, inds in enumerate(indices_list):
        counts[i] = np.sum(np.isin(spikes_indices, inds))

    return counts

def compute_spike_count_limbs_online(
        running_spike_count: np.ndarray,
        spikemon           : b2.SpikeMonitor,
        curtime            : float,
        sig_dt             : float,
        callback_dt        : float,
        cpg_limb_module    : SnnNetworkModule,
    ) -> tuple[np.ndarray, np.ndarray]:
    '''
    Updates the spike count of excitatory pools for the current timestep
    '''

    window_n = running_spike_count.shape[1]
    callback_n = round( callback_dt / sig_dt )

    cur_step = round( curtime / sig_dt) % window_n
    considered_steps = np.arange(cur_step, cur_step + callback_n, dtype= int) % window_n

    # Indeces of limb neurons (flexor, extensor)
    indmin, indmax = cpg_limb_module['ex'].indices_limits

    ex_ind_f = cpg_limb_module['ex'].indices_pools_sides[0]
    ex_ind_e = cpg_limb_module['ex'].indices_pools_sides[1]

    count_cpg_ex_f = np.zeros( ( cpg_limb_module['ex'].pools ) )
    count_cpg_ex_e = np.zeros( ( cpg_limb_module['ex'].pools ) )
    count_cpg_ex   = np.zeros( ( cpg_limb_module['ex'].pools * 2 ) )

    for step in range(callback_n):

        # Spike indeces for the desired timestep
        target_spikes = (spikemon.t == curtime - callback_n * sig_dt + step * sig_dt)
        spikes_i = np.array( spikemon.i[ target_spikes ])
        spikes_i = spikes_i[ np.logical_and( spikes_i >= indmin, spikes_i <= indmax) ]

        count_cpg_ex_f = count_spikes_indices(count_cpg_ex_f, spikes_i, ex_ind_f)
        count_cpg_ex_e = count_spikes_indices(count_cpg_ex_e, spikes_i, ex_ind_e)
        count_cpg_ex[0::2] = count_cpg_ex_f
        count_cpg_ex[1::2] = count_cpg_ex_e

        running_spike_count[:, considered_steps[step] ] = count_cpg_ex

    return running_spike_count

def compute_spike_count_module(
        spikemon_t    : np.ndarray,
        spikemon_i    : np.ndarray,
        dt_sig        : float,
        duration      : float,
        net_module    : SnnNetworkModule,
        ner_type      : str = 'ex',
    ) -> tuple[np.ndarray, np.ndarray]:
    ''' Compute the spike count for every pool of the specified ner_type '''

    # Spike times and indeces
    spikemon_t  = np.array( spikemon_t % float(duration) // float(dt_sig), dtype= int )

    imin, imax = net_module.indices_limits
    considered_spikes = np.logical_and( spikemon_i >= imin, spikemon_i <= imax )

    spikemon_t = spikemon_t[ considered_spikes ]
    spikemon_i = spikemon_i[ considered_spikes ] - imin

    # Indeces of the pools of excitatory neurons (left, right)
    ner_ind_l = - imin + np.concatenate(
        [
            inds
            for inds in [
                net_module['axial'][ner_type].indices_pools_sides[0],
                net_module['limbs'][ner_type].indices_pools_sides[0],
            ]
            if len(inds) > 0
        ],
        axis = 0
    )

    ner_ind_r = - imin + np.concatenate(
        [
            inds
            for inds in [
                net_module['axial'][ner_type].indices_pools_sides[1],
                net_module['limbs'][ner_type].indices_pools_sides[1],
            ]
            if len(inds) > 0
        ],
        axis = 0
    )

    ### Collect spikes in a sparse format
    n_steps = int(duration/dt_sig)
    times   = np.linspace(0, duration, n_steps)

    spike_trains = np.zeros( ( net_module.n_tot, n_steps ) )
    spike_trains[spikemon_i, spikemon_t ] = 1

    ### Compute population activity at each timestep
    spike_count = np.zeros( ( 2*net_module.pools, n_steps ) )

    for seg in range( net_module.pools):
        spike_count[2*seg, :]     = np.sum( spike_trains[ ner_ind_l[seg], :], axis= 0 )
        spike_count[2*seg + 1, :] = np.sum( spike_trains[ ner_ind_r[seg], :], axis= 0 )

    return times, spike_count

def compute_spike_count_all_online(
    running_spike_count: np.ndarray,
    spikemon           : b2.SpikeMonitor,
    window_n           : float,
    curtime            : float,
    dt_sig             : float,
    n_e_semi           : int,
    n_i_semi           : int,
    pools_n            : tuple[str, int, int]) -> tuple[np.ndarray, np.ndarray]:
    '''
    Updates the spike count of all pools for the current timestep
    '''

    # Spike indeces for the last timestep
    spikes_i = [] if len(spikemon) == 0 else spikemon.i[spikemon.t == spikemon.t[-1]]
    spikes_i = np.sort(spikes_i)

    # Intervals of the pools
    segments_cpg = sum( [pool[2] for pool in pools_n[:2] ] )
    subpools = sum( [pool[2] for pool in pools_n[:] ] )

    n_cpg = sum( [pool[1] for pool in pools_n[:2] ] )
    n_tot = sum( [pool[1] for pool in pools_n[:]  ] )

    # Indeces of excitatory axial neurons (left, right)
    seg_ind = [n_e_semi, n_e_semi, n_i_semi, n_i_semi]
    intervals_cpg = [
        seg * 2 * (n_e_semi + n_i_semi) + np.sum( seg_ind[:i+1], dtype= int )
        for seg in range(segments_cpg)
        for i in range(len(seg_ind))
    ]

    # Indeces of the neurons for all the other pools (left, right)
    intervals_pools = []

    start_ind = n_cpg
    for _, pool_n, sub_pools in pools_n[2:]:
        if pool_n == 0:
            continue

        sp_n = pool_n // sub_pools
        sp_semi_n = sp_n // 2

        intervals_pool = [
            start_ind + sp_n * sp + sp_semi_n * side
            for sp in range(sub_pools)
            for side in [1,2]
        ]

        intervals_pools += intervals_pool
        start_ind += pool_n

    # Compute population activity at last timestep
    n_step = round( curtime / dt_sig) % window_n

    running_spike_count[n_step] = count_spikes_intervals(
        spikes_i,
        intervals_cpg + intervals_pools
    )

    return running_spike_count

def compute_spike_count_all(
    spikemon: b2.SpikeMonitor,
    dt_sig  : float,
    duration: float,
    n_e_semi: int,
    n_i_semi: int,
    pools_n : tuple[str, int, int]) -> tuple[np.ndarray, np.ndarray]:
    ''' Compute the spike count for every excitatory pool '''

    segments = sum( [pool[2] for pool in pools_n[:2] ] )
    subpools = sum( [pool[2] for pool in pools_n[:] ] )

    n_cpg = sum( [pool[1] for pool in pools_n[:2] ] )
    n_tot = sum( [pool[1] for pool in pools_n[:]  ] )

    # Indeces of excitatory axial neurons (left, right)
    ex_ind_l, ex_ind_r, _, _ = utils.cpg_seg_pools_indices(
        segments,
        n_e_semi,
        n_i_semi
    )

    # Indeces of the neurons for all the other pools (left, right)
    pools_ind_l = []
    pools_ind_r = []

    start_ind = n_cpg
    for _, pool_n, sub_pools in pools_n[2:]:
        if pool_n == 0:
            continue

        sp_n = pool_n // sub_pools
        sp_semi_n = sp_n // 2

        inds_l = [
            np.arange( start_ind + sp_n * sp, start_ind + sp_n * sp + sp_semi_n)
            for sp in range(sub_pools)
        ]
        inds_r = [
            np.arange( start_ind + sp_n * sp + sp_semi_n, start_ind + sp_n * (sp+1) )
            for sp in range(sub_pools)
        ]

        pools_ind_l += inds_l
        pools_ind_r += inds_r
        start_ind += pool_n

    # Spike times and indeces
    spike_t = np.array( np.array(spikemon.t) % float(duration)//dt_sig, dtype= int)
    spike_i = np.array(spikemon.i)

    ### Collect spikes in a sparse format
    n_steps = int(duration/dt_sig)
    times = np.linspace(0, duration, n_steps)

    spike_trains = np.zeros( (n_tot, n_steps) )
    spike_trains[spike_i, spike_t ] = 1

    ### Compute population activity at each timestep
    spike_count = np.zeros( (2*subpools, n_steps) )

    # CPG network
    # TODO: Substitute with spike trains

    for seg in range(segments):
        ex_l = ex_ind_l[seg]
        ex_r = ex_ind_r[seg]

        spike_count[2*seg, :]     = np.sum( spike_trains[ex_l, :], 0 )
        spike_count[2*seg + 1, :] = np.sum( spike_trains[ex_r, :], 0 )

    # Non-CPG network
    for seg in range(segments, subpools):
        inds_l = pools_ind_l[seg]
        inds_r = pools_ind_r[seg]

        spike_count[2*seg, :]     = np.sum( spike_trains[inds_l, :], 0 )
        spike_count[2*seg + 1, :] = np.sum( spike_trains[inds_r, :], 0 )

    return times, spike_count

def compute_smooth_activation_module(
        spikemon_t : np.ndarray,
        spikemon_i : np.ndarray,
        dt_sig     : float,
        duration   : float,
        net_module : SnnNetworkModule,
        ner_type   : str = 'ex',
    ) -> tuple[np.ndarray, np.ndarray]:
    ''' Convert the spike trains in a continuous curve representing the activation '''

    times, spike_count = compute_spike_count_module(
        spikemon_t = spikemon_t,
        spikemon_i = spikemon_i,
        dt_sig     = dt_sig,
        duration   = duration,
        net_module = net_module,
        ner_type   = ner_type,
    )

    ### Apply csaps function
    dt_f      = 2 * dt_sig
    n_steps_f = int(duration / dt_f)

    times_f       = np.linspace(0, duration, n_steps_f)
    spike_count_f = csaps( times, spike_count, times_f, smooth = 0.996, axis= 1 )

    return times_f, spike_count_f

def compute_smooth_activation_all(
    spikemon: b2.SpikeMonitor,
    dt_sig  : float,
    duration: float,
    n_e_semi: int,
    n_i_semi: int,
    pools_n : tuple[str, int, int]) -> tuple[np.ndarray, np.ndarray]:
    ''' Convert the spike trains in a continuous curve representing the activation '''

    times, spike_count = compute_spike_count_all(
        spikemon,
        dt_sig,
        duration,
        n_e_semi,
        n_i_semi,
        pools_n
    )

    # Downsample and apply cubic spline approximation
    n_steps_f = int( duration/ (2 * dt_sig) )

    times_f = np.linspace(0, duration, n_steps_f)
    spike_count_f = csaps( times, spike_count, times_f, smooth = 0.996, axis= 1 )

    return times_f, spike_count_f

# \---------------[ PROCESS SPIKING DATA ] -----------------

# ---------------[ PROCESS FILTERED DATA (single signal) ] -----------------
def detect_threshold_crossing_online(
      signal              : float,
      crossing            : dict,
      onsets              : list,
      offsets             : list,
      th1                 : float,
      th2                 : float,
      curtime             : float,
      discard_time        : float = 1 * b2.second,
    ) -> tuple[list, list]:
    ''' Detect threshold crossing with hystheresis '''

    if curtime < discard_time:
        return

    # Onset detection
    if signal < th1:
        crossing['on1'] = False

    elif not crossing['on1']:
        # Crossing first threshold (up)
        crossing['on1'] = True
        crossing['onset_candidate'] = curtime

    if signal < th2:
        crossing['on2'] = False

    elif not crossing['on2']:
        # Crossing second threshold (up)
        crossing['on2'] = True
        if abs( crossing['onset_candidate'] - discard_time ) > 1 * b2.usecond and \
            ( len(onsets) == 0 or crossing['onset_candidate'] != onsets[-1] ):
            onsets.append(crossing['onset_candidate'])

    # Offset detection
    if signal > th2:
        crossing['off2'] = False

    elif not crossing['off2']:
        # Crossing first threshold (down)
        crossing['off2'] = True
        crossing['offset_candidate'] = curtime

    if signal > th1:
        crossing['off1'] = False

    elif not crossing['off1']:
        # Crossing second threshold (down)
        crossing['off1'] = True
        if abs( crossing['offset_candidate'] - discard_time ) > 1 * b2.usecond and \
            ( len(offsets) == 0 or crossing['offset_candidate'] != offsets[-1] ):
            offsets.append(crossing['offset_candidate'])

    return onsets, offsets

def detect_threshold_crossing_offline(
      times               : np.ndarray,
      smooth_signal       : np.ndarray,
      discard_time        : float = 0*b2.msecond
    ) -> tuple[list, list]:
    '''
    Single signal: Compute the start and end time of the bursing activity
    based on a threshold level
    '''

    discard_index = np.argwhere( np.diff( np.array(times) > float(discard_time) ) )[0,0]

    sig_mean = np.mean(smooth_signal[discard_index:])
    sig_std  = np.std(smooth_signal[discard_index:])

    threshold1 = sig_mean - 0.5 * sig_std
    threshold2 = sig_mean - 0.3 * sig_std

    onsets = np.array([])
    offsets = np.array([])

    ### Crossing of the first threshold
    th_crossed1 = smooth_signal>threshold1
    th_crossing1 = np.diff( th_crossed1 )

    cross1_1 = np.argwhere(th_crossing1)[::2,0]
    cross2_1 = np.argwhere(th_crossing1)[1::2,0]

    if (len(cross1_1)==0) or (len(cross2_1)==0):
        return onsets, offsets

    aux = int( th_crossed1[cross1_1[0]+1] )- int( th_crossed1[cross1_1[0]] ) # Disambiguate
    if aux < 0:
        strt1 = cross2_1[ cross2_1>discard_index ]
        stop1 = cross1_1[ cross1_1>discard_index ]
    elif aux > 0:
        strt1 = cross1_1[ cross1_1>discard_index ]
        stop1 = cross2_1[ cross2_1>discard_index ]
    else:
        raise ValueError('Found value is not a threshold crossing point')

    if (len(strt1)==0) or (len(stop1)==0):
        return onsets, offsets

    # Signal begins with a stop
    if strt1[0]>stop1[0]:
        if len(stop1)==1:
            return onsets, offsets
        stop1 = stop1[1:]    #Discard first crossing

    # Signal ends with a start
    if strt1[-1]>stop1[-1]:
        if len(strt1)==1:
            return onsets, offsets
        strt1 = strt1[:-1] #Discard last crossing

    ### Crossing of the second threshold
    th_crossed2 = smooth_signal>threshold2
    th_crossing2 = np.diff( th_crossed2 )

    cross1_2 = np.argwhere(th_crossing2)[::2,0]
    cross2_2 = np.argwhere(th_crossing2)[1::2,0]

    if (len(cross1_2)==0) or (len(cross2_2)==0):
        return onsets, offsets

    aux = int( th_crossed2[cross1_2[0]+1] )- int( th_crossed2[cross1_2[0]] ) # Disambiguate
    if aux < 0:
        strt2 = cross2_2[ cross2_2>discard_index ]
        stop2 = cross1_2[ cross1_2>discard_index ]
    elif aux > 0:
        strt2 = cross1_2[ cross1_2>discard_index ]
        stop2 = cross2_2[ cross2_2>discard_index ]
    else:
        raise ValueError('Found value is not a threshold crossing point')

    if (len(strt2)==0) or (len(stop2)==0):
        return onsets, offsets

    # Signal begins with a stop
    if strt2[0]>stop2[0]:
        if len(stop2)==1:
            return onsets, offsets
        stop2 = stop2[1:]    #Discard first crossing

    # Signal ends with a start
    if strt2[-1]>stop2[-1]:
        if len(strt2)==1:
            return onsets, offsets
        strt2 = strt2[:-1] #Discard last crossing

    ### Consider crossings starting with the first threshold
    strt2 = strt2[ (strt2>strt1[0]) ]
    stop2 = stop2[ (stop2>strt1[0]) ]

    ### Consider crossings ending with the first threshold
    strt2 = strt2[ (strt2<stop1[-1]) ]
    stop2 = stop2[ (stop2<stop1[-1]) ]

    if (len(strt1)==0) or (len(stop1)==0) or (len(strt2)==0) or (len(stop2)==0):
        return onsets, offsets

    ### Prune multiple threshold crossing
    onsets, offsets = prune_crossings(strt1, stop1, strt2, stop2)

    return onsets, offsets

def prune_crossings(on1: list, off1: list, on2: list, off2: list) -> tuple[list, list]:
    ''' Retain only double threshold crossings '''
    ind1 = ind2 = 0
    onsets = []
    offsets = []

    rise = True
    while ind1 < len(on1) and ind2 < len(on2):
        if rise:
            if on1[ind1]<on2[ind2]: # Look for the crossing from th1 to th2
                if off1[ind1]>on2[ind2]:
                    onsets.append(on1[ind1]) # Found
                    rise = not rise
                else:
                    ind1 += 1
        else:
            if off2[ind2]<off1[ind1]: # Look for the crossing from th2 to th1
                if ( ind2+1 == len(on2) ) or ( on2[ind2+1]>off1[ind1] ):
                    offsets.append(off2[ind2]) # Found
                    ind2 += 1
                    rise = not rise
                else:
                    ind2 += 1

    return np.array(onsets), np.array(offsets)

def center_of_mass(
        times        : np.ndarray,
        values       : np.ndarray,
        start_indices: list,
        stop_indices : list
    ) -> tuple[list, list, list]:
    ''' Compute centroid of the bursts for a single signal '''

    times = np.array(times)
    n_cycles = start_indices.shape[0]

    com_x = np.zeros( (n_cycles, 1) )
    com_y = np.zeros( (n_cycles, 1) )
    surfaces = np.zeros( (n_cycles, 1) )

    for cycle in range(n_cycles):

        start = int( start_indices[cycle] )
        stop  = int( stop_indices[cycle] )
        times_cycle = times[start:stop]

        # line connecting the start and end points
        slope = (values[stop] - values[start]) / (times[stop] - times[start])
        off = values[start] - slope * times[start]
        baseline = slope * times_cycle + off

        magnitudes = values[start:stop] - baseline
        com_sum = np.sum( magnitudes.dot(times_cycle) )

        if np.sum(magnitudes) == 0:
            com_x[cycle] = np.mean(times_cycle)
        else:
            com_x[cycle] = com_sum / np.sum(magnitudes)

        # Knusel:   com_y[cycle] = com_sum * 0.5 / np.sum(times_cycle) + a * com_x[cycle] + b
        # Bicanski: maybe wrong, I think we should just do
        #           0.5 * np.sum(magnitudes) / len(magnitudes) + a * com_x[cycle] + b

        com_y[cycle] = 0.5 * np.sum(magnitudes) / len(magnitudes) + slope * com_x[cycle] + off
        surfaces[cycle] = np.diff(times_cycle).T.dot( magnitudes[1:] )

    return com_x, com_y, surfaces

# \---------------[ PROCESS FILTERED DATA (single signal) ] -----------------

# ---------------[ PROCESS FILTERED DATA (all signals) ] -----------------
def burst_start_stop_2thresholds_allsignals(
        times         : np.ndarray,
        smooth_signals: np.ndarray,
        signals_ptccs : np.ndarray,
        discard_time  : float = 0*b2.msecond
    ) -> tuple[list, list]:
    '''
    Detect threshold crossing based on two threshold levels
    to increase robustness to noisy signals
    '''

    strt_ind = []
    stop_ind = []

    for sig_ind, sig in enumerate(smooth_signals):

        if signals_ptccs[sig_ind] <= 1.0:
            strt_ind.append( np.array([]) )
            stop_ind.append( np.array([]) )
            continue

        strt, stop = detect_threshold_crossing_offline(times, sig, discard_time)

        strt_ind.append( strt )
        stop_ind.append( stop )

    return strt_ind, stop_ind

def center_of_mass_allsignals(
        times        : np.ndarray,
        signals      : np.ndarray,
        start_indices: list,
        stop_indices : list
    ) -> tuple[list, list]:
    '''
    Multiple signals: compute the centroid of the activation of all signals
    '''

    seg = len(signals)
    com_x = []
    com_y = []

    for i in range(seg):
        comx, comy, _ = center_of_mass(times, signals[i], start_indices[i], stop_indices[i])
        com_x.append( comx )
        com_y.append( comy )

    return com_x, com_y

# \---------------[ PROCESS FILTERED DATA (all signals) ] -----------------

# ---------------[ OVERALL SIGNAL PROCESSING (single signals) ] -----------------
def process_smooth_signal_com(
        times       : np.ndarray,
        signal      : np.ndarray,
        discard_time: float = 0*b2.second,
        filtering   : bool = False
    ) -> tuple[list, list]:
    '''
    Compute the Centers of Mass (COM) of the oscillations
    of the provided signal (optionally pre-filtered)
    '''

    times = np.array(times)
    times = times - times[0]
    sig_dt = times[1]-times[0]

    if filtering:
        # ZERO-PHASE MAV FILTERING
        n_filt = int( 0.05/sig_dt )
        signal = np.convolve(signal, np.ones(n_filt)/n_filt, mode='valid')

    start_indices, stop_indices = detect_threshold_crossing_offline(times, signal, discard_time )
    com_x, com_y, _ = center_of_mass(times, signal, start_indices, stop_indices)

    return com_x, com_y

# \---------------[ OVERALL SIGNAL PROCESSING (single signals) ] -----------------

# ---------------[ OVERALL SIGNAL PROCESSING (all signals) ] -----------------
def process_network_smooth_signals_com(
        times        : np.ndarray,
        signals      : np.ndarray,
        signals_ptccs: np.ndarray,
        discard_time : float = 0*b2.msecond,
        filtering    : bool = False
    ) -> tuple[list, list, list, list]:
    '''
    Compute the COMs, onset times and offset times for every provided (smooth) signal
    '''
    dt_sig = times[1] - times[0]

    # Filter high-frequency noise
    smooth_signals = signals.copy()
    if filtering:
        smooth_signals = filter_signals(smooth_signals, dt_sig, fcut_lp= 50)

    # Eliminate offset to smooth_signals
    smooth_signals_nooffset = remove_signals_offset(smooth_signals)

    (
        start_indices,
        stop_indices
    ) = burst_start_stop_2thresholds_allsignals(times, smooth_signals_nooffset, signals_ptccs, discard_time)

    (
        com_x,
        com_y
    ) = center_of_mass_allsignals(times, smooth_signals, start_indices, stop_indices)

    return com_x, com_y, start_indices, stop_indices

# \---------------[ OVERALL SIGNAL PROCESSING (all signals)] -----------------

# ---------------[ COMPUTE SIGNAL TRANSFORMATIONS ] -----------------
def compute_hilb_transform(
        times       : np.ndarray,
        signals     : np.ndarray,
        discard_time: float = 0 * b2.second,
        filtering   : bool = True,
    ) -> tuple[dict[str, np.ndarray]]:
    '''
    Computes the hilbert transform of the input signals.
    Returns the analytic signals, instantaneus phase and instantaneus frequencies
    '''

    if len(signals.shape) == 1:
        signals = np.array([signals])

    n_signals = signals.shape[0]
    dt_sig    = times[1] - times[0]

    smooth_signals = signals.copy()
    if filtering:
        smooth_signals = filter_signals(smooth_signals, dt_sig, 0.5, 5.0)

    # Remove offset and initial transient
    smooth_signals  = remove_signals_offset(smooth_signals)

    istart          = round( float(discard_time)/float(dt_sig) )
    chopped_signals = smooth_signals[:, istart:]
    chopped_times   = times[istart:]
    siglen          = chopped_times.size

    # Compute hilbert transform
    # hilb_sigs   = np.zeros((2*segments, siglen   ))
    inst_phases = np.zeros((n_signals, siglen   ))
    inst_freqs  = np.zeros((n_signals, siglen-1 ))

    for i in range(n_signals):

        sig = chopped_signals[i]

        hilb_sig = hilbert(sig)
        inst_phase = np.unwrap(np.angle(hilb_sig))
        inst_freq = (np.diff(inst_phase) / (2.0*np.pi) / dt_sig)

        # hilb_sigs[i,:]   =  hilb_sig
        inst_phases[i,:] =  inst_phase
        inst_freqs[i,:]  =  inst_freq

    # hilb   = { 'times': chopped_times,     'sig': hilb_sigs }
    phases : dict[str, np.ndarray] = { 'times': chopped_times,     'sig': inst_phases }
    freqs  : dict[str, np.ndarray] = { 'times': chopped_times[1:], 'sig': inst_freqs }

    return phases, freqs

def compute_fft_transform(
    times   : np.ndarray,
    signals : np.ndarray,
    plot    : bool = False,
    plot_ind: int = 0,
) -> np.ndarray:
    ''' Computes the fourier transform of the input signals. '''
    dt_sig = times[1] - times[0]

    # Filter high-frequency noise
    smooth_signals = signals.copy()
    smooth_signals = remove_signals_offset(smooth_signals)
    smooth_signals = filter_signals(smooth_signals, dt_sig, fcut_lp= 50)

    # Compute FFT
    signals_fft = np.fft.fft(
        smooth_signals,
        axis= 1
    )

    if plot:
        signal_processor_snn_plot.plot_fft_signals(
            times       = times,
            signals_fft = signals_fft,
            sig_ind     = plot_ind,
        )
    return signals_fft

def compute_spectrogram(
    times   : np.ndarray,
    signals : np.ndarray,
    plot    : bool = False,
    plot_ind: int = 0,
):
    ''' Computes the fourier transform of the input signals. '''
    dt_sig = times[1] - times[0]

    # Filter high-frequency noise
    smooth_signals = signals.copy()
    smooth_signals = remove_signals_offset(smooth_signals)
    smooth_signals = filter_signals(smooth_signals, dt_sig, fcut_lp= 50)

    # Compute spectrogram
    signals_spectrograms = []
    for sig_ind, smooth_signal in enumerate(smooth_signals):
        freqs_spec, times_spec, fft_spec = signal.spectrogram(
            x    = smooth_signal,
            fs   = 1/dt_sig,
            nfft = 2048,
        )
        signals_spectrograms.append(fft_spec)

        fft_spec_max = np.argmax(fft_spec, axis=0) // 2

        if plot and plot_ind == sig_ind:
            # Plot spectrogram
            plt.pcolormesh(times_spec, freqs_spec, fft_spec, shading='gouraud')
            plt.plot(times_spec, fft_spec_max, 'r')
            plt.xlabel('Time [sec]')
            plt.ylabel('Frequency [Hz]')
            plt.show()

    return signals_spectrograms

# \---------------[ COMPUTE SIGNAL TRANSFORMATIONS ] -----------------

# ---------------[ COMPUTE METRICS ] -----------------
def compute_frequency_com( com_x: list ) -> np.ndarray:
    '''
    Computes the frequency of oscillations based on the np.mean lag between COMs.
    Note: Returns the mean frequency computed for each segment
    '''

    seg = len(com_x)
    freqs = np.zeros( (seg, 1) )

    for i, comx in enumerate(com_x):
        isi = np.array([]) if not comx.size else np.diff( comx, axis = 0 )
        mean_isi = np.array([]) if not isi.size else np.mean( isi )
        freq = 1/mean_isi if mean_isi.size else 0

        freqs[i] = freq

    return freqs

def compute_frequency_hilb( freqs: dict[str, np.ndarray] ) -> dict[str, np.ndarray]:
    '''
    Computes the frequency based on the hilbert transforms.
    Returns the temporal evolution of the mean instantaneous frequency and its std.
    '''
    times      = freqs['times']
    inst_freqs = freqs['sig']

    mean_inst_freq = np.mean(inst_freqs, 0)
    std_inst_freq  = np.std(inst_freqs, 0)

    # Filter out sharp peaks
    dt_sig = times[1]-times[0]

    mean_inst_freq = filter_signals(mean_inst_freq, dt_sig, fcut_lp= 10)
    std_inst_freq  = filter_signals( std_inst_freq, dt_sig, fcut_lp= 10)

    return { 'times': times, 'mean': mean_inst_freq, 'std': std_inst_freq}

def compute_frequency_fft(
    times      : np.ndarray,
    signals_fft: np.ndarray,
):
    ''' Computes the average frequency based on the FFT '''

    n_step = len(times)
    freqs  = np.arange(n_step) / (times[1] - times[0]) / n_step

    signals_fft_mod = np.abs(signals_fft)
    ind_max_signals  = np.argmax(signals_fft_mod[:, :n_step//2], axis= 1)
    freq_max_signals = freqs[ind_max_signals]

    return freq_max_signals, ind_max_signals

def compute_amplitudes_fft(
    signals : np.ndarray
):
    ''' Calculate the amplitude using the dominant frequency component of the signal's FFT '''
    num_signals = signals.shape[1]

    signals_fft       = np.fft.fft(signals, axis=0)
    fft_dominant_inds = np.argmax(np.abs(signals_fft[1:, :]), axis=0) + 1

    amplitudes = 2 * np.abs(signals_fft[fft_dominant_inds, np.arange(num_signals)]) / signals.shape[0]

    return amplitudes

def isolate_frequency_ifft(
    signals_fft     : np.ndarray,
    ind_max_signals : np.ndarray,
):
    ''' Isolates a frequency component in the FFT of the signals '''

    signals_fft_max = np.zeros_like(signals_fft)
    signals_fft_max[:, ind_max_signals] = signals_fft[:, ind_max_signals]

    return np.fft.ifft(signals_fft_max, axis=1)

def compute_ipls_hilb(
        phases                 : dict[str, np.ndarray],
        segments               : int,
        limbs_pairs_i_positions: list[int] = None,
        jump_at_girdles        : bool      = False,
        trunk_only             : bool      = False
    ) -> np.ndarray:
    '''
    Computes the IPL evolution based on the phase difference of the hilbert transforms.
    Returns the IPLs between adjacent segments and the evolution of the mean IPL
    '''

    if not bool(limbs_pairs_i_positions):
        limbs_pairs_i_positions = []

    times = phases['times']
    inst_phases = phases['sig']

    dt_sig = times[1] - times[0]
    siglen = inst_phases.shape[1]
    istart, istop = int(0.1*siglen), int(0.9*siglen)

    # Specify the intervals of segments to take into account
    # Only consider the trunk portion(s) if trunk_only == True
    # Estimate the IPL at girdles if jump_at_girdles == True

    if len(limbs_pairs_i_positions) == 0:
        # Can only use the entire net
        limits_to_consider = [0, segments]

    else:
        if not trunk_only and not jump_at_girdles:
            # Entire net
            limits_to_consider = [0, segments]

        elif not trunk_only and jump_at_girdles:
            # Entire net accounting for jumps at girdles
            limits_to_consider = [0] + limbs_pairs_i_positions + [segments]

        elif trunk_only and not jump_at_girdles:
            if len(limbs_pairs_i_positions) == 1:
                # Caudal part only
                limits_to_consider = limbs_pairs_i_positions + [segments]
            else:
                # Consider only the segments between the girdles
                limits_to_consider = [ limbs_pairs_i_positions[0], limbs_pairs_i_positions[-1] ]

        elif trunk_only and jump_at_girdles:
            if len(limbs_pairs_i_positions) == 1:
                # Caudal part only
                limits_to_consider = limbs_pairs_i_positions + [segments]

            else:
                # Consider only the segments between the girdles
                limits_to_consider = limbs_pairs_i_positions

    # Compute the IPLs for the specified intervals
    considered_segments = limits_to_consider[-1] - limits_to_consider[0]

    if not considered_segments:
        ipl_evolution = {}
        return ipl_evolution

    inst_ipls = np.zeros((considered_segments, istop-istart))
    lastlimb_num = 0

    for ind, lbpos in enumerate( limits_to_consider[:-1] ):

        next_lbpos = limits_to_consider[ind+1]

        for seg in range(lbpos, next_lbpos-1):
            # LAG BETWEEN ADJACENT SEGMENTS
            inst_phase1 = inst_phases[2*seg]
            inst_phase2 = inst_phases[2*seg+2]

            phase_diff = inst_phase1[istart:istop] - inst_phase2[istart:istop]
            phase_diff = medfilt(phase_diff, 101)
            phase_diff = phase_diff % (2*np.pi)

            ipl =  phase_diff / (2*np.pi)
            ipl[ipl> +0.5] = -1 + ipl[ipl> +0.5]
            ipl = medfilt(ipl, 251)

            inst_ipls[seg,:] = ipl

        if next_lbpos not in [0, segments] and not trunk_only:
            # ESTIMATE LAG ACROSS THE GIRDLE
            # Lag between the first segments after the (previous and next) limbs
            inst_phase1 = inst_phases[2*lbpos]
            inst_phase2 = inst_phases[2*next_lbpos]

            phase_diff = inst_phase1[istart:istop] - inst_phase2[istart:istop]
            phase_diff = medfilt(phase_diff, 101)
            phase_diff = phase_diff % (2*np.pi)

            ipl_tot =  phase_diff / (2*np.pi)
            ipl_tot = medfilt(ipl_tot, 251)

            # Lag in the spinal tract before the next limb
            ipl_tract = np.sum(inst_ipls[lastlimb_num : next_lbpos - 1], 0)
            ipl_tract = medfilt(ipl_tract, 101)
            lastlimb_num = next_lbpos

            # Estimate of cross-girdle lag
            ipl = ipl_tot - ipl_tract

            inst_ipls[next_lbpos - 1,:] = ipl

    # Evolution of the mean IPL
    mean_ipl_evolution = np.mean(inst_ipls, 0)
    std_ipl_evolution  = np.std(inst_ipls, 0)

    # Smoother evolution
    mean_ipl_evolution = filter_signals(mean_ipl_evolution, dt_sig, fcut_lp= 10)
    std_ipl_evolution  = filter_signals( std_ipl_evolution, dt_sig, fcut_lp= 10)

    ipl_evolution = {
        'times'   : times[istart:istop],
        'seg_ipls': inst_ipls,
        'mean_ipl': mean_ipl_evolution,
        'std_ipl' : std_ipl_evolution
    }

    return ipl_evolution

def compute_ipls_corr(
    times       : np.ndarray,
    signals     : np.ndarray,
    freqs       : np.ndarray,
    inds_couples: list[list[int, int]]
) -> np.ndarray:
    '''
    Computes the IPL evolution based on the cross correlation of signals.
    Returns the IPLs between adjacent signals
    '''
    n_couples = len(inds_couples)
    dt_sig    = times[1] - times[0]
    ipls      = np.zeros(n_couples)

    signals_f = ( np.mean(signals, axis=1) - signals.T ).T

    for ind_couple, (ind1, ind2) in enumerate(inds_couples):
        sig1 = signals_f[ind1]
        sig2 = signals_f[ind2]

        xcorr = np.correlate(sig2, sig1, "full")
        n_lag = np.argmax(xcorr) - len(xcorr) // 2
        t_lag = n_lag * dt_sig

        # plt.plot( ( np.arange(len(xcorr)) - len(xcorr)//2 ) * dt_sig,  np.abs(xcorr))

        ipls[ind_couple] = t_lag * np.mean( [freqs[ind1], freqs[ind2]] )

    return ipls

def compute_ptcc(
        times       : np.ndarray,
        signals     : np.ndarray,
        discard_time: float = 0 * b2.second,
        filtering   : bool  = False,
        plot        : bool  = False,
        plot_ind    : int   = 0,
    ) -> np.ndarray:
    '''
    Computes the peak-to-through correlatio coefficient
    based on the difference between the the maximum and the minimum of the
    auto-correlogram of smoothened pools' oscillations
    '''

    sig_dt = times[1] - times[0]

    smooth_signals = signals.copy()
    if filtering:
        smooth_signals = filter_signals(smooth_signals, sig_dt, fcut_lp= 10)

    # Remove offset and initial transient
    smooth_signals = remove_signals_offset(smooth_signals)

    istart = round( float(discard_time)/float(sig_dt) )
    chopped_signals = smooth_signals[:, istart:]

    ptccs = []
    for ind, sig in enumerate(chopped_signals):

        # Auto-correlogram
        power    = np.correlate(sig, sig)
        autocorr = fftconvolve(sig, sig[::-1], 'full')

        if power > 0:
            autocorr = autocorr / power
        else:
            autocorr = np.zeros(autocorr.shape)

        siglen = len(autocorr)//2   # Computed in [-n//2,+n//2-1]

        sig_maxs = find_peaks(+autocorr[siglen:])
        first_max_ind = siglen if sig_maxs[0].size == 0 else sig_maxs[0][0] + siglen
        sig_mins = find_peaks(-autocorr[siglen:])
        first_min_ind = siglen if sig_mins[0].size == 0 else sig_mins[0][0] + siglen

        ptccs.append(autocorr[first_max_ind] - autocorr[first_min_ind])

        if plot and ind == plot_ind:
            signal_processor_snn_plot.plot_auto_correlogram(
                autocorr      = autocorr,
                sig_dt        = sig_dt,
                siglen        = siglen,
                first_max_ind = first_max_ind,
                first_min_ind = first_min_ind,
            )

    return np.array(ptccs)

def compute_effort(
        times       : np.ndarray,
        signals     : np.ndarray,
        discard_time: float = 0 * b2.second,
    ) -> np.ndarray:
    '''
    Computes the energy of the provided signals
    as the integral of the squared signals over the considered interval.
    '''

    times   = np.array(times)
    signals = np.array(signals)

    discard_time = float(discard_time)
    sig_dt       = times[1] - times[0]

    # Squared signals
    squared_signals = signals.copy()
    squared_signals = np.power(squared_signals, 2)

    # Discard time
    istart = round( float(discard_time)/float(sig_dt) )
    chopped_signals = squared_signals[:, istart:]

    # Compute energy of the signals
    considered_time = times[-1] - discard_time
    efforts = np.sum(chopped_signals, axis=1) * sig_dt / considered_time

    return np.array(efforts)

def compute_duty_cycle(
    times       : np.ndarray,
    signals     : np.ndarray,
    discard_time: float = 0 * b2.second,
    threshold   : float = 0.0,
) -> np.ndarray:
    '''
    Computes the duty cycle of the provided signals
    '''

    times     = np.array(times)
    signals   = np.array(signals)
    n_signals = signals.shape[0]
    n_diff    = n_signals // 2

    discard_time = float(discard_time)
    sig_dt       = times[1] - times[0]

    # Discard time
    istart  = round( float(discard_time)/float(sig_dt) )
    signals = signals[:, istart:]
    n_steps = signals.shape[1]

    # Signals difference
    signals_diff = signals[:n_diff] - signals[n_diff:]

    # Compute duty cycle
    diff_l = np.zeros(n_diff)
    diff_r = np.zeros(n_diff)

    for i in range(n_diff):
        diff_l[i] = np.shape( np.where( signals_diff[i] > +threshold ) )[1]
        diff_r[i] = np.shape( np.where( signals_diff[i] < -threshold ) )[1]

    duty_cycle_l = diff_l / n_steps
    duty_cycle_r = diff_r / n_steps

    return duty_cycle_l, duty_cycle_r

# \---------------[ COMPUTE METRICS ] -----------------
