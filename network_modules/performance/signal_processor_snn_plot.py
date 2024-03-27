import numpy as np
import matplotlib.pyplot as plt

def plot_threshold_crossing(
    smooth_signal: np.ndarray,
    onsets       : list[int],
    offsets      : list[int],
    strt1        : list[int],
    strt2        : list[int],
    stop1        : list[int],
    stop2        : list[int],
):
    ''' Plot crossing of the two thresholds in the noisy signal '''

    plt.plot( smooth_signal, 'tab:blue', linewidth = 0.5, label= 'Noisy signal' )

    # All crossings
    plt.plot(strt1, smooth_signal[strt1],'g.', label= 'th1 cross (up)')
    plt.plot(stop1, smooth_signal[stop1],'g.', label= 'th1 cross (down)')
    plt.plot(strt2, smooth_signal[strt2],'r.', label= 'th2 cross (up)')
    plt.plot(stop2, smooth_signal[stop2],'r.', label= 'th2 cross (down)')

    # Double threshold crossings (after pruning)
    plt.plot( onsets, smooth_signal[onsets], 'k^', markersize = 10, label = 'Onset')
    plt.plot(offsets, smooth_signal[offsets],'kv', markersize = 10, label = 'Offset')

    plt.xlabel("Time [s]")
    plt.ylabel("Signal")
    plt.title("Processing noisy signals")
    plt.legend(bbox_to_anchor=(1.0,1), loc="upper left")

    return

def plot_com_surface(
    axis       : plt.Axes,
    values     : np.ndarray,
    start      : int,
    stop       : int,
    line_slope : float,
    line_offset: float,
):
    ''' Plots the enclosing surface for a COM in the oscillatory signal '''
    axis.fill_between(
        np.arange(start,stop),
        values[start:stop],
        line_slope/1000 * np.arange(start,stop) + line_offset,
        facecolor= 'lightgrey'
    )

    return

def plot_com_signals(
    values     : np.ndarray,
    onsets     : list[int],
    offsets    : list[int],
    com_x      : list[float],
    com_y      : list[float],
):
    ''' Plots the COM positions in the oscillatory signal '''

    plt.plot( values, 'tab:blue', linewidth = 0.5, label= 'Noisy signal' )

    plt.plot( onsets, values[onsets], 'g^', markersize = 10, label = 'Onset')
    plt.plot(offsets, values[offsets],'rv', markersize = 10, label = 'Offset')
    plt.plot(com_x[:,0]*1000, com_y[:,0],'kx', markersize = 10, label = 'COM')

    plt.xlabel("Time [s]")
    plt.ylabel("Signal")
    plt.title("Processing noisy signals")
    plt.legend(bbox_to_anchor=(1.0,1), loc="upper left")

    return

def plot_fft_signals(
    times      : np.ndarray,
    signals_fft: np.ndarray,
    sig_ind    : int,
):
    ''' Plot the FFT of the signals '''
    n_step = len(times)
    sr_sig = 1 / (times[1] - times[0])
    freqs  = np.arange(n_step) * sr_sig / n_step

    plt.figure(f'FFT - Signal {sig_ind}')
    plt.stem(
        freqs,
        np.abs(signals_fft[sig_ind]),
        'b',
        markerfmt = " ",
        basefmt   = "-b"
    )
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(freqs[0], freqs[-1])

    return

def plot_auto_correlogram(
    autocorr     : np.ndarray,
    sig_dt       : float,
    siglen       : int,
    first_max_ind: int,
    first_min_ind: int,
):
    ''' Plots the auto-correlogram of the signal '''
    ## PLOTS
    plt.figure('Auto-correlogram')
    plt.grid()
    plt.title('Auto-correlogram')
    auxtime = np.arange(-siglen,siglen+1)*sig_dt
    plt.plot(auxtime, autocorr, label= 'Auto-correlogram')

    plt.plot([0,0], [-1,1], linewidth= 0.5, color= 'k')
    plt.plot( auxtime[first_max_ind], autocorr[first_max_ind], 'ro', label= 'MAX')
    plt.plot( auxtime[first_min_ind], autocorr[first_min_ind], 'bo', label= 'MIN')

    plt.xlabel('Time lag [s]')
    plt.ylabel('Crosscorrelation')
    plt.legend()

    return