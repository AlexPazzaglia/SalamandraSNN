import control
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import harold

from scipy.signal import lfilter

from sysidentpy.metrics import mean_squared_error
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation

def plot_transfer_function():

    F0  = 1
    M   = 1
    OHM = 2 * np.pi * np.array([0.1, 1, 10])

    K = np.logspace(-3, 2, 1000)
    C = np.logspace(-3, 2, 1000)

    K_GRID, C_GRID = np.meshgrid(K, C)


    for ohm in OHM:

        PHI = np.arctan( - C_GRID * ohm /  ( K_GRID - M * ohm**2 ) )
        PHI[ PHI > 0 ] = PHI[ PHI > 0 ] - np.pi

        X0  = F0 / ( ( K_GRID - M * ohm**2 ) * np.cos( PHI ) - C_GRID * ohm * np.sin( PHI ) )

        plt.figure(f'OHM = {ohm / (2 * np.pi)} Hz')

        plt.subplot(1, 2, 1)
        plt.contourf(K_GRID, C_GRID, np.clip(0, 5 * np.mean(X0), X0), levels=100)
        plt.plot(K, 2 * np.sqrt(M * K), 'r--')
        plt.xlabel('K')
        plt.ylabel('C')
        plt.xscale('log')
        plt.yscale('log')
        plt.colorbar()
        plt.title('X0')

        plt.subplot(1, 2, 2)
        plt.contourf(K_GRID, C_GRID, PHI, levels=100)
        plt.plot(K, 2 * np.sqrt(M * K), 'r--')
        plt.xlabel('K')
        plt.ylabel('C')
        plt.xscale('log')
        plt.yscale('log')
        plt.colorbar()
        plt.title('PHI')

    plt.show()

def plot_continuous_system_impulse_response(tr_fun_cont, times):
    ''' Plot the impulse response of a continuous time transfer function '''

    # Control impulse response
    t_cont, y_cont = control.impulse_response(tr_fun_cont, T=times)

    # Plot
    plt.plot(t_cont, y_cont, label= 'Continuous (control)')

    plt.title("Step Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

def get_continuous_system(zeta, fn):
    ''' Get the continuous time transfer function of a second order system'''

    wn  = 2 * np.pi * fn

    # # Continuous time transfer function
    # pole_s_1 = - zeta * wn + wn * np.sqrt(zeta**2 - 1 + 0j)
    # pole_s_2 = - zeta * wn - wn * np.sqrt(zeta**2 - 1 + 0j)

    # Numerator
    a0_s = wn**2

    # Denominator
    b0_s = 1
    b1_s = 2 * zeta * wn
    b2_s = wn**2

    return control.TransferFunction([a0_s], [b0_s, b1_s, b2_s])

def plot_discrete_system_impulse_response(tr_fun_disc, times):
    ''' Plot the impulse response of a discrete time transfer function '''

    dt  = tr_fun_disc.dt
    a_z = tr_fun_disc.den[0][0]
    b_z = tr_fun_disc.num[0][0]

    # Control impulse response
    t_disc, y_disc = control.impulse_response(tr_fun_disc, T=times)

    # Scipy filter
    t_disc    = times
    x_disc    = np.zeros_like(t_disc)
    x_disc[0] = 1

    y_filt= lfilter(b_z, a_z, x_disc) / dt

    # Plot
    plt.plot(t_disc, y_disc, label= 'Discrete (control)')
    plt.plot(t_disc, y_filt, label= 'Discrete (scipy)')

    plt.title("Step Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

def get_discrete_system(zetam, fn, fs):
    ''' convert_continuous_to_discrete '''

    # Continuous time transfer function
    tr_fun_cont = get_continuous_system(zetam, fn)

    # Discrete time transfer function
    tr_fun_disc = control.sample_system(tr_fun_cont, 1/fs, method='bilinear')

    return tr_fun_disc

def get_FROLS_model(x_train, y_train, x_valid, y_valid, plot=True, tag = ''):
    ''' Get the FROLS model '''

    # Model
    basis_function = Polynomial(degree=1)
    model = FROLS(
        extended_least_squares = True,
        ylag                   = 2,
        xlag                   = 2,
        order_selection        = True,
        info_criteria          = 'aic',
        n_info_values          = 5,
        estimator              = 'least_squares',
        basis_function         = basis_function,
        model_type             = 'NARMAX'
    )

    # Train
    model.fit(X=x_train, y=y_train)

    # Train set
    y_train_hat = model.predict(X=x_train, y=y_train)
    mse_train   = mean_squared_error(y_train, y_train_hat)
    # print(f'MSE_TRAIN: {mse_train}')

    # Test set
    y_valid_hat = model.predict(X=x_valid, y=y_valid)
    mse_valid   = mean_squared_error(y_valid, y_valid_hat)
    # print(f'MSE_TEST: {mse_valid}')

    # Results
    model_results = pd.DataFrame(
        results(
            model.final_model, model.theta, model.err,
            model.n_terms, err_precision=8, dtype='sci'
            ),
        columns=['Regressors', 'Parameters', 'ERR'])

    # print(model_results)

    if not plot:
        return model, model_results

    tag = '- ' + tag if tag else ''

    # Train set
    plot_results(
        y     = y_train,
        yhat  = y_train_hat,
        n     = y_train.shape[0],
        title = f'TRAIN SET - Model prediction {tag}'
    )

    plot_residues_correlation(
        data   = compute_residues_autocorrelation(y_train, y_train_hat),
        title  = f"TRAIN SET - Residues correlation {tag}",
        ylabel = "$e^2$"
    )

    # Test set
    plot_results(
        y     = y_valid,
        yhat  = y_valid_hat,
        n     = y_valid.shape[0],
        title = f'TEST SET - Model prediction {tag}'
    )

    plot_residues_correlation(
        data   = compute_residues_autocorrelation(y_valid, y_valid_hat),
        title  = f"TEST SET - Residues correlation {tag}",
        ylabel = "$e^2$"
    )

    return model, model_results

def get_fitting_second_order_system(
    signal_input    : np.ndarray,
    signal_response : np.ndarray,
    times           : np.ndarray,
    freq_sampling   : float,
    train_percentage: float,
    plot            : bool = True,
    tag             : str  = ''
):
    ''' Get the fitting of a second order system '''

    # Split train and validation data
    n_steps       = np.shape(times)[0]
    n_steps_train = round(n_steps * train_percentage)

    x_train = signal_input[:n_steps_train, np.newaxis]
    x_valid = signal_input[n_steps_train:, np.newaxis]
    y_train = signal_response[:n_steps_train, np.newaxis]
    y_valid = signal_response[n_steps_train:, np.newaxis]

    # FROLS
    _model_disc, results_disc = get_FROLS_model(
        x_train = x_train,
        y_train = y_train,
        x_valid = x_valid,
        y_valid = y_valid,
        plot    = plot,
        tag     = tag,
    )

    # Get discrete time transfer function of the resulting model
    a1 = results_disc[results_disc['Regressors'] == 'y(k-1)']
    a2 = results_disc[results_disc['Regressors'] == 'y(k-2)']
    b0 = results_disc[results_disc['Regressors'] == 'x1(k-1)']
    b1 = results_disc[results_disc['Regressors'] == 'x1(k-2)']

    a1 = float(a1['Parameters']) if a1.size else 0
    a2 = float(a2['Parameters']) if a2.size else 0
    b0 = float(b0['Parameters']) if b0.size else 0
    b1 = float(b1['Parameters']) if b1.size else 0

    G_disc = harold.Transfer( [b0, b1, 0], [1, -a1, -a2], dt=1/freq_sampling)

    # Convert to continuous time
    G_cont = harold.undiscretize(G_disc)

    # Remove zeros
    poles  = np.real( np.polynomial.polynomial.polyfromroots(G_cont.poles) )[::-1]
    G_cont = harold.Transfer( G_cont.num[0][-1], poles)

    return G_cont, G_disc

def main():

    ZC = 0.1
    FN   = 7
    FS   = 1000

    # Transfer function
    tr_fun_cont = get_continuous_system(ZC, FN)
    tr_fun_disc = get_discrete_system(ZC, FN, FS)

    WN = 2 * np.pi * FN
    B0 = tr_fun_cont.num[0][0][0]
    G0 = tr_fun_cont.num[0][0][0] / tr_fun_cont.den[0][0][2]

    M  = 1 / B0
    K  = 1 / G0
    C  = 2 * ZC * np.sqrt(M * K)

    plot = True

    # Plot impulse response
    if plot:
        plt.figure('Impulse response')
        times = np.arange(0, 10, 1/FS)
        plot_continuous_system_impulse_response(tr_fun_cont, times)
        plot_discrete_system_impulse_response(tr_fun_disc, times)

    # Forced response (sweep frequency)
    times = np.arange(0, 10, 1/FS)

    signal_amp      = 1
    signal_freq_min = 0.0
    signal_freq_max = 5

    frequency_sweep = signal_freq_min + (signal_freq_max - signal_freq_min) * times / (times[-1] - times[0])

    signal_vals     = signal_amp * np.cos(2 * np.pi * frequency_sweep * times)

    _time_cont, signal_response_cont = control.forced_response(tr_fun_cont, T=times, U=signal_vals)
    _time_disc, signal_response_disc = control.forced_response(tr_fun_disc, T=times, U=signal_vals)

    # Plot forced response
    plt.figure('Forced response')
    plt.plot(_time_cont, signal_response_cont, label= 'Continuous')
    plt.plot(_time_disc, signal_response_disc, label= 'Discrete')
    plt.title("Forced Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    # Fitting
    G_cont, G_disc = get_fitting_second_order_system(
        signal_input    = signal_vals,
        signal_response = signal_response_cont,
        times           = times,
        freq_sampling   = FS,
        train_percentage= 0.8,
        plot            = True,
    )

    # Compare to initial values
    num = G_cont.num[0] / G_cont.den[0][0]
    den = G_cont.den[0] / G_cont.den[0][0]

    B0_hat = num[0]
    G0_hat = num[0] / den[2]
    WN_hat = np.sqrt( den[2] )
    ZC_hat = den[1] / (2 * WN_hat)

    M_hat = 1 / num[0]
    C_hat = den[1] / num[0]
    K_hat = den[2] / num[0]

    # print('')
    # print(f'b0: {B0 :.3f}, b0_hat: {B0_hat :.3f} -> Error: {np.abs(B0 - B0_hat) / B0 * 100 :.3f} %')
    # print(f'G0: {G0 :.3f}, G0_hat: {G0_hat :.3f} -> Error: {np.abs(G0 - G0_hat) / G0 * 100 :.3f} %')
    # print(f'Wn: {WN :.3f}, Wn_hat: {WN_hat :.3f} -> Error: {np.abs(WN - WN_hat) / WN * 100 :.3f} %')
    # print(f'Zc: {ZC :.3f}, Zc_hat: {ZC_hat :.3f} -> Error: {np.abs(ZC - ZC_hat) / ZC * 100 :.3f} %')

    # print(f'M:  {M :.3f},  M_hat: {M_hat :.3f}  -> Error: {np.abs(M - M_hat) / M * 100 :.3f} %')
    # print(f'K:  {K :.3f},  K_hat: {K_hat :.3f}  -> Error: {np.abs(K - K_hat) / K * 100 :.3f} %')
    # print(f'C:  {C :.3f},  C_hat: {C_hat :.3f}  -> Error: {np.abs(C - C_hat) / C * 100 :.3f} %')

    # Plot impulse response
    times = np.arange(0, 10, 1/FS)
    y_d, t_d = harold.simulate_impulse_response(G_disc, times)
    y_c, t_c = harold.simulate_impulse_response(G_cont, times)

    plt.figure('Impulse response of the resulting model')

    plt.plot(t_d, y_d, label= 'Discrete')
    plt.plot(t_c, y_c, label= 'Continuous')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()


    plt.show()



if __name__ == "__main__":
    main()