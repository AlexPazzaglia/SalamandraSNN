"""Analyze results of the mechanical simulation """

import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Any

from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from farms_amphibious.data.data import SpikingData, AmphibiousKinematicsData
from farms_core.io.yaml import yaml2pyobject

### -------- [ DATA LOADING ] --------
def load_mechanical_simulation_data(
    data_folder : str,
    control_type: str,
) -> tuple[ Union[AmphibiousKinematicsData, SpikingData], dict[str, Any] ]:
    ''' Load mechanical simulation data '''
    data = (
        AmphibiousKinematicsData.from_file(f'{data_folder}/simulation.hdf5')
        if control_type in ['position_control', 'hybrid_position_control']
        else
        SpikingData.from_file(f'{data_folder}/simulation.hdf5')
    )
    parameters : dict[str, Any] = yaml2pyobject(f'{data_folder}/animat_options.yaml')
    return data, parameters

### \-------- [ DATA LOADING ] --------

### -------- [ UTILS ] --------

def _compute_signals_frequencies_fft(
    signal_2D : np.ndarray,
    timestep : float,
    max_freq = 50
):
    ''' Calculate the frequency using the dominant frequency component of the signal's FFT '''

    max_freq_ind      = round(max_freq * signal_2D.shape[0] * timestep)
    fft_results       = np.fft.fft(signal_2D, axis=0)
    fft_dominant_inds = np.argmax(np.abs(fft_results[1:max_freq_ind, :]), axis=0) + 1

    frequencies = fft_dominant_inds / (signal_2D.shape[0] * timestep)

    return frequencies

def _compute_signals_amplitudes_fft(
    signal_2D : np.ndarray
):
    ''' Calculate the amplitude using the dominant frequency component of the signal's FFT '''
    num_signals = signal_2D.shape[1]

    fft_results       = np.fft.fft(signal_2D, axis=0)
    fft_dominant_inds = np.argmax(np.abs(fft_results[1:, :]), axis=0) + 1

    amplitudes = 2 * np.abs(fft_results[fft_dominant_inds, np.arange(num_signals)]) / signal_2D.shape[0]

    return amplitudes

def _compute_signals_delays(
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

    for ind_couple, (ind1, ind2) in enumerate(inds_couples):
        sig1 = signals[:, ind1] - np.mean(signals[:, ind1])
        sig2 = signals[:, ind2] - np.mean(signals[:, ind2])

        xcorr = np.correlate(sig2, sig1, "full")
        n_lag = np.argmax(xcorr) - len(xcorr) // 2
        t_lag = n_lag * dt_sig

        ipl = t_lag * np.mean( [freqs[ind1], freqs[ind2]] )

        if ipl > 0.5:
            ipl = ipl - 1
        if ipl < -0.5:
            ipl = ipl + 1

        ipls[ind_couple] = ipl

    return ipls

def _compute_body_linear_fit_step(
    coordinates_xy: np.ndarray,
    n_links_pca   : int,
    step          : int,
) -> tuple[np.ndarray, np.ndarray]:
    ''' Compute the PCA of the links positions at a given step '''

    cov_mat = np.cov(
        [
            coordinates_xy[step, :n_links_pca, 0],
            coordinates_xy[step, :n_links_pca, 1],
        ]
    )
    eig_values, eig_vecs = np.linalg.eig(cov_mat)
    largest_index        = np.argmax(eig_values)
    direction_fwd        = eig_vecs[:, largest_index]

    # Align the direction with the tail-head axis
    p_tail2head    = coordinates_xy[step, 0] - coordinates_xy[step, n_links_pca-1]
    direction_sign = np.sign( np.dot( p_tail2head, direction_fwd ) )
    direction_fwd  = direction_sign * direction_fwd

    direction_left = np.cross(
        [0,0,1],
        [direction_fwd[0], direction_fwd[1], 0]
    )[:2]

    return direction_fwd, direction_left

def _compute_body_linear_fit_all(
    coordinates_xy: np.ndarray,
    n_links_pca   : int = None,
) -> tuple[np.ndarray, np.ndarray]:
    ''' Compute the PCA of the links positions at all steps '''

    flattened_coordinates = coordinates_xy[:, :n_links_pca].reshape(-1, 2)
    cov_mat = np.cov(
        [
            flattened_coordinates[:, 0],
            flattened_coordinates[:, 1],
        ]
    )

    eig_values, eig_vecs = np.linalg.eig(cov_mat)
    largest_index        = np.argmax(eig_values)
    direction_fwd        = eig_vecs[:, largest_index]

    # Align the direction with the start-finish axis
    p_start2end    = coordinates_xy[-1, 0] - coordinates_xy[0, 0]
    direction_sign = np.sign( np.dot( p_start2end, direction_fwd ) )
    direction_fwd  = direction_sign * direction_fwd

    direction_left = np.cross(
        [0,0,1],
        [direction_fwd[0], direction_fwd[1], 0]
    )[:2]

    return direction_fwd, direction_left

def _transform_coordinates_to_body_frame(
    coordinates_xy: np.ndarray,
    n_links_pca   : int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ''' Transform coordinates to the body frame, defined by the PCA of the links positions '''

    direction_fwd, direction_left = _compute_body_linear_fit_all(
        coordinates_xy = coordinates_xy,
        n_links_pca    = n_links_pca,
    )

    if np.all( direction_fwd == [0,0] ) and np.all( direction_left == [0,0] ):
        # No transformation possible, keep the original coordinates
        return coordinates_xy, np.array([1,0]), np.array([0,1])

    coordinates_transformed = np.zeros_like(coordinates_xy)
    for step in range(coordinates_xy.shape[0]):
        coordinates_transformed[step, :, 0] = np.dot(coordinates_xy[step], direction_fwd)
        coordinates_transformed[step, :, 1] = np.dot(coordinates_xy[step], direction_left)

    return coordinates_transformed, direction_fwd, direction_left

def _transform_coordinates_to_global_frame(
    coordinates_xy: np.ndarray,
    direction_x   : np.ndarray,
    direction_y   : np.ndarray,
) -> np.ndarray:
    ''' Transform coordinates to the global frame '''

    if np.cross(direction_x, direction_y) < 0:
        direction_y = -direction_y

    rotation_matrix         = np.array([direction_x, direction_y])
    transformed_coordinates = np.dot(coordinates_xy, rotation_matrix)

    return transformed_coordinates

def _compute_body_quadratic_fit_all(
    coordinates_xy: np.ndarray[float],
):
    # Convert to body axis coordinates
    (
        coordinates_transformed,
        direction_fwd,
        direction_left,
    ) = _transform_coordinates_to_body_frame(coordinates_xy)

    # Polynomial fitting in body axis coordinates
    flattened_coordinates = coordinates_transformed.reshape(-1, 2)

    x = flattened_coordinates[:, 0]
    y = flattened_coordinates[:, 1]

    coefficients    = np.polyfit(x, y, 2)
    polynomial_func = np.poly1d(coefficients)

    # Compute the fitted coordinates in body axis coordinates
    x_fit_transformed = np.linspace(
        coordinates_transformed[:, :, 0].min(),
        coordinates_transformed[:, :, 0].max(),
        100
    )
    y_fit_transformed = polynomial_func(x_fit_transformed)

    coordinates_fit_transformed = np.array(
        [
            x_fit_transformed,
            y_fit_transformed,
        ]
    ).T

    # Convert back to global coordinates
    coordinates_fit = _transform_coordinates_to_global_frame(
        coordinates_fit_transformed,
        direction_fwd,
        direction_left,
    )

    x_fit = coordinates_fit[:, 0]
    y_fit = coordinates_fit[:, 1]

    return x_fit, y_fit

def _get_distance_to_quadratic(
    point       : np.ndarray[float],
    coefficients: np.ndarray[float],
) -> float:
    ''' Find the point on a parabola where the distance to a given point is minimized '''

    # Find roots of the derivative of the squared distance between point and parabola
    x0, y0  = point
    a, b, c = coefficients

    squared_distance_derivative_coeff = [
        2 * a**2,
        3 * a * b,
        1 + b**2 + 2 * a * (c-y0),
        b * (c-y0) - x0,
    ]

    x_roots = np.roots(squared_distance_derivative_coeff)

    # Find the minimum distance
    min_distance = np.inf
    f_y0         = np.polyval(coefficients, x0)
    for x_root in x_roots:

        if not np.isreal(x_root):
            continue

        x_root    = np.real(x_root)
        y_root   = np.polyval(coefficients, x_root)
        distance = np.sqrt((x_root - point[0])**2 + (y_root - point[1])**2)

        if distance < min_distance:
            min_distance = distance * np.sign(f_y0 - y_root)

    return min_distance

def _compute_links_displacements(
    coordinates_xy: np.ndarray[float],
    n_links_pca   : int = None,
) -> dict[str, np.ndarray[float]]:
    ''' Compute the displacement of the links from the fitted parabola'''

    # Convert to body axis coordinates
    (
        coordinates_xy_transformed,
        direction_fwd,
        direction_left,
    ) = _transform_coordinates_to_body_frame(
        coordinates_xy = coordinates_xy,
        n_links_pca    = n_links_pca,
    )

    # Polynomial fitting in body axis coordinates
    flattened_coordinates = coordinates_xy_transformed.reshape(-1, 2)

    quadratic_fit_coefficients = np.polyfit(
        flattened_coordinates[:, 0],
        flattened_coordinates[:, 1],
        2
    )

    # Find the minimum distance to the fitted parabola
    links_displacements = np.array(
        [
            _get_distance_to_quadratic(point, quadratic_fit_coefficients)
            for point in flattened_coordinates
        ]
    )

    links_displacements = links_displacements.reshape(coordinates_xy.shape[0], -1)

    return {
        'links_displacements'        : links_displacements,
        'links_positions'            : coordinates_xy,
        'links_positions_transformed': coordinates_xy_transformed,
        'direction_fwd'              : direction_fwd,
        'direction_left'             : direction_left,
        'quadratic_fit_coefficients' : quadratic_fit_coefficients,
    }

def _compute_tail_positions_from_links_positions(
    links_positions : np.ndarray,
    joints_positions: np.ndarray,
    length_tail_link: float,
    n_links_axis    : int,
) -> np.ndarray:
    ''' Compute the tail positions '''

    # Compute the direction of the second last link
    v_last     = links_positions[:, n_links_axis-1, :] - links_positions[:, n_links_axis-2, :]
    v_last     = v_last / np.linalg.norm(v_last, axis=1).reshape(-1, 1)
    angle_last = joints_positions[:, n_links_axis-2]

    # Rotate the last link
    v_tail = np.array(
        [
            np.cos(angle_last) * v_last[:, 0] - np.sin(angle_last) * v_last[:, 1],
            np.sin(angle_last) * v_last[:, 0] + np.cos(angle_last) * v_last[:, 1],
            v_last[:, 2],
        ]
    ).T

    # Compute the tail positions
    tail_positions = links_positions[:, n_links_axis-1] + length_tail_link * v_tail

    return tail_positions

### -------- [ UTILS ] --------

### -------- [ METRICS COMPUTATION ] --------

def compute_joints_displacements_metrics(
    joints_positions: np.ndarray,
    n_joints_axis   : int,
    sim_fraction    : float= 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Computes evolution of the displacements of the joints'''

    n_steps = joints_positions.shape[0]
    n_steps_considered = round(n_steps * sim_fraction)

    joints_positions = joints_positions[-n_steps_considered:, :n_joints_axis]

    # Compute metrics
    joints_displacements_mean = np.mean(joints_positions, axis= 0)
    joints_displacements_std  = np.std(joints_positions, axis= 0)
    joints_displacements_amp  = _compute_signals_amplitudes_fft(joints_positions)

    return (
        joints_positions,
        joints_displacements_mean,
        joints_displacements_std,
        joints_displacements_amp,
    )

def compute_links_displacements_metrics(
    points_positions: np.ndarray,
    n_points_axis   : int,
    sim_fraction    : float = 1.0,
    sample_fraction : float= 0.1,
    n_points_pca    : int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    '''Computes evolution of the displacements of the links'''

    n_points_pca  = n_points_axis if n_points_pca is None else n_points_pca

    n_steps            = points_positions.shape[0]
    n_steps_considered = round(n_steps * sim_fraction)
    n_steps_skip       = round(1/sample_fraction)

    # Compute links displacements
    points_positions = points_positions[-n_steps_considered::n_steps_skip, :n_points_axis, :2]

    links_displacements_data = _compute_links_displacements(
        coordinates_xy = points_positions,
        n_links_pca    = n_points_pca,
    )
    links_displacements = links_displacements_data['links_displacements']

    # # Compute CoM and PCA
    # links_displacements = np.zeros((n_steps_considered, n_links_axis))
    # vect_com            = np.mean(links_positions[:, :n_links_pca], axis= 1)
    # vect_fwd_unitary    = np.zeros((n_steps_considered, 2))
    # vect_lat_unitary    = np.zeros((n_steps_considered, 2))

    # for ind in range(n_steps_considered):

    #     # Compute the PCA of the links positions
    #     vect_fwd_unitary[ind], vect_lat_unitary[ind] = _compute_body_linear_fit_step(
    #         coordinates_xy = links_positions,
    #         n_links_pca  = n_links_pca,
    #         step         = ind,
    #     )

    #     # Projection along the lateral direction
    #     vect_com2link            = links_positions[ind] - vect_com[ind]
    #     links_displacements[ind] = np.sum( vect_com2link * vect_lat_unitary[ind], axis= 1 )

    # Compute metrics
    links_displacements_mean = np.mean(links_displacements, axis= 0)
    links_displacements_std  = np.std(links_displacements, axis= 0)
    links_displacements_amp  = _compute_signals_amplitudes_fft(links_displacements)

    return (
        links_displacements,
        links_displacements_mean,
        links_displacements_std,
        links_displacements_amp,
        links_displacements_data,
    )

def compute_trajectory_linearity(
    timestep     : float,
    com_positions: np.ndarray,
    sim_fraction = 1.0
):
    ''' Compute the linearity of the trajectory '''

    n_dim               = com_positions.shape[1]
    n_steps             = com_positions.shape[0]
    n_steps_considered  = round(n_steps * sim_fraction)
    duration_considered = n_steps_considered * timestep

    times = np.arange(0, duration_considered, timestep).reshape(-1, 1)

    com_mse = np.zeros(n_dim)
    for dim in range(n_dim):

        dim_positions = com_positions[-n_steps_considered:, dim ]
        com_linear_model = LinearRegression()
        com_linear_model.fit(
            times,
            dim_positions,
        )

        dim_predict  = com_linear_model.predict(times)
        com_mse[dim] = mean_squared_error(dim_positions, dim_predict)

    return np.sum(com_mse)

def compute_trajectory_curvature(
    timestep     : float,
    com_pos      : np.ndarray,
    sim_fraction : float = 1.0,
):
    ''' Compute the curvature of the trajectory '''

    n_elem            = com_pos.shape[0]
    n_elem_considered = round(n_elem * sim_fraction)

    com_vel           = np.diff(com_pos[-n_elem_considered:], axis=0)

    # Get the left vector
    com_vel_l = np.zeros_like(com_vel)
    for ind in range(n_elem_considered - 1):
        left_vect   = np.cross([0, 0, 1], com_vel[ind, :])
        left_vect_n = np.linalg.norm(left_vect)

        if left_vect_n == 0:
            continue

        com_vel_l[ind] = left_vect / left_vect_n

    # Get the left velocity
    vel_sign  = np.zeros(n_elem_considered - 2)
    for ind in range(n_elem_considered - 2):
        vel_sign[ind]  = np.sign( np.sum(com_vel_l[ind] * com_vel[ind+1]) )

    # Get the curvature
    dx_dt    = np.gradient(com_pos[-n_elem_considered:, 0])
    dy_dt    = np.gradient(com_pos[-n_elem_considered:, 1])
    velocity = np.array( [ dx_dt, dy_dt] ).transpose()

    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

    tangent   = np.array([1/ds_dt] * 2).transpose() * velocity
    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]

    deriv_tangent_x = np.gradient(tangent_x)
    deriv_tangent_y = np.gradient(tangent_y)

    dT_dt = np.array(
        [ [deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size) ]
    )

    length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)
    length_dT_dt[length_dT_dt == 0] = np.nan

    normal = np.array([1/length_dT_dt] * 2).transpose() * dT_dt
    normal[np.isnan(normal)] = 0

    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5

    # Signed curvature
    signed_curvature            = np.zeros_like(curvature)
    signed_curvature[:n_elem_considered - 2] = curvature[:n_elem_considered - 2] * vel_sign

    # Filter
    forder = 5
    fnyq   = 0.5 / timestep
    fcut   = 5.0

    num, den               = butter(forder, fcut/fnyq, btype= 'lowpass' )
    signed_curvature_filt  = filtfilt(num, den, signed_curvature.T).T

    # import matplotlib.pyplot as plt
    # plt.figure('trajectory')
    # plt.plot(com_pos[:,0], com_pos[:,1])
    # plt.grid()

    # plt.figure('curvature')
    # plt.plot(signed_curvature)
    # plt.plot(signed_curvature_filt)
    # plt.xlim(0, n_elem_considered)
    # plt.grid()

    return np.mean(signed_curvature)

def compute_frequency(
    joints_positions: np.ndarray,
    n_joints_axis   : int,
    n_joints_limbs  : int,
    timestep        : float,
    sim_fraction    : float = 1.0,
):
    '''
    Computes the frequency of the joints positions
    '''

    n_steps            = joints_positions.shape[0]
    n_steps_considered = round(n_steps * sim_fraction)
    joints_positions  = joints_positions[-n_steps_considered:]

    # Compute from FFT
    freq_joints  = _compute_signals_frequencies_fft(joints_positions, timestep)

    freq_ax = (
        np.mean(freq_joints[:n_joints_axis])
        if n_joints_axis else np.nan
    )
    freq_lb = (
        np.mean(freq_joints[n_joints_axis:n_joints_axis+n_joints_limbs])
        if n_joints_limbs else np.nan
    )
    freq_diff = np.abs(freq_ax - freq_lb)

    return freq_ax, freq_lb, freq_diff, freq_joints

def compute_joints_ipls(
    joints_angles    : np.ndarray,
    joints_freqs     : np.ndarray,
    n_joints_axis    : int,
    n_joints_per_limb: int,
    n_active_joints  : int,
    limb_pairs_inds  : np.ndarray,
    timestep         : float,
):
    '''
    Computes the IPLS of oscillations if not already computed
    For swiming, consider the entire network.
    For walking, consider the phase jump at girdles.
    '''

    n_steps       = joints_angles.shape[0]
    times         = np.arange(n_steps) * timestep
    joints_angles = joints_angles[:, :n_active_joints]

    n_joints_limbs = n_active_joints - n_joints_axis

    n_limbs        = n_joints_limbs // n_joints_per_limb if n_joints_per_limb else 0
    n_limbs_pairs  = n_limbs // 2

    n_joints_trunk = (
        limb_pairs_inds[1] - limb_pairs_inds[0]
        if limb_pairs_inds.size
        else
        n_joints_axis
    )

    # All Axial
    axial_inds_a0  = np.array(
        [ [ joint, joint + 1 ] for joint in range(n_joints_axis-1)]
    )

    # Trunk Axial
    axial_inds_t0  = np.array(
        [ [ joint, joint + 1 ] for joint in range(n_joints_trunk-2 ) ]
    )

    # Homolateral (left-left)
    limbs_inds_h0 =  np.array(
        [
            [
                n_joints_axis + 2 * n_joints_per_limb * lb_pair,
                n_joints_axis + 2 * n_joints_per_limb * (lb_pair+1)
            ]
            for lb_pair in range( n_limbs_pairs-1 )
        ] if n_joints_per_limb else []
    )
    # Homolateral (right-right)
    limbs_inds_h1 = np.array(
        [
            [
                n_joints_axis + 2 * n_joints_per_limb * lb_pair     + n_joints_per_limb,
                n_joints_axis + 2 * n_joints_per_limb * (lb_pair+1) + n_joints_per_limb
            ]
            for lb_pair in range( n_limbs_pairs-1 )
        ] if n_joints_per_limb else []
    )
    # Contralateral (left-right)
    limbs_inds_c0 =  np.array(
        [
            [
                n_joints_axis + 2 * n_joints_per_limb * lb_pair,
                n_joints_axis + 2 * n_joints_per_limb * lb_pair + n_joints_per_limb
            ]
            for lb_pair in range( n_limbs_pairs-1 )
        ] if n_joints_per_limb else []
    )
    # Contralateral (right-left)
    limbs_inds_c1 = np.array(
        [
            [
                n_joints_axis + 2 * n_joints_per_limb * lb_pair + n_joints_per_limb,
                n_joints_axis + 2 * n_joints_per_limb * lb_pair
            ]
            for lb_pair in range( n_limbs_pairs-1 )
        ] if n_joints_per_limb else []
    )
    # Diagonal (left-right)
    limbs_inds_d0 =  np.array(
        [
            [
                n_joints_axis + 2 * n_joints_per_limb * lb_pair,
                n_joints_axis + 2 * n_joints_per_limb * (lb_pair+1) + n_joints_per_limb
            ]
            for lb_pair in range( n_limbs_pairs-1 )
        ] if n_joints_per_limb else []
    )
    # Diagonal (right-left)
    limbs_inds_d1 =  np.array(
        [
            [
                n_joints_axis + 2 * n_joints_per_limb * lb_pair + n_joints_per_limb,
                n_joints_axis + 2 * n_joints_per_limb * (lb_pair+1)
            ]
            for lb_pair in range( n_limbs_pairs-1 )
        ] if n_joints_per_limb else []
    )

    get_ipls = lambda inds : _compute_signals_delays(times, joints_angles, joints_freqs, inds)

    ipls_all = {}
    ipls_all['ipls_ax_a0'] = get_ipls(axial_inds_a0)
    ipls_all['ipls_ax_t0'] = get_ipls(axial_inds_t0)
    ipls_all['ipls_lb_h0'] = get_ipls(limbs_inds_h0)
    ipls_all['ipls_lb_h1'] = get_ipls(limbs_inds_h1)
    ipls_all['ipls_lb_c0'] = get_ipls(limbs_inds_c0)
    ipls_all['ipls_lb_c1'] = get_ipls(limbs_inds_c1)
    ipls_all['ipls_lb_d0'] = get_ipls(limbs_inds_d0)
    ipls_all['ipls_lb_d1'] = get_ipls(limbs_inds_d1)

    ipl_ax_a = np.mean( ipls_all['ipls_ax_a0'] ) if n_joints_axis else np.nan
    ipl_ax_t = np.mean( ipls_all['ipls_ax_t0'] ) if n_joints_trunk else np.nan
    ipl_lb_h = np.mean( [ ipls_all['ipls_lb_h0'], ipls_all['ipls_lb_h1'] ] ) if n_joints_per_limb else np.nan
    ipl_lb_c = np.mean( [ ipls_all['ipls_lb_c0'], ipls_all['ipls_lb_c1'] ] ) if n_joints_per_limb else np.nan
    ipl_lb_d = np.mean( [ ipls_all['ipls_lb_d0'], ipls_all['ipls_lb_d1'] ] ) if n_joints_per_limb else np.nan

    return ipl_ax_a, ipl_ax_t, ipl_lb_h, ipl_lb_c, ipl_lb_d, ipls_all

def compute_joints_neuro_muscolar_ipls(
    joints_commands: np.ndarray,
    joints_angles  : np.ndarray,
    joints_freqs   : np.ndarray,
    n_joints_axis  : int,
    n_active_joints: int,
    limb_pairs_inds: np.ndarray,
    timestep       : float,
):
    '''
    Computes the delay between the joint commands and the joint angles
    '''

    if joints_commands is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    n_steps         = joints_angles.shape[0]
    times           = np.arange(n_steps) * timestep
    joints_angles   = joints_angles[:, :n_active_joints]
    joints_commands = joints_commands[:, :2*n_active_joints]

    n_joints_trunk = limb_pairs_inds[1] - limb_pairs_inds[0] if limb_pairs_inds.size else n_joints_axis

    # Consider left-right joint commands
    joints_commands = joints_commands[:, 1::2] - joints_commands[:, 0::2]

    if joints_commands.shape[0] == joints_angles.shape[0] + 1:
        joints_commands = joints_commands[1:]

    # Signals and indices
    signals = np.concatenate( [ joints_commands, joints_angles ], axis= 1)
    freqs   = np.concatenate( [ joints_freqs, joints_freqs ], axis= 0)
    indices = np.array( [ [ joint, joint + n_active_joints ] for joint in range(n_active_joints)])

    # Compute IPLS
    lags_all = _compute_signals_delays(times, signals, freqs, indices)

    # Delay must be positive
    lags_all[lags_all < 0] = lags_all[lags_all < 0] + 1

    ok_axis  = n_joints_axis   != 0
    ok_trunk = n_joints_trunk  != n_joints_axis
    ok_limbs = n_active_joints != n_joints_axis

    lag_ax_all = np.mean( lags_all[:n_joints_axis] )               if ok_axis  else np.nan
    lag_ax_trk = np.mean( lags_all[:n_joints_trunk] )              if ok_trunk else np.nan
    lag_ax_tal = np.mean( lags_all[n_joints_trunk:n_joints_axis] ) if ok_trunk else np.nan
    lag_lb_all = np.mean( lags_all[n_joints_axis:] )               if ok_limbs else np.nan

    return lag_ax_all, lag_ax_trk, lag_ax_tal, lag_lb_all, lags_all

def compute_speed_pca(
    links_positions : np.ndarray,
    links_velocities: np.ndarray,
    n_links_axis    : int,
    sim_fraction    : float = 1.0,
) -> tuple[float, float, float]:
    '''
    Computes the axial and lateral speed based on the PCA of the links positions
    '''

    n_steps            = links_positions.shape[0]
    n_steps_considered = round(n_steps * sim_fraction)

    links_pos_xy = links_positions[-n_steps_considered:,:,:2]
    links_vel_xy = links_velocities[-n_steps_considered:,:,:2]
    time_idx     = links_pos_xy.shape[0]

    speed_forward = []
    speed_lateral = []

    for idx in range(time_idx):

        # Compute the PCA of the links positions
        direction_fwd, direction_left = _compute_body_linear_fit_step(
            coordinates_xy = links_pos_xy,
            n_links_pca  = n_links_axis,
            step         = idx,
        )

        # Compute the forward and lateral speed
        vcom_xy = np.mean( links_vel_xy[idx, :n_links_axis, :], axis=0)

        v_com_forward_proj = np.dot(vcom_xy, direction_fwd)
        v_com_lateral_proj = np.dot(vcom_xy, direction_left)

        speed_forward.append(v_com_forward_proj)
        speed_lateral.append(v_com_lateral_proj)

    # Forward and Lateral speed
    speed_fwd = np.mean(speed_forward)
    speed_lat = np.mean(speed_lateral)

    # Absolute speed
    com_vel_xy = np.mean(links_vel_xy[:, :n_links_axis, :], axis=1)
    speed_abs  = np.sum(
        np.sqrt(
            np.sum(com_vel_xy**2, axis= 1)
        )
    ) / n_steps_considered

    return speed_fwd, speed_lat, speed_abs

def compute_speed(
    links_positions_pca: np.ndarray,
    n_links_axis       : int,
    duration           : float,
    sim_fraction       : float = 1.0,
) -> tuple[float, float, float]:
    '''
    Computes the axial and lateral speed based on the PCA of the links positions
    '''

    n_steps             = links_positions_pca.shape[0]
    n_steps_considered  = round(n_steps * sim_fraction)
    duration_considered = duration * sim_fraction

    links_positions_pca_com = np.mean(
        links_positions_pca[-n_steps_considered:, :n_links_axis],
        axis= 1
    )

    speed_com_average = (
        links_positions_pca_com[-1] -
        links_positions_pca_com[0]
    ) / duration_considered

    # Forward and Lateral speed
    speed_fwd = speed_com_average[0]
    speed_lat = speed_com_average[1]

    # Absolute speed
    links_velocities_pca_com = np.diff(links_positions_pca_com, axis=0)
    speed_abs = np.sum(
        np.sqrt(
            np.sum( links_velocities_pca_com**2, axis=1 )
        )
    ) / duration_considered

    return speed_fwd, speed_lat, speed_abs

def travel_distance(links_data, sim_fraction=1.0):
    """
    Compute total travel distance, regardless of its curvature.
    Considers the distance covered by the center of mass of the model.
    """
    n_steps = links_data.shape[0]
    n_steps_considered = round(n_steps * sim_fraction)

    com = np.mean( links_data[-n_steps_considered:], axis= 1)

    # Downsample (filter oscillations)
    inds = np.arange(0, n_steps_considered, 100, dtype=int)
    comx = com[inds,0]
    comy = com[inds,1]

    return np.sum( np.sqrt( np.diff( comx )**2 + np.diff( comy )**2 ) )

def sum_torques(joints_torques, sim_fraction=1.0):
    """
    Compute sum of all the exerted active torques
    Summmation in time and across joints
    """
    n_steps = joints_torques.shape[0]
    n_steps_considered = round(n_steps * sim_fraction)

    return np.sum(np.abs(joints_torques[-n_steps_considered:, :]))

def sum_energy(torques, speeds, timestep, sim_fraction=1.0):
    """
    Compute sum of energy consumptions.
    Summation in time and across joints
    """
    n_steps = torques.shape[0]
    n_steps_considered = round(n_steps * sim_fraction)

    #NOTE: Only take positive values (no energy storing of the active part)
    powers = np.clip( torques * speeds, a_min=0, a_max=None)
    return np.sum( powers[-n_steps_considered:] ) * timestep

### \------- [ METRICS COMPUTATION ] --------
