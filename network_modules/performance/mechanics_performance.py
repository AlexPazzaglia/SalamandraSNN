import logging
import dill
import json
import os

import numpy as np

from network_modules.simulation.mechanical_simulation import MechSimulation

import network_modules.performance.signal_processor_mech as MechProc

MECH_METRICS = [

    # Frequency
    'mech_freq_ax',
    'mech_freq_lb',
    'mech_freq_diff',

    # Speed
    'mech_speed_fwd',
    'mech_speed_lat',
    'mech_speed_abs',
    'mech_stride_len',

    # Phase lag
    'mech_ipl_ax_a',
    'mech_ipl_ax_t',
    'mech_wave_number_a',
    'mech_wave_number_t',
    'mech_ipl_lb_h',
    'mech_ipl_lb_c',
    'mech_ipl_lb_d',

    # Neuro-muscolar phase lag
    'mech_n2m_lag_ax_all',
    'mech_n2m_lag_ax_trk',
    'mech_n2m_lag_ax_tal',
    'mech_n2m_lag_lb_all',

    # Torque and energy
    'mech_torque',
    'mech_energy',
    'mech_cot',

    # Trajectory
    'mech_traj_mse',
    'mech_traj_curv',

    # Tail beat
    'mech_tail_beat_amp',

    # Dimensionless numbers
    'mech_strouhal_number',
    'mech_swimming_number',
    'mech_reynolds_number',

    # Normalized metrics
    'mech_speed_fwd_bl',
    'mech_speed_lat_bl',
    'mech_speed_abs_bl',
    'mech_stride_len_bl',
    'mech_tail_beat_amp_bl',

    # Displacements
    'mech_joints_disp_mean',
    'mech_joints_disp_amp',
    'mech_links_disp_mean',
    'mech_links_disp_amp',

    # Displacements normalized
    'mech_links_disp_amp_bl',
]

MECH_METRICS_VECT_JOINTS = [
    'mech_joints_disp_mean',
    'mech_joints_disp_amp',
]

MECH_METRICS_VECT_LINKS = [
    'mech_links_disp_mean',
    'mech_links_disp_amp',
    'mech_links_disp_amp_bl',
]

class MechPerformance(MechSimulation):
    '''
    Class used to evaluate the performance of a mechanical simulation
    '''

    ## POST PROCESSING
    def simulation_post_processing(self) -> dict[str, float]:
        ''' Post-processing for the mechanical simulation '''
        super().simulation_post_processing()
        return self.farms_performance_metrics()

    # DATA LOADING
    def load_mechanical_simulation_data(self, from_file = False):
        ''' Load mechanical simulation data '''

        # NOTE:
        # For the links arrays: positions[iteration, link_id, xyz]
        # For the positions arrays: positions[iteration, xyz]
        # For the joints arrays: positions[iteration, joint]

        if not from_file:
            self.sim_res_data = self.animat_data
            self.sim_res_pars = self.animat_options
            return

        self.sim_res_data, self.sim_res_pars = MechProc.load_mechanical_simulation_data(
            data_folder  = self.results_data_folder,
            control_type = self.snn_network.control_type,
        )

    def load_mechanical_simulation_arrays(self):
        ''' Load mechanical simulation arrays '''

        data       = self.sim_res_data
        n_ax_links = self.snn_network.params.mechanics.mech_axial_joints + 1

        joints_positions      = np.array( data.sensors.joints.positions_all() )
        joints_velocities     = np.array( data.sensors.joints.velocities_all() )
        joints_active_torques = np.array( data.sensors.joints.active_torques() )
        links_positions       = np.array( data.sensors.links.urdf_positions() )
        links_velocities      = np.array( data.sensors.links.com_lin_velocities() )

        joints_commands = (
            None
            if self.snn_network.control_type in ['position_control', 'hybrid_position_control']
            else
            np.array( data.state.array )
        )

        # Take care of last iteration
        if not np.any(joints_positions[-1]):
            joints_positions  = joints_positions[:-1]

        if not np.any(joints_velocities[-1]):
            joints_velocities = joints_velocities[:-1]

        if not np.any(joints_active_torques[-1]):
            joints_active_torques = joints_active_torques[:-1]

        if not np.any(links_positions[-1]):
            links_positions = links_positions[:-1]

        if not np.any(links_velocities[-1]):
            links_velocities = links_velocities[:-1]

        if joints_commands is not None and not np.any(joints_commands[-1]):
            joints_commands = joints_commands[:-1]

        n_steps = min(
            [
                len(state_arr) for state_arr in [
                    joints_positions,
                    joints_velocities,
                    joints_active_torques,
                    links_positions,
                    links_velocities,
                    joints_commands,
                ]
                if state_arr is not None
            ]
        )

        joints_positions      = joints_positions[:n_steps]
        joints_velocities     = joints_velocities[:n_steps]
        joints_active_torques = joints_active_torques[:n_steps]
        links_positions       = links_positions[:n_steps]
        links_velocities      = links_velocities[:n_steps]
        joints_commands       = joints_commands[:n_steps] if joints_commands is not None else None

        #  Compute center of mass position
        com_positions         = np.mean( links_positions[:, :n_ax_links], axis= 1)

        # Extend links positions including the tail
        lenght_tail_link = self.snn_network.params.mechanics.mech_axial_links_length[-1]

        tail_positions = MechProc._compute_tail_positions_from_links_positions(
            links_positions  = links_positions,
            joints_positions = joints_positions,
            length_tail_link = lenght_tail_link,
            n_links_axis     = n_ax_links,
        )
        links_positions = np.insert(
            links_positions,
            n_ax_links,
            tail_positions,
            axis = 1,
        )

        return (
            joints_positions,
            joints_velocities,
            joints_active_torques,
            links_positions,
            links_velocities,
            com_positions,
            joints_commands,
        )

    ## METRICS COMPUTATION
    def farms_performance_metrics(
        self,
        sim_fraction      : float = 0.5,
        load_from_file    : bool  = False,
    ) -> dict[str,float]:
        ''' Campute all the FARMS-related metrics '''

        # Load data
        self.load_mechanical_simulation_data(load_from_file)
        data = self.sim_res_data

        # Parameters
        timestep          = data.timestep
        n_steps           = np.shape(data.sensors.links.array)[0]
        n_steps_fraction  = round( n_steps * sim_fraction )
        duration_fraction = timestep * n_steps_fraction

        n_joints_limb = self.snn_network.params.mechanics.mech_n_lb_joints
        n_lb_joints   = self.snn_network.params.mechanics.mech_limbs_joints
        n_ax_joints   = self.snn_network.params.mechanics.mech_axial_joints
        n_ax_links    = n_ax_joints + 1
        n_ax_points   = n_ax_links + 1

        inds_lb_pairs = self.snn_network.params.mechanics.mech_limbs_pairs_indices

        n_pca_joints = self.snn_network.params.mechanics.mech_pca_joints
        n_pca_links  = n_pca_joints + 1

        # Extract data
        (
            joints_positions,
            joints_velocities,
            joints_active_torques,
            links_positions,
            links_velocities,
            com_positions,
            joints_commands,
        ) = self.load_mechanical_simulation_arrays()

        # Compute metrics
        (
            joints_displacements,
            joints_displacements_mean,
            _joints_displacements_std,
            joints_displacements_amp,
        ) = MechProc.compute_joints_displacements_metrics(
            joints_positions = joints_positions,
            n_joints_axis    = n_ax_joints,
            sim_fraction     = sim_fraction,
        )

        (
            links_displacements,
            links_displacements_mean,
            _links_displacements_std,
            links_displacements_amp,
            links_displacements_data,
        ) = MechProc.compute_links_displacements_metrics(
            points_positions = links_positions,
            n_points_axis    = n_ax_points,
            sim_fraction     = sim_fraction,
            n_points_pca     = n_pca_links,
        )

        (
            freq_ax,
            freq_lb,
            freq_diff,
            freq_joints,
        ) = MechProc.compute_frequency(
            joints_positions = joints_positions,
            n_joints_axis    = n_ax_joints,
            n_joints_limbs   = n_lb_joints,
            timestep         = timestep,
            sim_fraction     = sim_fraction,
        )

        (
            ipl_ax_a,
            ipl_ax_t,
            ipl_lb_h,
            ipl_lb_c,
            ipl_lb_d,
            ipls_all,
        ) = MechProc.compute_joints_ipls(
            joints_angles     = joints_positions,
            joints_freqs      = freq_joints,
            n_joints_axis     = n_ax_joints,
            n_joints_per_limb = n_joints_limb,
            n_active_joints   = n_ax_joints + n_lb_joints,
            limb_pairs_inds   = inds_lb_pairs,
            timestep          = timestep,
        )

        (
            wave_number_a,
            wave_number_t
        ) = self.compute_wave_number(
            ipl_ax_a = ipl_ax_a,
            ipl_ax_t = ipl_ax_t,
        )

        # torque_commands = joints_commands[:, 1::2] - joints_commands[:, 0::2]
        # (
        #     ipl_command_ax_a,
        #     ipl_command_ax_t,
        #     ipl_command_lb_h,
        #     ipl_command_lb_c,
        #     ipl_command_lb_d,
        #     ipls_command__all,
        # ) = MechProc.compute_joints_ipls(
        #     joints_angles     = torque_commands,
        #     joints_freqs      = freq_joints,
        #     n_joints_axis     = n_ax_joints,
        #     n_joints_per_limb = n_joints_limb,
        #     n_active_joints   = n_ax_joints + n_lb_joints,
        #     limb_pairs_inds   = inds_lb_pairs,
        #     timestep          = timestep,
        # )

        (
            n2m_lag_ax_all,
            n2m_lag_ax_trk,
            n2m_lag_ax_tal,
            n2m_lag_lb_all,
            n2m_lags_all,
        ) = MechProc.compute_joints_neuro_muscolar_ipls(
            joints_commands = joints_commands,
            joints_angles   = joints_positions,
            joints_freqs    = freq_joints,
            n_joints_axis   = n_ax_joints,
            n_active_joints = n_ax_joints + n_lb_joints,
            limb_pairs_inds = inds_lb_pairs,
            timestep        = timestep,
        )

        speed_fwd, speed_lat, speed_abs = MechProc.compute_speed(
            links_positions_pca = links_displacements_data['links_positions_transformed'],
            n_links_axis        = n_ax_points,
            duration            = duration_fraction,
        )

        traj_mse = MechProc.compute_trajectory_linearity(
            timestep      = timestep,
            com_positions = com_positions,
            sim_fraction  = sim_fraction
        )

        traj_curv = MechProc.compute_trajectory_curvature(
            timestep     = timestep,
            com_pos      = com_positions,
            sim_fraction = sim_fraction
        )

        torque = MechProc.sum_torques(
            joints_torques = joints_active_torques,
            sim_fraction   = sim_fraction
        )

        energy = MechProc.sum_energy(
            torques      = joints_active_torques,
            speeds       = joints_velocities,
            timestep     = timestep,
            sim_fraction = sim_fraction,
        )

        # Derived metrics
        lenght_ax         = float( self.snn_network.params.topology.length_axial )
        tail_beat_amp     = links_displacements_amp[-1]
        distance_fwd      = speed_fwd * duration_fraction
        cost_of_transport = energy / np.abs(distance_fwd)
        stride_length     = speed_fwd / freq_ax

        if self.snn_network.params.simulation.gait == 'swim':
            velocity      = np.amax([np.abs(speed_fwd), 1e-6])
            mu_kin_water  = 1.0034 * 1e-6

            strouhal_number   = freq_ax * tail_beat_amp / velocity
            swimming_number   = 2*np.pi*freq_ax * tail_beat_amp * lenght_ax / mu_kin_water
            reynolds_number   = velocity * lenght_ax / mu_kin_water

        else:
            strouhal_number   = np.nan
            swimming_number   = np.nan
            reynolds_number   = np.nan

        # Normalize metrics
        speed_fwd_bl     = speed_fwd / lenght_ax
        speed_lat_bl     = speed_lat / lenght_ax
        speed_abs_bl     = speed_abs / lenght_ax

        tail_beat_amp_bl = tail_beat_amp / lenght_ax
        stride_length_bl = stride_length / lenght_ax

        links_displacements_amp_bl  = links_displacements_amp / lenght_ax

        # Return metrics
        mech_metrics = {
            # Frequency
            'mech_freq_ax'         : freq_ax,
            'mech_freq_lb'         : freq_lb,
            'mech_freq_diff'       : freq_diff,

            # Speed
            'mech_speed_fwd'       : speed_fwd,
            'mech_speed_lat'       : speed_lat,
            'mech_speed_abs'       : speed_abs,
            'mech_stride_len'      : stride_length,

            # Phase lag
            'mech_ipl_ax_a'        : ipl_ax_a,
            'mech_ipl_ax_t'        : ipl_ax_t,
            'mech_wave_number_a'   : wave_number_a,
            'mech_wave_number_t'   : wave_number_t,
            'mech_ipl_lb_h'        : ipl_lb_h,
            'mech_ipl_lb_c'        : ipl_lb_c,
            'mech_ipl_lb_d'        : ipl_lb_d,

            # Neuro-muscolar phase lag
            'mech_n2m_lag_ax_all'  : n2m_lag_ax_all,
            'mech_n2m_lag_ax_trk'  : n2m_lag_ax_trk,
            'mech_n2m_lag_ax_tal'  : n2m_lag_ax_tal,
            'mech_n2m_lag_lb_all'  : n2m_lag_lb_all,

            # Torque and energy
            'mech_torque'          : torque,
            'mech_energy'          : energy,
            'mech_cot'             : cost_of_transport,

            # Trajectory
            'mech_traj_mse'        : traj_mse,
            'mech_traj_curv'       : traj_curv,

            # Tail beat
            'mech_tail_beat_amp'   : tail_beat_amp,

            # Dimensionless numbers
            'mech_strouhal_number' : strouhal_number,
            'mech_swimming_number' : swimming_number,
            'mech_reynolds_number' : reynolds_number,

            # Normalized metrics
            'mech_speed_fwd_bl'    : speed_fwd_bl,
            'mech_speed_lat_bl'    : speed_lat_bl,
            'mech_speed_abs_bl'    : speed_abs_bl,
            'mech_stride_len_bl'   : stride_length_bl,
            'mech_tail_beat_amp_bl': tail_beat_amp_bl,

            # Displacements
            'mech_joints_disp_mean': joints_displacements_mean.tolist(),
            'mech_joints_disp_amp' : joints_displacements_amp.tolist(),
            'mech_links_disp_mean' : links_displacements_mean.tolist(),
            'mech_links_disp_amp'  : links_displacements_amp.tolist(),

            # Displacements normalized
            'mech_links_disp_amp_bl'  : links_displacements_amp_bl.tolist(),
        }

        assert all( [key in MECH_METRICS for key in mech_metrics.keys()] ), 'Not all metrics are listed'
        assert all( [key in mech_metrics for key in MECH_METRICS] ),        'Not all metrics are computed'

        logging.info(f'MECHANICS METRICS: {json.dumps(mech_metrics, indent=4)}')

        self.all_metrics_data = {
            'mech_metrics'            : mech_metrics,
            'joints_commands'         : joints_commands,
            'joints_positions'        : joints_positions,
            'joints_velocities'       : joints_velocities,
            'joints_active_torques'   : joints_active_torques,
            'links_positions'         : links_positions,
            'links_velocities'        : links_velocities,
            'com_positions'           : com_positions,
            'joints_displacements'    : joints_displacements,
            'links_displacements'     : links_displacements,
            'links_displacements_data': links_displacements_data,
            'freq_joints'             : freq_joints,
            'n2m_lags_all'            : n2m_lags_all,
        }

        if self.save_all_metrics_data:
            os.makedirs(self.results_data_folder, exist_ok=True)
            filename = f'{self.results_data_folder}/mechanics_metrics.dill'
            with open(filename, 'wb') as outfile:
                logging.info(f'Saving all metrics data to {filename}')
                dill.dump(self.all_metrics_data, outfile)

        return mech_metrics

    def compute_wave_number(self, ipl_ax_a, ipl_ax_t):
        ''' Compute the wave number '''

        n_ax_joints   = self.snn_network.params.mechanics.mech_axial_joints
        lenght_ax     = self.snn_network.params.mechanics.mech_axial_length
        joints_pos_ax = np.array( self.snn_network.params.mechanics.mech_axial_joints_position )
        joints_pos_lb = np.array( self.snn_network.params.mechanics.mech_limbs_pairs_positions )

        p0_ax = joints_pos_ax[0]
        p1_ax = joints_pos_ax[-1]
        p0_tr = joints_pos_lb[0] if len(joints_pos_lb) > 0 else p0_ax
        p1_tr = joints_pos_lb[1] if len(joints_pos_lb) > 1 else p1_ax

        range_ax = p1_ax - p0_ax
        range_tr = p1_tr - p0_tr

        ind0_t      = np.argmax(joints_pos_ax >= p0_tr)
        ind1_t      = np.argmax(joints_pos_ax >= p1_tr) - 1
        n_tr_joints = ind1_t - ind0_t + 1

        # Wave number
        wave_number_a = ipl_ax_a * (n_ax_joints - 1) * lenght_ax / range_ax
        wave_number_t = ipl_ax_t * (n_tr_joints - 1) * lenght_ax / range_tr

        return wave_number_a, wave_number_t