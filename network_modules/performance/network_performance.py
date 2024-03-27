'''
Network simulation including metrics to evaluate the network
Included network redefine method to initialize monitors
Included method to assign new parameters to the network
'''
import os
import dill
import json
import logging
import numpy as np
import brian2 as b2

from queue import Queue

import network_modules.performance.signal_processor_snn as SigProc
import network_modules.performance.signal_processor_mech as MechProc

from network_modules.simulation.network_simulation import SnnSimulation

SNN_METRICS = [
    'neur_ptcc_ax',
    'neur_ptcc_lb',
    'neur_freq_ax',
    'neur_freq_lb',
    'neur_freq_diff',
    'neur_ipl_ax_a',
    'neur_ipl_ax_t',
    'neur_twl'
    'neur_wave_number_a',
    'neur_wave_number_t',
    'neur_ipl_lb_h',
    'neur_ipl_lb_c',
    'neur_ipl_lb_d',
    'neur_eff_mn_all',
    'neur_eff_mn_ax',
    'neur_eff_mn_lb',
    'neur_eff_mc_all',
    'neur_eff_mc_ax',
    'neur_eff_mc_lb',
    'neur_duty_cycle_r',
    'neur_duty_cycle_l',
    'neur_mean_cpg_ax',
    'neur_mean_cpg_lb',
    'neur_mean_mn_ax',
    'neur_mean_mn_lb',
    'neur_mean_mc_ax',
    'neur_mean_mc_lb',
    'neur_mean_mo_ax',
    'neur_mean_mo_lb',
    'neur_amp_cpg_ax',
    'neur_amp_cpg_lb',
    'neur_amp_mn_ax',
    'neur_amp_mn_lb',
    'neur_amp_mc_ax',
    'neur_amp_mc_lb',
    'neur_amp_mo_ax',
    'neur_amp_mo_lb',
]

class SnnPerformance(SnnSimulation):
    '''
    Class used to evaluate the performance of a simulation
    '''

    ## Initialization
    def __init__(
        self,
        network_name: str,
        params_name : str,
        results_path: str,
        q_in        : Queue = None,
        q_out       : Queue = None,
        new_pars    : dict  = None,
        **kwargs,
    ) -> None:
        ''' Parameter initialization, network setup, defines metrics '''
        super().__init__(
            network_name = network_name,
            params_name  = params_name,
            results_path = results_path,
            q_in         = q_in,
            q_out        = q_out,
            new_pars     = new_pars,
            **kwargs,
        )
        self._define_metrics()

    ## RUN SIMULATIONS
    def simulation_run(self, param_scalings: np.ndarray= None) -> None:
        '''
        Run the simulation either with nominal or scaled parameter values.
        Save the simulation data for the following post-processing.
        '''
        self._define_metrics()
        super().simulation_run(param_scalings)
        self.save_simulation_data()

    ## INITIALIZE METRICS
    def _define_online_metrics(self) -> None:
        ''' Initializes the values of the metrics computed online '''

        pars_sim = self.params.simulation

        if not pars_sim.include_online_act:
            return

        pools_lb = 2 * self.params.topology.segments_limbs
        window_n = round( float( pars_sim.online_act_window // pars_sim.timestep ) )

        self.online_onsets_lb  = [ [] for _ in range(pools_lb) ]
        self.online_offsets_lb = [ [] for _ in range(pools_lb) ]

        self.online_count_lb     = np.zeros((pools_lb, window_n))

        self.all_online_activities_lb = np.zeros((pools_lb, pars_sim.callback_steps))
        self.all_online_periods_lb    = np.zeros((pools_lb, pars_sim.callback_steps))
        self.all_online_duties_lb     = np.zeros((pools_lb, pars_sim.callback_steps))

        self.online_activities_lb = np.zeros(pools_lb)
        self.online_periods_lb    = np.zeros(pools_lb)
        self.online_duties_lb     = np.zeros(pools_lb)

        self.online_crossings = [
            {
                'on1'  : False,
                'on2'  : False,
                'off1' : False,
                'off2' : False,
                'onset_candidate' : 0,
                'offset_candidate' : 0,
            }
            for _ in range(pools_lb)
        ]

    def _define_metrics(self) -> None:
        ''' Initialize the value of the metrics '''

        if hasattr(self, 'initialized_metrics') and self.initialized_metrics:
            return

        # SIGNAL PROCESSING
        self.spikemon_dict    : dict[str, np.ndarray] = {}
        self.statemon_dict    : dict[str, np.ndarray] = {}
        self.musclemon_dict   : dict[str, np.ndarray] = {}
        self.times_f          = np.array([], dtype= float)
        self.spike_count_f    = np.array([], dtype= float)   # Filtered spike count (cpg)
        self.times_mn_f       = np.array([], dtype= float)
        self.spike_count_mn_f = np.array([], dtype= float)   # Filtered spike count (motor neurons)
        self.spike_count_fft  = np.array([], dtype= float)   # FFT of the filtered spike count
        self.com_x            : list[list[float]]     = []
        self.com_y            : list[list[float]]     = []   # COM of oscillations
        self.start_indices    : list[list[int]]       = []
        self.stop_indices     : list[list[int]]       = []   # Onsets and offsets

        # DEFINE METRICS
        self._define_online_metrics()
        self.ptccs           = np.array([], dtype= float)  # Peak to through correlation coefficients
        self.freqs           = np.array([], dtype= float)  # Mean frequenct of every segment
        self.ipls_evolutions = {}                          # Evolution of intersegmental phase lags amd their mean
        self.ipls_parts      = {}                          # Phase lag between different parts of the network
        self.hilb_freqs      = np.array([], dtype= float)  # Instantaneous frequency of every segment
        self.hilb_phases     = np.array([], dtype= float)  # Instantaneous phase of every segment
        self.freq_evolution  = np.array([], dtype= float)  # Evolution of the mean and std of the instantaneous frequency
        self.effs_mn         = np.array([], dtype= float)  # Efforts for the motor neurons
        self.effs_mc         = np.array([], dtype= float)  # Efforts for the muscle cells

        self.metrics : dict[str, float ]= {}

        self.initialized_metrics = True

    ## POST-PROCESSING
    def simulation_post_processing(self) -> dict[str, float]:
        ''' Computes the performance metrics of the simulation '''
        self.load_simulation_data()
        return self.network_performance_metrics()

    # SIMULATION DATA SAVING
    def _save_monitor_data(self, monitor_pars, monitor_name) -> None:
        ''' Saves monitor data as a dictionary of array '''

        data_path = self.params.simulation.results_data_folder_run
        data_file = f'{data_path}/' + '{mon_name}.dill'
        os.makedirs(data_path, exist_ok=True)

        if monitor_pars['active'] and monitor_pars['save']:
            filename = data_file.format(mon_name= monitor_name)
            logging.info('Saving self.%s data in %s', monitor_name, filename)
            with open(filename, 'wb') as outfile:
                dill.dump( getattr(self, monitor_name).get_states(units= False), outfile)

    def save_simulation_data(self) -> None:
        ''' Saves the data from the last simulation '''
        self._save_monitor_data(self.params.monitor.spikes, 'spikemon')
        self._save_monitor_data(self.params.monitor.states, 'statemon')
        self._save_monitor_data(self.params.monitor.muscle_cells, 'musclemon')
        return

    # SIMULATION DATA RETRIEVAL
    def _load_monitor_data(self, monitor_pars, monitor_name) -> dict[str, np.ndarray]:
        ''' Loads monitor data as a dictionary of array '''
        data_path = self.params.simulation.results_data_folder_run
        data_file = f'{data_path}/' + '{mon_name}.dill'

        if monitor_pars['active'] and monitor_pars['save']:
            filename = data_file.format(mon_name= monitor_name)
            logging.info('Loading self.%s data from %s', monitor_name, filename)
            with open(filename, 'rb') as infile:
                return dill.load(infile)
        return {}

    def load_simulation_data(self) -> None:
        ''' Loads the data from the last simulation '''
        self.spikemon_dict  = self._load_monitor_data(self.params.monitor.spikes, 'spikemon')
        self.statemon_dict  = self._load_monitor_data(self.params.monitor.states, 'statemon')
        self.musclemon_dict = self._load_monitor_data(self.params.monitor.muscle_cells, 'musclemon')
        return

    # METRICS COMPUTATION
    def _get_smooth_activations_cpg(self, recompute: bool = False) -> tuple[np.ndarray]:
        ''' Calculate smooth oscillations of the CPG segments if not already computed'''

        if recompute or not np.any(self.times_f) or not np.any(self.spike_count_f):
            self.times_f, self.spike_count_f = SigProc.compute_smooth_activation_module(
                    spikemon_t = self.spikemon_dict['t'],
                    spikemon_i = self.spikemon_dict['i'],
                    dt_sig     = b2.defaultclock.dt,
                    duration   = self.params.simulation.duration,
                    net_module = self.params.topology.network_modules['cpg'],
                    ner_type   = 'ex',
                )

        return self.times_f, self.spike_count_f

    def _get_smooth_activations_mn(self, recompute: bool = False) -> tuple[np.ndarray]:
        ''' Calculate smooth oscillations of the Motor Neurons if not already computed'''

        if not self.params.topology.include_motor_neurons:
            return self.times_mn_f, self.spike_count_mn_f

        if recompute or not np.any(self.times_mn_f) or not np.any(self.spike_count_mn_f):
            self.times_mn_f, self.spike_count_mn_f = SigProc.compute_smooth_activation_module(
                    spikemon_t = self.spikemon_dict['t'],
                    spikemon_i = self.spikemon_dict['i'],
                    dt_sig     = b2.defaultclock.dt,
                    duration   = self.params.simulation.duration,
                    net_module = self.params.topology.network_modules['mn'],
                    ner_type   = 'mn',
                )

        return self.times_mn_f, self.spike_count_mn_f

    def _get_oscillations_com(self, recompute: bool = False) -> tuple[list]:
        ''' Calculate COM of oscillations if not already computed'''

        if recompute or [] in [ self.com_x, self.com_y, self.start_indices, self.stop_indices ]:
            (
                self.com_x,
                self.com_y,
                self.start_indices,
                self.stop_indices
            ) = SigProc.process_network_smooth_signals_com(
                times         = self.times_f,
                signals       = self.spike_count_f,
                signals_ptccs = self.ptccs,
                discard_time  = 1 * b2.second
            )

        return self.com_x, self.com_y, self.start_indices, self.stop_indices

    def _get_hilbert_transform(self, recompute: bool = False) -> tuple[np.ndarray]:
        ''' Calculate phases and frequencies of oscillations if not already computed'''

        if recompute or not np.any(self.hilb_freqs) or not np.any(self.hilb_phases) :
            self.hilb_phases, self.hilb_freqs = SigProc.compute_hilb_transform(
                times        = self.times_f,
                signals      = self.spike_count_f,
                discard_time = 0 * b2.second,
                filtering    = self.params.simulation.metrics_filtering
            )

        return self.hilb_phases, self.hilb_freqs

    def _get_fourier_transform(self, recompute: bool = False) -> tuple[np.ndarray]:
        ''' Calculate phases and frequencies of oscillations if not already computed'''

        if recompute or not np.any(self.spike_count_fft):
            self.spike_count_fft = SigProc.compute_fft_transform(
                times        = self.times_f,
                signals      = self.spike_count_f,
            )

        return self.spike_count_fft

    def _get_network_ptcc(self, recompute: bool = False) -> np.ndarray:
        ''' Computes the PTCC of oscillations if not already computed '''

        if recompute or not np.any(self.ptccs):
            self.ptccs = SigProc.compute_ptcc(
                self.times_f,
                self.spike_count_f,
            )

        top     = self.params.topology
        ptcc_ax = float( np.mean(self.ptccs[:2*top.segments_axial]) ) if top.include_cpg_axial else np.nan
        ptcc_lb = float( np.mean(self.ptccs[2*top.segments_axial:]) ) if top.include_cpg_limbs else np.nan

        return ptcc_ax, ptcc_lb

    def _get_network_freq(self, recompute: bool = False) -> float:
        ''' Computes the frequencies of oscillations if not already computed '''

        # HILBERT (Frequency Evolutions)
        if recompute or not np.any(self.freq_evolution):
            self.freq_evolution = SigProc.compute_frequency_hilb(self.hilb_freqs)

        # FFT (Mean Frequency Values)
        if recompute or not np.any(self.freqs):
            self.freqs, _inds_freqs = SigProc.compute_frequency_fft(self.times_f, self.spike_count_fft)

        top     = self.params.topology
        freq_ax = float( np.mean(self.freqs[:2*top.segments_axial]) ) if top.include_cpg_axial else np.nan
        freq_lb = float( np.mean(self.freqs[2*top.segments_axial:]) ) if top.include_cpg_limbs else np.nan

        freq_diff = np.abs(freq_ax - freq_lb)

        return freq_ax, freq_lb, freq_diff

    def _get_network_ipls(self, recompute: bool = False) -> float:
        '''
        Computes the IPLS of oscillations if not already computed
        For swiming, consider the entire network.
        For walking, consider the phase jump at girdles.
        '''
        top_pars       = self.params.topology
        signals        = self.spike_count_f
        times          = self.times_f
        segments_axial = top_pars.segments_axial
        seg_per_limb   = top_pars.segments_per_limb
        limb_pair_pos  = top_pars.limbs_pairs_i_positions

        # All Axial (left/right)
        axial_inds_a0  = np.array(
            [ [ 2*seg +0, 2*(seg+1) +0 ] for seg in range(segments_axial-1)]
        )
        axial_inds_a1  = np.array(
            [ [ 2*seg +1, 2*(seg+1) +1 ] for seg in range(segments_axial-1)]
        )
        # Trunk Axial (left/right)
        trunk_ind_0 = limb_pair_pos[0] if limb_pair_pos else 0
        trunk_ind_1 = limb_pair_pos[1] if limb_pair_pos else segments_axial + 1

        axial_inds_t0  = np.array(
            [ [ 2*seg +0, 2*(seg+1) +0 ] for seg in range(trunk_ind_0, trunk_ind_1 - 2 ) ]
        )
        axial_inds_t1  = np.array(
            [ [ 2*seg +1, 2*(seg+1) +1 ] for seg in range(trunk_ind_0, trunk_ind_1 - 2 ) ]
        )

        # Homolateral (left-left)
        limbs_inds_h0 =  np.array(
            [
                [
                    2*segments_axial + 4 * seg_per_limb * lb_pair,
                    2*segments_axial + 4 * seg_per_limb * (lb_pair+1)
                ]
                for lb_pair in range( top_pars.limb_pairs-1 )
            ] if seg_per_limb else []
        )
        # Homolateral (right-right)
        limbs_inds_h1 = np.array(
            [
                [
                    2*segments_axial + 4 * seg_per_limb * lb_pair + 2*seg_per_limb,
                    2*segments_axial + 4 * seg_per_limb * (lb_pair+1) + 2*seg_per_limb
                ]
                for lb_pair in range( top_pars.limb_pairs-1 )
            ] if seg_per_limb else []
        )
        # Contralateral (left-right)
        limbs_inds_c0 =  np.array(
            [
                [
                    2*segments_axial + 4 * seg_per_limb * lb_pair,
                    2*segments_axial + 4 * seg_per_limb * lb_pair + 2*seg_per_limb
                ]
                for lb_pair in range( top_pars.limb_pairs-1 )
            ] if seg_per_limb else []
        )
        # Contralateral (right-left)
        limbs_inds_c1 = np.array(
            [
                [
                    2*segments_axial + 4 * seg_per_limb * lb_pair + 2*seg_per_limb,
                    2*segments_axial + 4 * seg_per_limb * lb_pair
                ]
                for lb_pair in range( top_pars.limb_pairs-1 )
            ] if seg_per_limb else []
        )
        # Diagonal (left-right)
        limbs_inds_d0 =  np.array(
            [
                [
                    2*segments_axial + 4 * seg_per_limb * lb_pair,
                    2*segments_axial + 4 * seg_per_limb * (lb_pair+1) + 2*seg_per_limb
                ]
                for lb_pair in range( top_pars.limb_pairs-1 )
            ] if seg_per_limb else []
        )
        # Diagonal (right-left)
        limbs_inds_d1 =  np.array(
            [
                [
                    2*segments_axial + 4 * seg_per_limb * lb_pair + 2*seg_per_limb,
                    2*segments_axial + 4 * seg_per_limb * (lb_pair+1)
                ]
                for lb_pair in range( top_pars.limb_pairs-1 )
            ] if seg_per_limb else []
        )

        # HILBERT (IPLs Evolutions)
        if recompute or not np.any(self.ipls_evolutions):
            self.ipls_evolutions = SigProc.compute_ipls_hilb(
                phases                  = self.hilb_phases,
                segments                = top_pars.segments_axial,
                limbs_pairs_i_positions = top_pars.limbs_pairs_i_positions,
                jump_at_girdles         = self.params.simulation.gait not in ['swim'],
                trunk_only              = self.params.simulation.metrics_trunk_only
            )

        # CROSS-CORRELATION (Mean IPLs values)
        if recompute or self.ipls_parts == {}:
            get_ipls = lambda inds : SigProc.compute_ipls_corr(times, signals, self.freqs, inds)

            self.ipls_parts['ipls_ax_a0'] = get_ipls(axial_inds_a0)
            self.ipls_parts['ipls_ax_a1'] = get_ipls(axial_inds_a1)
            self.ipls_parts['ipls_ax_t0'] = get_ipls(axial_inds_t0)
            self.ipls_parts['ipls_ax_t1'] = get_ipls(axial_inds_t1)
            self.ipls_parts['ipls_lb_h0'] = get_ipls(limbs_inds_h0)
            self.ipls_parts['ipls_lb_h1'] = get_ipls(limbs_inds_h1)
            self.ipls_parts['ipls_lb_c0'] = get_ipls(limbs_inds_c0)
            self.ipls_parts['ipls_lb_c1'] = get_ipls(limbs_inds_c1)
            self.ipls_parts['ipls_lb_d0'] = get_ipls(limbs_inds_d0)
            self.ipls_parts['ipls_lb_d1'] = get_ipls(limbs_inds_d1)

        # COMPUTE MEAN IPLS
        ipl_ax_a   = (
            np.mean( [ self.ipls_parts['ipls_ax_a0'], self.ipls_parts['ipls_ax_a1'] ] )
            if top_pars.include_cpg_axial else np.nan
        )
        ipl_ax_t   = (
            np.mean( [ self.ipls_parts['ipls_ax_t0'], self.ipls_parts['ipls_ax_t1'] ] )
            if top_pars.include_cpg_axial else np.nan
        )
        ipl_lb_h = (
            np.mean( [ self.ipls_parts['ipls_lb_h0'], self.ipls_parts['ipls_lb_h1'] ] )
            if top_pars.include_cpg_limbs else np.nan
        )
        ipl_lb_c = (
            np.mean( [ self.ipls_parts['ipls_lb_c0'], self.ipls_parts['ipls_lb_c1'] ] )
            if top_pars.include_cpg_limbs else np.nan
        )
        ipl_lb_d = (
            np.mean( [ self.ipls_parts['ipls_lb_d0'], self.ipls_parts['ipls_lb_d1'] ] )
            if top_pars.include_cpg_limbs else np.nan
        )

        # # PLOT OF IPLS
        # import matplotlib.pyplot as plt
        # ipls_ax_bilat = np.array( [ self.ipls_parts['ipls_ax_a0'], self.ipls_parts['ipls_ax_a1'] ] )
        # ipls_ax_mean  = np.mean(ipls_ax_bilat, axis= 0)
        # ipls_ax_cum   = np.cumsum(ipls_ax_mean) - np.cumsum(ipls_ax_mean)[7]
        # plt.plot( ipls_ax_cum / self.freqs[0] )
        # plt.grid()

        # COMPUTE WAVE NUMBER
        total_wave_lag = ipl_ax_a * (segments_axial - 1)

        ax_length = float(top_pars.length_axial)
        inds_cpg  = top_pars.network_modules.cpg.axial.indices
        y_ax_0    = np.amax( top_pars.neurons_y_mech[0][inds_cpg] )
        y_ax_1    = np.amin( top_pars.neurons_y_mech[0][inds_cpg] )
        y_ax_fr   = (y_ax_0 - y_ax_1) * (1 - 1/segments_axial) / ax_length

        wave_number_a = ipl_ax_a * (segments_axial - 1) / y_ax_fr

        segments_trunk = trunk_ind_1 - trunk_ind_0 - 1
        lb_pos = top_pars.limbs_pairs_y_positions
        y_tr_0 = float( top_pars.limbs_pairs_y_positions[0] ) if lb_pos else 0
        y_tr_1 = float( top_pars.limbs_pairs_y_positions[1] ) if lb_pos else ax_length
        y_tr_fr = (y_tr_1 - y_tr_0) * (1 - 1/segments_trunk) / ax_length

        wave_number_t = ipl_ax_t * (segments_trunk - 1) / y_tr_fr

        return (
            ipl_ax_a,
            ipl_ax_t,
            total_wave_lag,
            wave_number_a,
            wave_number_t,
            ipl_lb_h,
            ipl_lb_c,
            ipl_lb_d,
        )

    def _get_motor_neurons_effort(self, recompute: bool = False) -> float:
        '''
        Compute effort of the motor neurons activations
        based on the integral of their activations
        '''

        if not self.params.topology.include_motor_neurons:
            return np.nan, np.nan, np.nan

        if recompute or not np.any(self.effs_mn):
            self.effs_mn = SigProc.compute_effort(
                times   = self.times_mn_f,
                signals = self.spike_count_mn_f,
            )

        top = self.params.topology
        eff_mn_all = np.mean(self.effs_mn)
        eff_mn_ax  = np.mean(self.effs_mn[:2*top.segments_axial]) if top.include_motor_neurons_axial else np.nan
        eff_mn_lb  = np.mean(self.effs_mn[2*top.segments_axial:]) if top.include_motor_neurons_limbs else np.nan

        return eff_mn_all, eff_mn_ax, eff_mn_lb

    def _get_muscle_cells_effort(self, recompute: bool = False) -> float:
        '''
        Compute effort of the muscle cells activations
        based on the integral of their activations
        '''

        if not self.params.monitor.muscle_cells['active']:
            return np.nan, np.nan, np.nan

        if recompute or not np.any(self.effs_mc):
            self.effs_mc = SigProc.compute_effort(
                times   = self.musclemon.t,
                signals = self.musclemon.v,
            )

        top = self.params.topology
        eff_mc_all = np.mean(self.effs_mc)
        eff_mc_ax  = np.mean(self.effs_mc[:2*top.segments_axial]) if top.include_muscle_cells_axial else np.nan
        eff_mc_lb  = np.mean(self.effs_mc[2*top.segments_axial:]) if top.include_muscle_cells_limbs else np.nan

        return eff_mc_all, eff_mc_ax, eff_mc_lb

    def _get_duty_cycle(self, recompute: bool = False) -> float:
        '''
        Compute duty cycle of the muscle cells activations
        based on their activation period and total period
        '''

        if not self.params.monitor.muscle_cells['active']:
            return np.nan, np.nan

        duty_cycle_l, duty_cycle_r = SigProc.compute_duty_cycle(
            times     = self.musclemon.t,
            signals   = self.musclemon.v,
            threshold = 0.00
        )

        duty_cycle_l = np.mean(duty_cycle_l)
        duty_cycle_r = np.mean(duty_cycle_r)

        return duty_cycle_l,duty_cycle_r

    def _get_oscillations_means(self, recompute: bool = False) -> tuple[np.ndarray]:
        ''' Calculate amplitudes of oscillations if not already computed'''

        n_seg    = self.params.topology.segments
        n_seg_ax = self.params.topology.segments_axial
        n_osc    = 2 * self.params.topology.segments
        n_osc_ax = 2 * self.params.topology.segments_axial
        n_osc_lb = 2 * self.params.topology.segments_limbs
        mc_mon   = self.musclemon_dict

        means_cpg = (
            np.mean(self.spike_count_f.T, axis=0)
            if self.params.topology.include_cpg
            else
            np.array([np.nan] * n_osc)
        )
        means_mn  = (
            np.mean(self.spike_count_mn_f.T, axis=0)
            if self.params.topology.include_motor_neurons
            else
            np.array([np.nan] * n_osc)
        )
        means_mc  = (
            np.mean(mc_mon['v'], axis=0)
            if self.params.monitor.muscle_cells['active']
            else
            np.array([np.nan] * n_osc)
        )
        means_mo  = (
            np.mean(mc_mon['v'][0::2] - mc_mon['v'][1::2], axis=0)
            if self.params.monitor.muscle_cells['active']
            else
            np.array([np.nan] * n_seg)
        )

        mean_cpg_ax = np.mean(means_cpg[0:n_osc_ax]) if n_osc_ax else np.nan
        mean_cpg_lb = np.mean(means_cpg[n_osc_ax:])  if n_osc_lb else np.nan

        mean_mn_ax = np.mean(means_mn[0:n_osc_ax]) if n_osc_ax else np.nan
        mean_mn_lb = np.mean(means_mn[n_osc_ax:])  if n_osc_lb else np.nan

        mean_mc_ax = np.mean(means_mc[0:n_osc_ax]) if n_osc_ax else np.nan
        mean_mc_lb = np.mean(means_mc[n_osc_ax:])  if n_osc_lb else np.nan

        mean_mo_ax = np.mean(means_mo[0:n_seg_ax]) if n_osc_ax else np.nan
        mean_mo_lb = np.mean(means_mo[n_seg_ax:])  if n_osc_lb else np.nan

        return (
            mean_cpg_ax,
            mean_cpg_lb,
            mean_mn_ax,
            mean_mn_lb,
            mean_mc_ax,
            mean_mc_lb,
            mean_mo_ax,
            mean_mo_lb,
        )

    def _get_oscillations_amplitudes(self, recompute: bool = False) -> tuple[np.ndarray]:
        ''' Calculate amplitudes of oscillations if not already computed'''

        n_seg    = self.params.topology.segments
        n_seg_ax = self.params.topology.segments_axial
        n_osc    = 2 * self.params.topology.segments
        n_osc_ax = 2 * self.params.topology.segments_axial
        n_osc_lb = 2 * self.params.topology.segments_limbs
        mc_mon   = self.musclemon_dict

        amplitudes_cpg = (
            SigProc.compute_amplitudes_fft(self.spike_count_f.T)
            if self.params.topology.include_cpg
            else
            np.array([np.nan] * n_osc)
        )
        amplitudes_mn  = (
            SigProc.compute_amplitudes_fft(self.spike_count_mn_f.T)
            if self.params.topology.include_motor_neurons
            else
            np.array([np.nan] * n_osc)
        )
        amplitudes_mc  = (
            SigProc.compute_amplitudes_fft(mc_mon['v'])
            if self.params.monitor.muscle_cells['active']
            else
            np.array([np.nan] * n_osc)
        )
        amplitudes_mo  = (
            SigProc.compute_amplitudes_fft(mc_mon['v'][0::2] - mc_mon['v'][1::2] )
            if self.params.monitor.muscle_cells['active']
            else
            np.array([np.nan] * n_seg)
        )

        amplitude_cpg_ax = np.mean(amplitudes_cpg[0:n_osc_ax]) if n_osc_ax else np.nan
        amplitude_cpg_lb = np.mean(amplitudes_cpg[n_osc_ax:])  if n_osc_lb else np.nan

        amplitude_mn_ax = np.mean(amplitudes_mn[0:n_osc_ax]) if n_osc_ax else np.nan
        amplitude_mn_lb = np.mean(amplitudes_mn[n_osc_ax:])  if n_osc_lb else np.nan

        amplitude_mc_ax = np.mean(amplitudes_mc[0:n_osc_ax]) if n_osc_ax else np.nan
        amplitude_mc_lb = np.mean(amplitudes_mc[n_osc_ax:])  if n_osc_lb else np.nan

        amplitude_mo_ax = np.mean(amplitudes_mo[0:n_seg_ax]) if n_osc_ax else np.nan
        amplitude_mo_lb = np.mean(amplitudes_mo[n_seg_ax:])  if n_osc_lb else np.nan

        return (
            amplitude_cpg_ax,
            amplitude_cpg_lb,
            amplitude_mn_ax,
            amplitude_mn_lb,
            amplitude_mc_ax,
            amplitude_mc_lb,
            amplitude_mo_ax,
            amplitude_mo_lb,
        )

    def network_performance_metrics(self, recompute: bool = False) -> tuple[float]:
        ''' Compute metrics to evaluate network's performance '''

        recompute = recompute or self.initialized_metrics
        metrics   = self.metrics

        self._get_smooth_activations_cpg(recompute)
        self._get_smooth_activations_mn(recompute)
        self._get_hilbert_transform(recompute)
        self._get_fourier_transform(recompute)

        (
            metrics['neur_ptcc_ax'],
            metrics['neur_ptcc_lb'],
        ) = self._get_network_ptcc(recompute)
        (
            metrics['neur_freq_ax'],
            metrics['neur_freq_lb'],
            metrics['neur_freq_diff'],
        ) = self._get_network_freq(recompute)
        (
            metrics['neur_ipl_ax_a'],
            metrics['neur_ipl_ax_t'],
            metrics['neur_twl'],
            metrics['neur_wave_number_a'],
            metrics['neur_wave_number_t'],
            metrics['neur_ipl_lb_h'],
            metrics['neur_ipl_lb_c'],
            metrics['neur_ipl_lb_d'],
        ) = self._get_network_ipls(recompute)
        (
            metrics['neur_eff_mn_all'],
            metrics['neur_eff_mn_ax'],
            metrics['neur_eff_mn_lb'],
        ) = self._get_motor_neurons_effort(recompute)
        (
            metrics['neur_eff_mc_all'],
            metrics['neur_eff_mc_ax'],
            metrics['neur_eff_mc_lb'],
        ) = self._get_muscle_cells_effort(recompute)
        (
            metrics['neur_duty_cycle_l'],
            metrics['neur_duty_cycle_r'],
        ) = self._get_duty_cycle(recompute)     #added HERE
        (
            metrics['neur_mean_cpg_ax'],
            metrics['neur_mean_cpg_lb'],
            metrics['neur_mean_mn_ax'],
            metrics['neur_mean_mn_lb'],
            metrics['neur_mean_mc_ax'],
            metrics['neur_mean_mc_lb'],
            metrics['neur_mean_mo_ax'],
            metrics['neur_mean_mo_lb'],
        ) = self._get_oscillations_means(recompute)
        (
            metrics['neur_amp_cpg_ax'],
            metrics['neur_amp_cpg_lb'],
            metrics['neur_amp_mn_ax'],
            metrics['neur_amp_mn_lb'],
            metrics['neur_amp_mc_ax'],
            metrics['neur_amp_mc_lb'],
            metrics['neur_amp_mo_ax'],
            metrics['neur_amp_mo_lb'],
        ) = self._get_oscillations_amplitudes(recompute)

        assert all( [key in SNN_METRICS  for key in metrics.keys()] ), 'Not all metrics are listed'
        assert all( [key in metrics for key in SNN_METRICS] ),         'Not all metrics are computed'

        self.initialized_metrics = False

        logging.info(f'NEURAL METRICS: {json.dumps(metrics, indent=4)}')
        return self.metrics

# TEST
def main():
    ''' Test case '''

    from queue import Queue

    logging.info('TEST: SNN Performance ')

    performance = SnnPerformance(
        network_name = 'snn_core_test',
        params_name  = 'pars_simulation_test',
        q_in         = Queue(),
        q_out        = Queue(),
    )

    performance.define_network_topology()
    performance.simulation_run()
    performance.simulation_post_processing()

    return performance

if __name__ == '__main__':
    main()
