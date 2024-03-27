'''
Network simulation including metrics to evaluate the network
Included network redefine method to initialize monitors
Included method to assign new parameters to the network
'''

import os
import shutil
import logging
import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt

from typing import Union
from queue import Queue
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

import network_modules.plotting.plots_utils as plots_utils
import network_modules.plotting.plots_snn as snn_plotting
import network_modules.plotting.animations_snn as net_anim

from network_modules.performance.network_performance import SnnPerformance

# FIGURE PARAMETERS
SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.rc(  'font',      size = SMALL_SIZE )  # controls default text sizes
plt.rc(  'axes', titlesize = SMALL_SIZE )  # fontsize of the axes title
plt.rc(  'axes', labelsize = MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc( 'xtick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc( 'ytick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend',  fontsize = SMALL_SIZE )  # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure',   figsize = (10.0, 5.0)) # size of the figure
plt.rc('lines',  linewidth = 2.0         ) # linewidth of the figure

# PLOTTING
class SnnPlotting(SnnPerformance):
    '''
    Class used plot the results of neuronal simulations
    '''

    def __init__(
        self,
        network_name: str,
        params_name : str,
        results_path: str,
        control_type: str,
        q_in        : Queue = None,
        q_out       : Queue = None,
        new_pars    : dict  = None,
        **kwargs,
    ) -> None:
        ''' Parameter initialization, network setup '''
        super().__init__(
            network_name = network_name,
            params_name  = params_name,
            results_path = results_path,
            control_type = control_type,
            q_in         = q_in,
            q_out        = q_out,
            new_pars     = new_pars,
            **kwargs
        )

        # Change matplotlib logging level to avoid undesired messages
        plt.set_loglevel("warning")

        # Initialize figures
        self.figures : dict[str, Union[Figure, list[Figure, FuncAnimation]]] = {}

        return

    # ALL RESULTS
    def simulation_plots(self) -> None:
        ''' Plots showing the network's behavior '''

        # Initialize figures
        self.figures : dict[str, Union[Figure, list[Figure, FuncAnimation]]] = {}

        # Plot
        self._plot_network_states_evolution()
        self._plot_connectivity_matrix()
        self._plot_raster_plot()
        self._plot_processed_pool_activation()
        self._plot_frequency_evolution()
        self._plot_ipl_evolution()
        self._plot_musclecells_evolution()
        self._plot_online_limb_activations_evolution()
        # Animations
        self._plot_raster_plot_animation()
        self._plot_network_states_animation()
        self._plot_processed_pool_animation()

    # AUXILIARY
    def _check_monitor_conditions(
            self,
            monitor: dict,
            mon_condlist: list[str],
            plot_condlist: list[str]
        ) -> bool:
        ''' Checks conditions from monitor '''

        condition = True
        for cond in mon_condlist:
            condition = condition and monitor[cond]
        for cond in plot_condlist:
            condition = condition and monitor['plotpars'][cond]

        return condition

    # INDIVIDUAL PLOTS
    # ------ [ Variables evolution ] ------
    def _plot_network_states_evolution(self) -> None:
        '''
        Evolution of the neuronal parameters
        '''
        monitor = self.params.monitor.states
        if not self._check_monitor_conditions(monitor, ['active','save'], ['showit', 'figure'] ):
            return

        inds = range(len(self.pop)) if monitor['indices'] is True else monitor['indices']

        v_rest  = float(
            self.params.neurons.shared_neural_params['V_rest'][0]
            * getattr(b2, self.params.neurons.shared_neural_params['V_rest'][1])
        )
        subpop = self.pop[inds]

        target_times = self.statemon_dict['t'] >= float(self.initial_time)

        def _get_var(var_name: str):
            ''' Return the desired recorded quantity in the target time interval '''
            return self.statemon_dict.get(var_name)[target_times]

        times = self.statemon_dict['t'][target_times]

        i_tot  = _get_var('I_tot')
        i_drv  = _get_var('I_ext')
        w1     = _get_var('w1')
        w2     = _get_var('w2')
        v_memb = _get_var('v')

        variables_values = {
            'Membrane potential' :           v_memb.T,
            'Descending drive'   :            i_drv.T,
            'Fast adaptation'    :               w1.T,
            'Slow adaptation'    :               w2.T,
            'Synaptic input'     : (   (i_tot - i_drv) / np.array(subpop.R_memb) ).T,
            'Leak current'       : ( (v_memb - v_rest) / np.array(subpop.R_memb)  ).T,
        }

        fig_te = plt.figure('Neuronal variables')
        snn_plotting.plot_temporal_evolutions(
            times            = times,
            variables_values = [ values for values in variables_values.values() ],
            variables_names  = [ keys   for keys   in variables_values.keys() ],
            inds             = inds,
            three_dim        = False
        )
        self.figures['fig_te'] = fig_te

    def _plot_network_states_animation(self) -> None:
        '''
        Animation of the evolution of the neuronal parameters
        '''
        monitor = self.params.monitor.states
        if not self._check_monitor_conditions(monitor, ['active','save'], ['showit', 'animate'] ):
            return

        fig_na = plt.figure('Neuronal activity animation')
        anim_na = net_anim.animation_neuronal_activity(
            fig            = fig_na,
            pop            = self.pop,
            statemon_times = self.statemon_dict['t'],
            statemon_dict  = self.statemon_dict,
            net_cpg_module = self.params.topology.network_modules['cpg'],
            height         = self.params.topology.height_segment_row,
            limb_positions = self.params.topology.limbs_i_positions
        )
        self.figures['fig_na'] = [fig_na, anim_na]

    def _plot_musclecells_evolution(self) -> None:
        '''
        Plot evolution of the muscle cells' variables
        '''
        monitor = self.params.monitor.muscle_cells
        if not self._check_monitor_conditions(monitor, ['active','save'], ['showit'] ):
            return

        mc_module = self.params.topology.network_modules['mc']

        if mc_module['axial'].include:
            fig_mc_a = plt.figure('Muscle cells evolution - AXIS')
            snn_plotting.plot_musclecells_evolutions_axial(
                musclemon_times = self.musclemon_dict['t'],
                musclemon_dict  = self.musclemon_dict,
                module_mc       = mc_module,
                plotpars        = monitor['plotpars'],
                starting_time   = self.initial_time,
            )
            self.figures['fig_mc_a'] = fig_mc_a

        if mc_module['limbs'].include:
            fig_mc_l = plt.figure('Muscle cells evolution - LIMBS')
            snn_plotting.plot_musclecells_evolutions_limbs(
                musclemon_times = self.musclemon_dict['t'],
                musclemon_dict  = self.musclemon_dict,
                module_mc       = mc_module,
                plotpars        = monitor['plotpars'],
                starting_time   = self.initial_time,
            )
            self.figures['fig_mc_l'] = fig_mc_l

    # \----- [ Variables evolution ] ------

    # ------ [ Connectivity matrices ] ------
    def _plot_connectivity_matrix(self) -> None:
        '''
        Show connections within the network
        '''
        monitor = self.params.monitor.connectivity
        if not self._check_monitor_conditions(monitor, ['active'], ['showit'] ):
            return

        fig_cm = plt.figure('Connectivity matrix')
        self.get_wmat()

        # TODO: Automatize depth to select for the modules
        leaf_modules = self.params.topology.network_leaf_modules[0]

        snn_plotting.plot_connectivity_matrix(
            pop_i                  = self.pop,
            pop_j                  = self.pop,
            w_syn                  = self.wmat,
            network_modules_list_i = leaf_modules,
            network_modules_list_j = leaf_modules,
        )
        self.figures['fig_cm'] = fig_cm

    def _plot_limbs_connectivity_matrix(self, label: str = '') -> None:
        '''
        Show connections within the limbs of the network
        '''
        fig_cm_lb = plt.figure(f'{label.upper()}_Limbs connectivity matrix')
        self.get_wmat()
        snn_plotting.plot_limb_connectivity(
            wsyn             = self.wmat,
            cpg_limbs_module = self.params.topology.network_modules['cpg']['limbs'],
            plot_label       = self.params.simulation.gait
        )
        self.figures[f'fig_cm_lb_{label}'] = fig_cm_lb

    # \----- [ Connectivity matrices ] ------

    # ------ [ Raster plots ] ------
    def _plot_raster_plot(self) -> None:
        '''
        Raster plot of spiking activity
        '''
        monitor = self.params.monitor.spikes
        if not self._check_monitor_conditions(monitor, ['active','save'], ['showit'] ):
            return

        # TODO: Automatize depth to select for the modules
        leaf_modules = self.params.topology.network_leaf_modules[0]

        fig_rp = plt.figure('Raster plot')
        snn_plotting.plot_raster_plot(
            pop                  = self.pop,
            spikemon_t           = self.spikemon_dict['t'],
            spikemon_i           = self.spikemon_dict['i'],
            duration             = self.params.simulation.duration,
            network_modules_list = leaf_modules,
            plotpars             = self.params.monitor.spikes['plotpars'],
            duration_ratio       = 1.0,
        )
        self.figures['fig_rp'] = fig_rp

        fig_rp_zoom = plt.figure('Raster plot - Zoom')
        snn_plotting.plot_raster_plot(
            pop                  = self.pop,
            spikemon_t           = self.spikemon_dict['t'],
            spikemon_i           = self.spikemon_dict['i'],
            duration             = self.params.simulation.duration,
            network_modules_list = leaf_modules,
            plotpars             = self.params.monitor.spikes['plotpars'],
            duration_ratio       = 0.2
        )
        self.figures['fig_rp_zoom'] = fig_rp_zoom


    def _plot_raster_plot_animation(self) -> None:
        '''
        Raster plot of spiking activity
        '''
        monitor = self.params.monitor.spikes
        if not self._check_monitor_conditions(monitor, ['active','save'], ['showit', 'animate'] ):
            return

        fig_rpa = plt.figure('Raster plot animation')
        anim_rpa = net_anim.animation_raster_plot(
            fig                  = fig_rpa,
            pop                  = self.pop,
            spikemon_t           = self.spikemon_dict['t'],
            spikemon_i           = self.spikemon_dict['i'],
            duration             = self.params.simulation.duration,
            timestep            = self.params.simulation.timestep,
            network_modules_list = self.params.topology.network_leaf_modules[0],
            plotpars             = self.params.monitor.spikes['plotpars']
        )
        self.figures['fig_rpa'] = [fig_rpa, anim_rpa]


    # \----- [ Raster plots ] ------

    # ------ [ Processed activations ] ------
    def _plot_processed_pool_activation(self) -> None:
        '''
        Evolution of the processed pools' activations
        '''
        monitor = self.params.monitor.pools_activation
        if not self._check_monitor_conditions(monitor, ['active'], ['showit'] ):
            return

        self._get_smooth_activations_cpg()
        self._get_smooth_activations_mn()
        self._get_oscillations_com()

        # CPG ACTIVATIONS
        if self.params.topology.include_cpg:
            fig_pa_cpg = plt.figure('CPG - Processed pools activations')

            signals_cpg = {
                'times_f'      : self.times_f,
                'spike_count_f': self.spike_count_f,
            }
            points_cpg  = {
                'com_x'   : self.com_x,
                'com_y'   : self.com_y,
                'strt_ind': self.start_indices,
                'stop_ind': self.stop_indices,
            }
            snn_plotting.plot_processed_pools_activations(
                signals        = signals_cpg,
                points         = points_cpg,
                seg_axial      = self.params.topology.segments_axial,
                seg_limbs      = self.params.topology.segments_limbs,
                duration       = self.params.simulation.duration,
                plotpars       = monitor['plotpars'],
            )
            self.figures['fig_pa_cpg'] = fig_pa_cpg

        # MN ACTIVATIONS
        if self.params.topology.include_motor_neurons:
            fig_pa_mn = plt.figure('MN - Processed pools activations')

            signals_mn = {
                'times_f'      : self.times_mn_f,
                'spike_count_f': self.spike_count_mn_f,
            }
            points_mn  = {}

            snn_plotting.plot_processed_pools_activations(
                signals        = signals_mn,
                points         = points_mn,
                seg_axial      = self.params.topology.segments_axial,
                seg_limbs      = self.params.topology.segments_limbs,
                duration       = self.params.simulation.duration,
                plotpars       = monitor['plotpars'],
            )
            self.figures['fig_pa_mn'] = fig_pa_mn

    def _plot_processed_pool_animation(self) -> None:
        '''
        Animation of the evolution of the processed pools' activations
        '''
        monitor = self.params.monitor.pools_activation
        if not self._check_monitor_conditions(monitor, ['active'], ['showit','animate'] ):
            return

        self._get_smooth_activations_cpg()
        fig_spa = plt.figure('Smooth neuronal activity animation')
        anim_spa = net_anim.animation_smooth_neural_activity(
            fig            = fig_spa,
            signals        = self.spike_count_f,
            limb_positions = self.params.topology.limbs_i_positions
        )
        self.figures['fig_spa'] = [fig_spa, anim_spa]

    # \----- [ Processed activations ] ------

    # ------ [ Metrics evolution ] ------
    def _plot_frequency_evolution(self) -> None:
        '''
        Plot evolution of the measured oscillations' frequencies
        '''
        monitor = self.params.monitor.freq_evolution
        if not self._check_monitor_conditions(monitor, ['active'], ['showit'] ):
            return

        self._get_network_freq() # Check or compute

        fig_fe = plt.figure('Frequency evolution')
        snn_plotting.plot_frequency_evolution(self.freq_evolution)
        self.figures['fig_fe'] = fig_fe

    def _plot_ipl_evolution(self) -> None:
        '''
        Plot evolution of the measured oscillations' IPL
        '''
        monitor = self.params.monitor.ipl_evolution
        if not self._check_monitor_conditions(monitor, ['active'], ['showit'] ):
            return

        self._get_network_ipls() # Check or compute

        fig_le = plt.figure('IPL evolution')
        snn_plotting.plot_ipl_evolution(
            ipls                = self.ipls_evolutions,
            plotpars            = monitor['plotpars'],
            limb_pair_positions = self.params.topology.limbs_pairs_i_positions
        )
        self.figures['fig_le'] = fig_le

    # \----- [ Metrics evolution ] ------

    # ------ [ Online metrics evolution ] ------
    def _plot_online_limb_activations_evolution(self) -> None:
        '''
        Plot evolution of the measured limb activations
        '''
        monitor = self.params.monitor.online_metrics
        if not monitor['active'] or not self.online_activities_lb.any():
            return

        callback_dt = self.params.simulation.callback_dt

        if monitor['plotpars']['activity']:
            fig_oa = plt.figure('Online activity evolution')
            snn_plotting.plot_online_activities_lb(self.all_online_activities_lb, callback_dt)
            self.figures['fig_oa'] = fig_oa

        if monitor['plotpars']['period']:
            fig_op = plt.figure('Online period evolution')
            snn_plotting.plot_online_periods_lb(self.all_online_periods_lb, callback_dt)
            self.figures['fig_op'] = fig_op

        if monitor['plotpars']['period']:
            fig_od = plt.figure('Online duty evolution')
            snn_plotting.plot_online_duties_lb(self.all_online_duties_lb, callback_dt)
            self.figures['fig_od'] = fig_od

    # \----- [ Online metrics evolution ] ------

    # ------ [ Data saving ] ------
    def save_prompt(
        self,
        figures_dict : dict[str, Figure] = None
    ) -> None:
        ''' Prompt user to choose whether to save the figures '''

        figures_snn  = self.figures if hasattr(self, 'figures') else {}
        figures_dict = figures_dict if figures_dict is not None else {}

        saved_figures, figures_path = plots_utils.save_prompt(
            figures_dict = figures_snn | figures_dict,
            folder_path  = self.params.simulation.figures_data_folder_run,
        )

        # Clear figures
        self.clear_figures()

        if not saved_figures:
            return

        # Move farms video file
        video_path_src = f'{self.params.simulation.results_data_folder_run}/farms/animation.mp4'
        video_path_dst = f'{figures_path}/animation.mp4'
        if os.path.isfile(video_path_src):
            logging.info('Copying %s file to %s', video_path_src,  video_path_dst)
            shutil.copyfile(video_path_src, video_path_dst)

        return

    # \----- [ Data saving ] ------

    # ------ [ Data cleaning ] ------
    def clear_figures(self) -> None:
        ''' Clear figures '''
        self.figures = {}
        plt.close('all')

    # \----- [ Data cleaning ] ------

# TEST
def main():
    ''' Test case '''

    from queue import Queue

    logging.info('TEST: SNN Plotting ')

    plotting = SnnPlotting(
        network_name = 'snn_core_test',
        params_name  = 'pars_simulation_test',
        q_in         = Queue(),
        q_out        = Queue(),
    )

    plotting.define_network_topology()
    plotting.simulation_run()
    plotting.simulation_post_processing()
    plotting.simulation_plots()
    plt.show()
    plotting.save_prompt()

    return plotting

if __name__ == '__main__':
    main()