'''
Build network, run simulation, process results, plotting
'''
import logging
import time
import numpy as np

from typing import Any
from brian2 import second, pamp
from brian2.core.variables import VariableView
from dm_control.rl.control import PhysicsError

from network_modules.plotting.network_plotting import SnnPlotting
from network_modules.performance.signal_processor_snn import (
    compute_spike_count_limbs_online,
    detect_threshold_crossing_online
)

class SnnExperiment(SnnPlotting):
    '''
    Class used to run a simulation and plot the results
    '''

    ## RUN SIMULATIONS
    def simulation_run(self, param_scalings: np.ndarray= None) -> None:
        '''
        Run the simulation either with nominal or scaled parameter values.
        Save the simulation data for the following post-processing.
        Show information about the simulation.
        '''

        start = time.time()
        super().simulation_run(param_scalings)
        end = time.time()
        logging.info('SimTime= %.2f s', end-start)

        return

    # FUNCTIONALITIES
    def _spike_counter(self):
        ''' Optionally provide information about the number of CPG spikes '''
        inds_cpg_ax_sides = self.params.topology.network_modules['cpg']['axial'].indices_sides
        inds_cpg_lb_sides = self.params.topology.network_modules['cpg']['axial'].indices_sides
        spike_counts   = np.array( self.spikemon.count )
        l_sum          = (
              np.sum( spike_counts[ inds_cpg_ax_sides[0] ] )
            + np.sum( spike_counts[ inds_cpg_lb_sides[0] ] )
        )
        r_sum          = (
              np.sum( spike_counts[ inds_cpg_ax_sides[1] ] )
            + np.sum( spike_counts[ inds_cpg_lb_sides[1] ] )
        )
        logging.info('CPG Spike count --> Left = %d Right = %d', l_sum, r_sum)

    def _brian_profiling(self, sim_time):
        ''' Optionally provide information from the external profiler '''

        if not self.params.simulation.brian_profiling:
            return

        logging.info('Simulation profiling')
        for i, netobj in enumerate(self.network.profiling_info[:10]):
            trel = 100 * float(netobj[1]) / sim_time
            logging.info(
                f'{i:2d}) Network object : {netobj[0]: ^50} - Relative time: {trel :.2f} % '
            )

    def _show_simulation_information(self, sim_time):
        ''' Show information about the current simulation '''

        if self.params.monitor.spikes['active']:
            self._spike_counter()

        if self.params.simulation.brian_profiling:
            self._brian_profiling(sim_time= sim_time)

    # MOTOR OFFSETS
    def get_motor_offsets(self, _mech_iteration: int = None):
        ''' Returns the offsets for the ekeberg muscle models (gait-dependent) '''
        if self.params.simulation.gait == 'swim':
            return self.params.mechanics.mech_motor_offset_swimming
        else:
            return self.params.mechanics.mech_motor_offset_walking

    # PROPRIOCEPTION
    def _map_axial_joint_angles(
            self,
            joint_angles: np.ndarray[Any, float]
        ) -> np.ndarray[Any, float]:
        '''
        Map the model's angles to the segments of the network
        '''

        neur_axial_joints = self.params.topology.network_modules['ps']['axial'].pools
        mech_axial_joints = self.params.mechanics.mech_axial_joints
        angles_mapped     = np.zeros(neur_axial_joints)

        for joint in range(mech_axial_joints):
            prev_angle = joint_angles[ joint ]
            post_angle = joint_angles[ joint+1 ] if joint < mech_axial_joints-1 else 0

            prev_pool = self.params.input_ps_axial_pool_inds[joint]
            post_pool = self.params.input_ps_axial_pool_inds[joint+1] if joint < mech_axial_joints-1 else neur_axial_joints

            for pool_id in range(prev_pool, post_pool):
                angles_mapped[pool_id] = (
                    prev_angle + (pool_id - prev_pool) / (post_pool - prev_pool) * (post_angle - prev_angle)
                )

        # NOTE: ANGLE INCREASES WHEN THE HEAD IS TILTED TO THE RIGHT IN THE MECHANICAL MODEL
        angles_mapped_sides = np.concatenate( [angles_mapped, - angles_mapped] )
        return angles_mapped_sides

    def get_mechanics_input(self) -> None:
        ''' Receives input from the mechanical simulation through the queue'''

        mech_input = self.q_in.get(
            block   = True,
            timeout = 600,
        )

        # Check if simulation returned with error (and raise it in case)
        if isinstance(mech_input, PhysicsError):
            logging.warning('PhysicsError raised: Stopping current simulation')
            self._stop_simulation()
            raise mech_input

        # Get sensory feedback
        if self.params.topology.include_proprioception:
            self.get_proprioception_input(mech_input)
        return

    def get_proprioception_input(self, mech_input: np.ndarray) -> None:
        ''' Receives angles from the simulation and maps them to the propriosensory neurons '''
        self.input_thetas[:] = mech_input[:self.params.mechanics.mech_axial_joints]
        mapped_thetas_sides = self._map_axial_joint_angles( self.input_thetas )

        # Fast way of computing the inputs to the sensory neurons
        self.input_thetas_ps[:] = np.repeat(
            mapped_thetas_sides,
            repeats = self.params.topology.network_modules['ps'].n_pool  // 2
        )

        self.input_thetas_ps[ self.input_thetas_ps <= 0 ] = 0

        ps_lims = self.params.topology.network_modules['ps']['axial'].indices_limits
        setattr(
            self.pop[ ps_lims[0] : ps_lims[1] + 1 ],
            'I_ext',
            self.params.ps_gain_axial_callback * self.input_thetas_ps * pamp
        )
        return

    # MOTOR OUTPUT
    def _get_motor_output_axis(self, mech_iteration: int =None):
        '''
        Map the network status to the 8 links of the simulator axis
        Closed loop: computed from the motor neurons' output
        '''
        mech_axial_outputs = 2 * self.params.mechanics.mech_axial_joints

        if self.params.topology.include_muscle_cells_axial and self.params.monitor.muscle_cells['active']:
            self.motor_output[:mech_axial_outputs] = ( self.mc_pop.v )[self.params.output_mc_axial_inds]
        else:
            self.motor_output[:mech_axial_outputs] = 0

        self.motor_output[:mech_axial_outputs] *= self.params.mc_gain_axial_callback
        return

    def _get_motor_output_limbs(self, mech_iteration: int =None):
        '''
        Map the network status to the 16 links of the simulator limbs
        Closed loop: computed from the motor neurons' output
        '''

        mechanics     = self.params.mechanics

        ### TIME PARAMETERS
        timestep      = self.params.simulation.timestep
        curtime       = self.musclemon.t[-1]
        snn_iteration = round( curtime / timestep )

        ### MUSCLE CELLS PARAMETERS
        n_mc_axial = self.params.topology.network_modules['mc']['axial'].n_tot
        n_mc_limb  = 2 * mechanics.mech_n_lb_joints_act

        mc_ind_lead = mechanics.mc_lb_ind_lead
        mc_ind_foll = mechanics.mc_lb_ind_foll
        mc_ind_free = mechanics.mc_lb_ind_free

        ### MOTOR OUTPUT PARAMETERS
        n_mo_axial = 2 * mechanics.mech_axial_joints
        n_mo_limb  = 2 * mechanics.mech_n_lb_joints

        ### CHECKS

        # No motor neurons or no muscle activations
        if (
            mech_iteration == 0
            or not self.params.topology.include_muscle_cells_limbs
            or not self.params.monitor.muscle_cells['active']
        ):
            self.motor_output[n_mo_axial:] = 0
            return

        # SNN and MECH synchronization
        if snn_iteration != mech_iteration:
            raise ValueError(
                f'''
                SNN_ITERATION {snn_iteration} != MECH_ITERATION {mech_iteration}
                Curtime           = {curtime}
                Timestep          = {timestep}
                mc_pop.t          = {self.mc_pop.t}
                musclemon.v.shape = {self.musclemon.v.shape}
                musclemon.t.shape = {self.musclemon.t.shape}
                '''
            )

        ### TIME SHIFT OF THE LIMB DOFS
        if self.params.simulation.include_online_act:
            shift_t = np.mean( self.online_periods_lb ) * mechanics.mech_activation_delays
            shift_n = np.round( shift_t / float(timestep) )
        else:
            shift_n = 0 * mechanics.mech_activation_delays

        mc_pop_v_past_ind = np.clip(snn_iteration-shift_n, 0, self.musclemon.v.shape[1]-1)
        mc_pop_v_past     = self.musclemon.v[ :, mc_pop_v_past_ind.astype(int) ]

        ### COMPUTE MOTOR OUTPUT
        for lb in range(mechanics.mech_limbs):

            mo_ind0 = n_mo_limb * lb + n_mo_axial
            mc_ind0 = n_mc_limb * lb + n_mc_axial

            # Leader
            if mechanics.mech_n_lb_joints_lead:
                lb_pools_lead = mo_ind0 + mechanics.mech_lb_pools_lead
                lb_gains_lead = mechanics.mc_limbs_gains_lead[lb]

                m_curr_lead = ( self.mc_pop.v )[mc_ind0 + mc_ind_lead]
                self.motor_output[lb_pools_lead] = m_curr_lead * lb_gains_lead

            # Followers
            if mechanics.mech_n_lb_joints_foll:
                lb_pools_foll = mo_ind0 + mechanics.mech_lb_pools_foll
                lb_gains_foll = mechanics.mc_limbs_gains_foll[lb]

                m_past_lead = np.concatenate(mc_pop_v_past[mc_ind0 + mc_ind_foll, :].T)
                self.motor_output[lb_pools_foll] = m_past_lead * lb_gains_foll

            # Freely moving
            if mechanics.mech_n_lb_joints_free:
                lb_pools_free = mo_ind0 + mechanics.mech_lb_pools_free
                lb_gains_free = mechanics.mc_limbs_gains_free[lb]

                m_curr_free = ( self.mc_pop.v )[mc_ind0 + mc_ind_free]
                self.motor_output[lb_pools_free] = m_curr_free * lb_gains_free

        self.motor_output[n_mo_axial:] *= self.params.mc_gain_limbs_callback
        return

    def get_motor_output(self, mech_iteration: int =None) -> np.ndarray[Any, float]:
        '''
        Map the network status to the links of the simulator
        Closed loop: computed from the motor neurons' output
        '''

        # Motor activation
        self._get_motor_output_axis(mech_iteration)
        self._get_motor_output_limbs(mech_iteration)

        # Scale the co-contraction
        motor_output_cocontraction = np.mean(
            [
                self.motor_output[::2],
                self.motor_output[1::2],
            ],
            axis = 0
        )
        motor_output_cocontraction = np.repeat(motor_output_cocontraction, 2)

        self.motor_output += (
            - motor_output_cocontraction
            + motor_output_cocontraction * self.params.simulation.mo_cocontraction_gain
            + self.params.simulation.mo_cocontraction_off / 2
        )

        # Silenced joints
        self.motor_output[self.params.mechanics.mech_pools_silenced] = 0
        return self.motor_output

    ## STEP FUNCTIONS
    def step_desired_time(
        self,
        curtime    : VariableView,
        target_time: float,
    ) -> bool:
        ''' Prints at desired time during the simulation '''
        simtime = curtime - self.initial_time

        if abs(simtime - target_time) < self.params.simulation.callback_dt / 2:
            return True
        else:
            return False

    def step_ramp_current_axis(
        self,
        curtime: VariableView,
        min_off: float,
        max_off: float,
    ) -> bool:
        ''' Prints at desired time during the simulation '''
        simtime = curtime - self.initial_time

        self.params.simulation.stim_a_off = simtime / self.params.simulation.duration * (max_off - min_off) + min_off
        self.assign_drives()

    def step_ramp_current_limbs(
        self,
        curtime: VariableView,
        min_off: float,
        max_off: float,
    ) -> bool:
        ''' Prints at desired time during the simulation '''
        simtime = curtime - self.initial_time

        self.params.simulation.stim_l_off = simtime / self.params.simulation.duration * (max_off - min_off) + min_off
        self.assign_drives()

    def step_ramp_ps_gains_axis(
        self,
        curtime       : VariableView,
        min_ps_scaling: float,
        max_ps_scaling: float,
    ) -> bool:
        ''' Prints at desired time during the simulation '''
        sim_frac   = ( curtime - self.initial_time ) / self.params.simulation.duration
        ps_scaling = sim_frac * (max_ps_scaling - min_ps_scaling) + min_ps_scaling
        self.params.__define_callback_ps_gain_axial(ps_scaling= ps_scaling)

    def step_feedback_toggle(
        self,
        curtime    : VariableView,
        switch_time: float,
        active     : bool,
        ) -> None    :
        '''
        Activate feedback during simulation
        '''
        simtime = curtime - self.initial_time
        if abs(simtime - switch_time) < self.params.simulation.callback_dt / 2:

            message = f'Current time: {simtime} s'
            logging.info(message)
            message = 'Feedback ON' if active else 'Feedback OFF'
            logging.info(message)

            self.toggle_silencing_by_neural_inds_range(
                pop      = self.pop,
                syns     = [self.syn_ex, self.syn_in],
                limits   = self.params.topology.network_modules['ps'].indices_limits,
                silenced = not active
            )

    def step_toggle_silencing_by_neural_inds_range(
        self,
        curtime    : VariableView,
        switch_time: float,
        limits     : tuple[int, int],
        silenced   : bool,
    ) -> None:
        '''
        Toggle silencing of specified neurons during simulation
        '''
        simtime = curtime - self.initial_time
        if abs(simtime - switch_time) < self.params.simulation.callback_dt / 2:

            message = f'Current time: {simtime} s'
            logging.info(message)
            message = 'Silencing ON' if silenced else 'Silencing OFF'
            logging.info(message)

            self.toggle_silencing_by_neural_inds_range(
                pop      = self.pop,
                syns     = [self.syn_ex, self.syn_in],
                limits   = limits,
                silenced = silenced
            )



    def step_update_turning(
        self,
        curtime    : VariableView,
        switch_time: float,
        asymmetry  : float,
        ) -> None    :
        '''
        Update the value of the asymmetry in the drive to left and right sides
        '''
        simtime = curtime - self.initial_time
        if abs(simtime - switch_time) < self.params.simulation.callback_dt / 2:
            message = f'Current time: {simtime} s'
            logging.info(message)
            self.params.simulation.stim_lr_asym = asymmetry
            self.assign_drives()

    def step_gait_transition(
        self,
        curtime    : VariableView,
        switch_time: float,
        gait_label : str,
    ) -> None:
        '''
        Implement transition to the specified gait
        '''
        simtime = curtime - self.initial_time
        if abs(simtime - switch_time) < self.params.simulation.callback_dt / 2:
            message = f'Current time: {simtime} s'
            logging.info(message)
            self.params.simulation.gait = gait_label
            self.update_limb_connectivity()
            self.assign_drives()

    def step_drive_toggle(
        self,
        curtime    : VariableView,
        switch_time: float,
        active     : bool,
    ) -> None:
        '''
        Activate feedback during simulation
        '''
        simtime = curtime - self.initial_time
        if abs(simtime - switch_time) < self.params.simulation.callback_dt / 2:

            message = f'Current time: {simtime} s'
            logging.info(message)
            message = 'Drive ON' if active else 'Drive OFF'
            logging.info(message)

            if active:
                self.assign_drives()
            else:
                self.pop.I_ext = 0

    def step_compute_pools_activation_limbs_online(self, curtime):
        ''' Updates the value of the pools activations computed online '''
        simtime = curtime - self.initial_time

        self.online_count_lb = compute_spike_count_limbs_online(
            running_spike_count = self.online_count_lb,
            spikemon            = self.spikemon,
            curtime             = simtime,
            sig_dt              = self.params.simulation.timestep,
            callback_dt         = self.params.simulation.callback_dt,
            cpg_limb_module     = self.params.topology.network_modules['cpg']['limbs'],
        )

        # Moving average filtering
        self.online_activities_lb[:] = np.mean( self.online_count_lb, axis= 1 )

        curstep = round( float(simtime / self.params.simulation.callback_dt) )
        self.all_online_activities_lb[:, curstep] = self.online_activities_lb

        # Threshold crossing
        for seg in range(2*self.params.topology.segments_limbs):
            detect_threshold_crossing_online(
                self.online_activities_lb[seg],
                self.online_crossings[seg],
                self.online_onsets_lb[seg],
                self.online_offsets_lb[seg],
                self.params.simulation.online_threshold1,
                self.params.simulation.online_threshold2,
                simtime,
            )

    def step_compute_frequency_limbs_online(self, curtime):
        ''' Computes the frequency and duty cycle of limbs' oscillations '''
        simtime = curtime - self.initial_time

        for pool, onsets in enumerate( self.online_onsets_lb ):
            if len(onsets) < 2:
                self.online_periods_lb[pool] = 1 * second
                continue

            self.online_periods_lb[pool] = onsets[-1] - onsets[-2]

        curstep = round( float(simtime / self.params.simulation.callback_dt) )
        self.all_online_periods_lb[:, curstep] = self.online_periods_lb

    def step_compute_duty_limbs_online(self, curtime):
        ''' Computes the frequency and duty cycle of limbs' oscillations '''
        simtime = curtime - self.initial_time

        for pool, (onsets, offsets) in enumerate( zip(self.online_onsets_lb, self.online_offsets_lb) ):
            if len(onsets) < 2 or len(offsets) < 2:
                self.online_duties_lb[pool] = 0.5
                continue

            if onsets[-1] >= offsets[-1]:
                self.online_duties_lb[pool] = (offsets[-1] - onsets[-2] ) / (onsets[-1] - onsets[-2])
            else:
                self.online_duties_lb[pool] = (offsets[-1] - onsets[-1]) / (offsets[-1] - offsets[-2])

        curstep = round( float(simtime / self.params.simulation.callback_dt) )
        self.all_online_duties_lb[:, curstep] = self.online_duties_lb

    def step_apply_axial_transection(
        self,
        curtime             : VariableView,
        transection_time    : float,
        transection_s       : float,
        transection_f       : float,
        included_populations: list[str],
    ) -> None:
        ''' Apply transection to the axial CPG '''
        simtime = curtime - self.initial_time
        if abs(simtime - transection_time) < self.params.simulation.callback_dt / 2:
            message = f'Current time: {simtime} s'
            logging.info(message)
            message = (
                f'Applying transection between {transection_s} m and {transection_f} m '
                f'involving populations {included_populations}'
            )
            logging.info(message)
            self.apply_axial_transection(
                transection_s        = transection_s,
                transection_f        = transection_f,
                included_populations = included_populations,
            )

    ## EXPERIMENTS
    def apply_axial_transection(
        self,
        transection_s       : float,
        transection_f       : float,
        included_populations: list[str],
    ):
        ''' Apply transection to the axial CPG '''

        net_modules = self.params.topology.network_modules

        for pop_origin in included_populations:
            inds_ner_i = net_modules[pop_origin]['axial'].indices

            # Silence neurons across the transection
            if transection_s != transection_f:
                extra_cond_ner = (
                    ('y_mech',),
                    (
                        f'( ( y_mech > {transection_s} ) and'
                        f'  ( y_mech < {transection_f} ) )'
                    )
                )

                self.toggle_ner_silencing_by_neural_inds(
                    pop        = self.pop,
                    inds_ner   = inds_ner_i,
                    silenced   = True,
                    extra_cond = extra_cond_ner,
                )

            for pop_target in included_populations:
                inds_ner_j = net_modules[pop_target]['axial'].indices

                # Silence conections across the transection
                extra_cond_syn = (
                    ('y_mech_pre', 'y_mech_post'),
                    (
                        f'( ( y_mech_pre <= {transection_f} ) and'
                        f'  ( y_mech_post > {transection_s} ) )'
                        ' or '
                        f'( ( y_mech_pre   > {transection_s} ) and'
                        f'  ( y_mech_post <= {transection_f} ) )'
                    )
                )

                self.toggle_syn_silencing_by_neural_inds(
                    syns_list  = [self.syn_ex, self.syn_in],
                    inds_ner_i = inds_ner_i,
                    inds_ner_j = inds_ner_j,
                    silenced   = True,
                    extra_cond = extra_cond_syn,
                )

    def set_max_synaptic_range(
        self,
        syn_group_names     : list[str],
        range_up            : float,
        range_dw            : float,
        included_populations: list[list[str]],
    ):
        ''' Apply transection to the axial CPG '''

        net_modules = self.params.topology.network_modules

        syn_list = [
            getattr(self, syn_name)
            for syn_name in syn_group_names
            if getattr(self, syn_name) in self.synaptic_groups_list
        ]

        for pop_origin, pop_target in included_populations:
            inds_ner_i = net_modules[pop_origin]['axial'].indices
            inds_ner_j = net_modules[pop_target]['axial'].indices

            # Silence conections outside the range
            extra_cond_syn = (
                ('y_mech_pre', 'y_mech_post'),
                (
                    f'( ( y_mech_pre - y_mech_post >= {range_up} ) or'
                    f'  ( y_mech_post - y_mech_pre >= {range_dw} )  )'
                )
            )

            self.toggle_syn_silencing_by_neural_inds(
                syns_list  = syn_list,
                inds_ner_i = inds_ner_i,
                inds_ner_j = inds_ner_j,
                silenced   = True,
                extra_cond = extra_cond_syn,
            )

# TEST
def main():
    ''' Test case '''

    import matplotlib.pyplot as plt
    from queue import Queue
    from network_modules.parameters.network_parameters import SnnParameters

    logging.info('TEST: SNN Experiment ')

    experiment = SnnExperiment(
        network_name = 'snn_core_test',
        params_name  = 'pars_simulation_test',
        q_in         = Queue(),
        q_out        = Queue(),
    )

    experiment.define_network_topology()
    experiment.simulation_run()
    experiment.simulation_post_processing()
    experiment.simulation_plots()
    plt.plot()
    experiment.save_prompt()

    return experiment

if __name__ == '__main__':
    main()