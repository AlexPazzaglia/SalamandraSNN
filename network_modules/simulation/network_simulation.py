'''
Network simulation including metrics to evaluate the network
Included network redefine method to initialize monitors
Included method to assign new parameters to the network
'''
import logging
import copy
import numpy as np

from typing import Union

from brian2 import pamp, NeuronGroup, Synapses

from network_modules.equations import parameter_setting
from network_modules.build.network_build import SnnBuild
from network_modules.parameters.pars_simulation import SnnParsSimulation

class SnnSimulation(SnnBuild):
    '''
    Class used to run a simulation
    '''

    ## RUN SIMULATION
    def _simulation_run_nominal(self) -> None:
        ''' Run the simulation with the nominal parameter. '''

        self.network.run(
            duration = self.params.simulation.duration,
            profile  = self.params.simulation.brian_profiling,
            report   = 'text' if self.params.simulation.verboserun else None
        )
        return

    def _simulation_run_scaling(
        self,
        param_scalings: dict[ str, list[ dict[str, Union[str, np.ndarray]] ] ]
    ) -> None:
        ''' Run the simulation with desired value scaling. '''

        # SCALE NEURAL PARAMETERS
        for scaling_params in  param_scalings.get('neural_params', []):

            self.assign_single_scaled_neural_parameter_by_neural_inds(
                ner_group       = self.neuron_groups_list[scaling_params['neuron_group_ind']],
                inds_ner        = scaling_params['indices'],
                par_key         = scaling_params['var_name'],
                par_val_nominal = scaling_params['nominal_value'],
                par_scaling     = scaling_params['scaling'],
                par_std_value   = self.params.neurons.std_val,
            )

        # SCALE SYNAPTIC PARAMETERS
        for scaling_params in param_scalings.get('synaptic_params', []):

            self.assign_single_scaled_synaptic_parameter_by_neural_inds(
                syn_group        = self.synaptic_groups_list[scaling_params['syn_group_ind']],
                inds_syn_i       = scaling_params['indices_i'],
                inds_syn_j       = scaling_params['indices_j'],
                par_key          = scaling_params['var_name'],
                par_val_nominal  = scaling_params['nominal_value'],
                par_scaling      = scaling_params['scaling'],
                par_std_value    = self.params.neurons.std_val,
            )

        # RUN SIMULATION
        self.network.run(duration= self.params.simulation.duration)
        return

    def simulation_run(self, param_scalings: np.ndarray = None) -> None:
        ''' Run the simulation either with nominal or scaled parameter values. '''

        # Set seed again for repeatability
        self._set_seed()

        if param_scalings is None:
            logging.info('Running simulation with nominal parameter values')
            self._simulation_run_nominal()
        else:
            logging.info('Running simulation with scaled parameter values')
            self._simulation_run_scaling(param_scalings)

    ## STOP SIMULATION
    def _stop_simulation(self) -> None:
        ''' Stop the simulation. '''
        self.network.stop()

       # Ensure that the queues are empty
        with self.q_in.mutex:
            self.q_in.queue.clear()
        with self.q_out.mutex:
            self.q_out.queue.clear()

        logging.info('Stopped simulation')
        return

    ## AUXILIARY FUNCTIONS
    def assign_single_scaled_neural_parameter_by_module_name(
        self,
        ner_group      : NeuronGroup,
        module_name    : str,
        par_val_nominal: dict[str, float],
        par_key        : list[str],
        par_scaling    : list[float],
        par_std_value  : float = 0.0,
    ) -> None:
        '''
        Auxiliary function to scale and assign a neuronal
        parameter to the desired population subset, according to the provided scaling
        '''
        net_modules = self.params.topology.network_modules
        trg_module  = net_modules.get_sub_module_from_full_name(module_name)

        parameter_setting.set_scaled_neural_parameters_by_neural_inds(
            ner_group         = ner_group,
            inds_ner          = trg_module.indices,
            pars_keys         = [par_key],
            pars_vals_nominal = {par_key : par_val_nominal},
            pars_scalings     = [par_scaling],
            pars_std_value    = par_std_value,
        )
        return

    def assign_multiple_scaled_neural_parameters_by_module_name(
        self,
        ner_group        : NeuronGroup,
        module_name      : str,
        inds_ner         : list[int],
        pars_vals_nominal: dict[str, float],
        pars_keys        : list[str],
        pars_scalings    : list[float],
        pars_std_value   : float = 0.0,
    ) -> None:
        '''
        Auxiliary function to scale and assign the neuronal
        parameters to the desired population subset, according to the provided scalings
        '''

        net_modules = self.params.topology.network_modules
        trg_module  = net_modules.get_sub_module_from_full_name(module_name)

        parameter_setting.set_scaled_neural_parameters_by_neural_inds(
            ner_group         = ner_group,
            inds_ner          = inds_ner,
            pars_vals_nominal = pars_vals_nominal,
            pars_keys         = pars_keys,
            pars_scalings     = pars_scalings,
            pars_std_value    = pars_std_value,
        )
        return

    def assign_single_scaled_neural_parameter_by_neural_inds(
        self,
        ner_group      : NeuronGroup,
        inds_ner       : list[int],
        par_val_nominal: dict[str, float],
        par_key        : list[str],
        par_scaling    : list[float],
        par_std_value  : float = 0.0,
    ) -> None:
        '''
        Auxiliary function to scale and assign a neuronal
        parameter to the desired population subset, according to the provided scaling
        '''
        parameter_setting.set_scaled_neural_parameters_by_neural_inds(
            ner_group         = ner_group,
            inds_ner          = inds_ner,
            pars_keys         = [par_key],
            pars_vals_nominal = {par_key : par_val_nominal},
            pars_scalings     = [par_scaling],
            pars_std_value    = par_std_value,
        )
        return

    def assign_multiple_scaled_neural_parameters_by_neural_inds(
        self,
        ner_group        : NeuronGroup,
        inds_ner         : list[int],
        pars_vals_nominal: dict[str, float],
        pars_keys        : list[str],
        pars_scalings    : list[float],
        pars_std_value   : float = 0.0,
    ) -> None:
        '''
        Auxiliary function to scale and assign the neuronal
        parameters to the desired population subset, according to the provided scalings
        '''
        parameter_setting.set_scaled_neural_parameters_by_neural_inds(
            ner_group         = ner_group,
            inds_ner          = inds_ner,
            pars_vals_nominal = pars_vals_nominal,
            pars_keys         = pars_keys,
            pars_scalings     = pars_scalings,
            pars_std_value    = pars_std_value,
        )
        return

    def assign_single_scaled_synaptic_parameter_by_neural_inds(
        self,
        syn_group      : Synapses,
        inds_syn_i     : list[int],
        inds_syn_j     : list[int],
        par_key        : list[str],
        par_val_nominal: dict[str, float],
        par_scaling    : list[float],
        par_std_value  : float = 0.0,
    ) -> None:
        '''
        Auxiliary function to scale and assign a synaptic
        parameter to the desired synaptic group, according to the provided scaling
        '''

        parameter_setting.set_scaled_synaptic_parameters_by_neural_ind(
            syn_group         = syn_group,
            inds_syn_i        = inds_syn_i,
            inds_syn_j        = inds_syn_j,
            pars_vals_nominal = {par_key : par_val_nominal},
            pars_keys         = [par_key],
            pars_scalings     = [par_scaling],
            pars_std_value    = par_std_value,
        )

    def assign_multiple_scaled_synaptic_parameters_by_inds(
        self,
        syn_group        : Synapses,
        inds_syn_i       : list[int],
        inds_syn_j       : list[int],
        pars_vals_nominal: dict[str, float],
        pars_keys        : list[str],
        pars_scalings    : list[float],
        pars_std_value   : float = 0.0,
    ) -> None:
        '''
        Auxiliary function to scale and assign the synaptic
        parameters to the desired synaptic group, according to the provided scalings
        '''

        parameter_setting.set_scaled_synaptic_parameters_by_neural_ind(
            syn_group         = syn_group,
            inds_syn_i        = inds_syn_i,
            inds_syn_j        = inds_syn_j,
            pars_vals_nominal = pars_vals_nominal,
            pars_keys         = pars_keys,
            pars_scalings     = pars_scalings,
            pars_std_value    = pars_std_value,
        )

    ## PROPERTIES
    def __update_currents(self):
        self.assign_drives()

    @SnnParsSimulation.stim_a_mul.setter
    def stim_a_mul(self, value):
        SnnParsSimulation.stim_a_mul.fset(self.params.simulation, value)
        self.__update_currents()
    @SnnParsSimulation.stim_a_off.setter
    def stim_a_off(self, value):
        SnnParsSimulation.stim_a_off.fset(self.params.simulation, value)
        self.__update_currents()
    @SnnParsSimulation.stim_l_mul.setter
    def stim_l_mul(self, value):
        SnnParsSimulation.stim_l_mul.fset(self.params.simulation, value)
        self.__update_currents()
    @SnnParsSimulation.stim_l_off.setter
    def stim_l_off(self, value):
        SnnParsSimulation.stim_l_off.fset(self.params.simulation, value)
        self.__update_currents()
    @SnnParsSimulation.stim_lr_asym.setter
    def stim_lr_asym(self, value):
        SnnParsSimulation.stim_lr_asym.fset(self.params.simulation, value)
        self.__update_currents()
    @SnnParsSimulation.stim_lr_off.setter
    def stim_lr_off(self, value):
        SnnParsSimulation.stim_lr_off.fset(self.params.simulation, value)
        self.__update_currents()
    @SnnParsSimulation.stim_fe_asym.setter
    def stim_fe_asym(self, value):
        SnnParsSimulation.stim_fe_asym.fset(self.params.simulation, value)
        self.__update_currents()
    @SnnParsSimulation.stim_fe_off.setter
    def stim_fe_off(self, value):
        SnnParsSimulation.stim_fe_off.fset(self.params.simulation, value)
        self.__update_currents()
    @SnnParsSimulation.stim_f_mn_off.setter
    def stim_f_mn_off(self, value):
        SnnParsSimulation.stim_f_mn_off.fset(self.params.simulation, value)
        self.__update_currents()
    @SnnParsSimulation.stim_e_mn_off.setter
    def stim_e_mn_off(self, value):
        SnnParsSimulation.stim_e_mn_off.fset(self.params.simulation, value)
        self.__update_currents()
    @SnnParsSimulation.gains_drives_axis.setter
    def gains_drives_axis(self, value):
        SnnParsSimulation.gains_drives_axis.fset(self.params.simulation, value)
        self.__update_currents()
    @SnnParsSimulation.gains_drives_limbs.setter
    def gains_drives_limbs(self, value):
        SnnParsSimulation.gains_drives_limbs.fset(self.params.simulation, value)
        self.__update_currents()

# TEST
def main():
    ''' Test case '''

    import matplotlib.pyplot as plt
    from queue import Queue
    import network_modules.plotting.plots_snn as snn_plotting

    logging.info('TEST: SNN Simulation ')

    simulation = SnnSimulation(
        network_name = 'snn_core_test',
        params_name  = 'pars_simulation_test',
        q_in         = Queue(),
        q_out        = Queue(),
        gaitflag = 0,
    )

    simulation.define_network_topology()

    vpars_units = simulation.get_variable_parameters()
    scalings    = 1 + 0.1 * ( -0.5 + np.random.rand(len(vpars_units)) )

    simulation.simulation_run(param_scalings= scalings)

    # Plotting
    sub_modules = [
        sub_sub_mod
        for module in simulation.params.topology.network_modules.sub_parts_list
        for sub_mod in module.sub_parts_list
        for sub_sub_mod in sub_mod.sub_parts_list
        if sub_sub_mod.neuron_group == 0
    ]

    plt.figure('Raster Plot')
    snn_plotting.plot_raster_plot(
        pop                  = simulation.pop,
        spikemon_t           = simulation.spikemon.get_states('t', units= False)['t'],
        spikemon_i           = simulation.spikemon.get_states('i', units= False)['i'],
        duration             = simulation.params.simulation.duration,
        network_modules_list = sub_modules,
        plotpars             = simulation.params.monitor.spikes['plotpars']
    )

    plt.figure('Neurons Identifiers')
    snn_plotting.plot_neuronal_identifiers(
        pop                  = simulation.pop,
        network_modules_list = sub_modules,
        identifiers_list     = [
            'ner_id',
            'side_id',
            'pool_id',
            'limb_id',
            'I_ext',
        ]
    )

    plt.show()

    return simulation

if __name__ == '__main__':
    main()
