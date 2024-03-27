'''
Functions to specify the connectivity within the network
'''
import os
import random
import logging
import numpy as np
import brian2 as b2

from queue import Queue

from network_modules.core.utils import is_writeable_property
from network_modules.parameters.network_parameters import SnnParameters

from network_modules.equations import (
    neural_models,
    synaptic_models,
)

class SnnCore():
    ''' Core network elements '''

    #### Initialization
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

        # Network parameters
        self.params = SnnParameters(
            parsname     = params_name,
            results_path = results_path,
            new_pars     = new_pars,
            netname      = network_name,
            **kwargs
        )

        self.netname      = network_name
        self.control_type = control_type
        self.q_in         = q_in
        self.q_out        = q_out

        # Brian2
        self.initial_time       = 0 * b2.second
        b2.defaultclock.dt      = self.params.simulation.timestep
        b2.prefs.codegen.target = 'cython'

        # Seed
        self.seed_assigned = False
        self._set_seed()

        # Define brian2 network objects
        self.neuron_groups_list             : list[b2.NeuronGroup] = []
        self.synaptic_groups_list           : list[b2.Synapses]    = []
        self.excluded_synaptic_groups_names : list[str]            = []
        self._define_network_objects()

    ### POPULATION PARAMETERS
    def _define_neuronal_identifiers(self) -> None:
        '''
        Define internal parameters to distinguish neurons
        '''
        for neuron_group, pop in enumerate(self.neuron_groups_list):
            pop[:].side_id    = self.params.topology.neurons_side_id[neuron_group]
            pop[:].ner_id     = self.params.topology.neurons_ner_id[neuron_group]
            pop[:].ner_sub_id = self.params.topology.neurons_ner_sub_id[neuron_group]
            pop[:].limb_id    = self.params.topology.neurons_limb_id[neuron_group]
            pop[:].pool_id    = self.params.topology.neurons_pool_id[neuron_group]
            pop[:].y_neur     = self.params.topology.neurons_y_neur[neuron_group] * b2.metre
            pop[:].y_mech     = self.params.topology.neurons_y_mech[neuron_group] * b2.metre

    ### NETWORK OBJECTS
    def _define_neuronal_populations(self):
        ''' Define the neuronal populations '''

        self.neuron_groups_list : list[b2.NeuronGroup] = []

        net_modules = self.params.topology.network_modules
        silencing = self.params.simulation.include_silencing
        noise_term = self.params.simulation.include_noise_term

        ### NEURON_GROUP_0
        # Equations
        self.eqs = neural_models.define_neuronal_model(
            neuronal_model_type = self.params.neurons.neuron_type_network,
            syn_labels          = self.params.neurons.synaptic_labels,
            silencing           = silencing,
            noise_term          = noise_term,
        )

        self.reset = neural_models.define_reset_condition(
            n_adaptation_variables= self.params.neurons.n_adaptation_variables
        )
        self.thres   = neural_models.define_threshold_condition()
        self.refract = neural_models.define_refractoriness()

        # Initialization values
        self.neuronal_initial_values = neural_models.define_model_initial_values(
            neuronal_model_type = self.params.neurons.neuron_type_network,
            synlabels           = self.params.neurons.synaptic_labels,
        )

        # Neuron group
        self.pop = b2.NeuronGroup(
            self.params.topology.n_tot[0],                         # Number of neurons
            self.eqs,                                              # Set of differential equations
            threshold  = self.thres,                               # Spike threshold
            reset      = self.reset,                               # Reset operations
            method     = self.params.simulation.int_method,        # Integration method
            refractory = self.refract,
            name       = 'pop'
        )
        self.network.add(self.pop)
        self.neuron_groups_list.append(self.pop)

        ### NEURON_GROUP_1
        if self.params.topology.include_muscle_cells:

            # Equations
            self.mc_eqs = neural_models.define_neuronal_model(
                neuronal_model_type = self.params.neurons.neuron_type_muscle,
                syn_labels          = self.params.neurons.synaptic_labels,
                silencing           = silencing,
                noise_term          = noise_term,
            )
            self.mc_refract = neural_models.define_refractoriness()

            # Initialization values
            self.muscle_initial_values = neural_models.define_model_initial_values(
                neuronal_model_type = self.params.neurons.neuron_type_muscle,
                synlabels           = self.params.neurons.synaptic_labels,
            )

            # Neuron group
            self.mc_pop = b2.NeuronGroup(
                net_modules['mc'].n_tot,
                self.mc_eqs,
                method     = self.params.simulation.int_method,
                refractory = self.mc_refract,
                name       = 'mc_pop'
            )
            self.network.add(self.mc_pop)
            self.neuron_groups_list.append(self.mc_pop)

    def _define_synaptic_populations(self):
        ''' Define the synaptic populations '''

        self.synaptic_groups_list : list[b2.Synapses] = []

        silencing = self.params.simulation.include_silencing
        weighted  = self.params.simulation.include_syn_weight

        ### NEURON_GROUP_0 --> NEURON_GROUP_0
        # Equations
        self.seq_ex1 = synaptic_models.define_syn_equation(
            self.params.synapses.synaptic_labels_ex,
            silencing = silencing,
            weighted  = weighted,
        )

        self.seq_in1 = synaptic_models.define_syn_equation(
            self.params.synapses.synaptic_labels_in,
            silencing = silencing,
            weighted  = weighted,
        )

        self.on_pre_ex1 = synaptic_models.syn_on_pre(
            1,
            self.params.synapses.synaptic_labels_ex,
            silencing = silencing,
            weighted  = weighted
        )

        self.on_pre_in1 = synaptic_models.syn_on_pre(
            1,
            self.params.synapses.synaptic_labels_in,
            silencing = silencing,
            weighted  = weighted
        )

        # Synaptic groups
        if 'syn_ex' not in self.excluded_synaptic_groups_names:
            self.syn_ex = b2.Synapses(
                self.pop,
                self.pop,
                self.seq_ex1,
                self.on_pre_ex1,
                method = self.params.simulation.int_method,
                name   = 'syn_ex'
            )
            self.network.add(self.syn_ex)
            self.synaptic_groups_list.append(self.syn_ex)

        if 'syn_in' not in self.excluded_synaptic_groups_names:
            self.syn_in = b2.Synapses(
                self.pop,
                self.pop,
                self.seq_in1,
                self.on_pre_in1,
                method = self.params.simulation.int_method,
                name   = 'syn_in'
            )
            self.network.add(self.syn_in)
            self.synaptic_groups_list.append(self.syn_in)

        # Placeholder for the connectivity matrix
        self.wmat = np.array([], dtype= int)

        ### NEURON_GROUP_0 --> NEURON_GROUP_1
        if self.params.topology.include_muscle_cells:

            # Equations
            self.seq_mc_ex = synaptic_models.define_syn_equation(
                self.params.synapses.synaptic_labels_ex,
                silencing = silencing,
                weighted  = weighted
            )

            self.on_pre_mc_ex = synaptic_models.syn_on_pre(
                '_mc',
                self.params.synapses.synaptic_labels_ex,
                silencing = silencing,
                weighted  = weighted
            )

            self.seq_mc_in = synaptic_models.define_syn_equation(
                self.params.synapses.synaptic_labels_in,
                silencing = silencing,
                weighted  = weighted
            )

            self.on_pre_mc_in = synaptic_models.syn_on_pre(
                '_mc',
                self.params.synapses.synaptic_labels_in,
                silencing = silencing,
                weighted  = weighted
            )

            # Synaptic groups
            if 'syn_mc_ex' not in self.excluded_synaptic_groups_names:
                self.syn_mc_ex = b2.Synapses(
                    self.pop,
                    self.mc_pop,
                    self.seq_mc_ex,
                    self.on_pre_mc_ex,
                    method = self.params.simulation.int_method,
                    name   = 'syn_mc_ex'
                )
                self.network.add(self.syn_mc_ex)
                self.synaptic_groups_list.append(self.syn_mc_ex)

            if 'syn_mc_in' not in self.excluded_synaptic_groups_names:
                self.syn_mc_in = b2.Synapses(
                    self.pop,
                    self.mc_pop,
                    self.seq_mc_in,
                    self.on_pre_mc_in,
                    method = self.params.simulation.int_method,
                    name   = 'syn_mc_in'
                )
                self.network.add(self.syn_mc_in)
                self.synaptic_groups_list.append(self.syn_mc_in)

            # Placeholder for the connectivity matrix
            self.wmat_mc = np.array([], dtype= int)

    def _define_monitors(self) -> None:
        ''' Define the monitors of the simulations and add them to the network '''

        monitor_spikes = self.params.monitor.spikes
        monitor_states = self.params.monitor.states
        monitor_mcells = self.params.monitor.muscle_cells

        # Create new monitor objects
        if monitor_spikes['active']:
            self.spikemon = b2.SpikeMonitor(
                self.pop,
                record=monitor_spikes['indices']
            )
            self.network.add(self.spikemon)

        if monitor_states['active']:
            self.statemon = b2.StateMonitor(
                self.pop,
                monitor_states['variables'],
                dt     = monitor_states['rate'],
                record = monitor_states['indices'],
            )
            self.network.add(self.statemon)

        if monitor_mcells['active']:
            self.musclemon = b2.StateMonitor(
                self.mc_pop,
                monitor_mcells['variables'],
                dt     = monitor_mcells['rate'],
                record = monitor_mcells['indices'],
            )
            self.network.add(self.musclemon)

    def _define_callback(self) -> None:
        ''' Defines the callback function and the corresponding parameters '''

        if not self.params.simulation.include_callback:
            self.callback = None
            return

        self.callback_clock = b2.Clock(
            self.params.simulation.callback_dt,
            name= 'callback_clock'
        )
        # NOTE: It will be executed after the monitor update (order = 1)
        self.callback = b2.NetworkOperation(
            self.step_function,
            clock = self.callback_clock,
            name  = 'callback',
            order = 1,
        )
        self.network.add(self.callback)

        topology = self.params.topology
        mechanics = self.params.mechanics
        net_modules = topology.network_modules

        # FEEDFORWARD
        if self.params.topology.include_muscle_cells:

            self.output_torques = np.zeros(net_modules['mc'].n_tot)
            self.motor_output = np.zeros(
                + 2 * mechanics.mech_axial_joints
                + 2 * mechanics.mech_limbs_joints
            )

        # FEEDBACK
        if self.params.topology.include_proprioception:
            self.input_thetas      = np.zeros(mechanics.mech_axial_joints)
            self.input_thetas_ps   = np.zeros(topology.segments_axial * net_modules['ps'].n_pool )

        return

    def _define_network_objects(self) -> None:
        '''
        Inizialize network objects
        '''
        self.network = b2.Network(name= 'salamandrasnn_network')

        self._define_neuronal_populations()
        self._define_synaptic_populations()
        self._define_monitors()
        self._define_callback()

        # Define scheduling
        self.network.schedule = ['start', 'thresholds', 'resets', 'synapses', 'groups', 'end']
        return

    def _delete_network_obj(self, obj, obj_list, condition=True, *args):
        ''' Delete a network object '''

        if not condition or not obj in self.network.objects:
            return

        # Remove the object from the network
        self.network.remove(obj)

        # Remove the object from the list
        if obj in obj_list:
            obj_list.remove(obj)

        # Delete the object
        del obj
        for arg in args:
            del arg

        return

    def _delete_neuronal_populations(self):
        ''' Define the neuronal populations '''
        include_mc = self.params.topology.include_muscle_cells
        self._delete_network_obj(   self.pop, self.neuron_groups_list)
        self._delete_network_obj(self.mc_pop, self.neuron_groups_list, include_mc)
        return

    def _delete_synaptic_populations(self):
        ''' Delete the synaptic populations '''
        include_mc = self.params.topology.include_muscle_cells
        self._delete_network_obj(   self.syn_ex, self.synaptic_groups_list)
        self._delete_network_obj(   self.syn_in, self.synaptic_groups_list)
        self._delete_network_obj(self.syn_mc_ex, self.synaptic_groups_list, include_mc)
        self._delete_network_obj(self.syn_mc_in, self.synaptic_groups_list, include_mc)
        return

    def _delete_monitors(self) -> None:
        ''' Delete the monitors of the simulations and add them to the network '''

        monitor_spikes = self.params.monitor.spikes
        monitor_states = self.params.monitor.states
        monitor_mcells = self.params.monitor.muscle_cells

        # Create new monitor objects
        if monitor_spikes['active']:
            self.network.remove(self.spikemon)
            del self.spikemon

        if monitor_states['active']:
            self.network.remove(self.statemon)
            del self.statemon

        if monitor_mcells['active']:
            self.network.remove(self.musclemon)
            del self.musclemon

        return

    def _delete_callback(self) -> None:
        ''' Delete the callback function and the corresponding parameters '''

        if self.params.simulation.include_callback:
            self.network.remove(self.callback)
            del self.callback_clock
            del self.callback

        return

    def _delete_network_objects(self) -> None:
        '''
        Inizialize network objects
        '''
        self._delete_neuronal_populations()
        self._delete_synaptic_populations()
        self._delete_monitors()
        self._delete_callback()
        del self.network
        return

    def _redefine_network_objects(self) -> None:
        '''
        Inizialize network objects
        '''
        logging.warning('Re-defining network objects')
        self._delete_network_objects()
        self._define_network_objects()
        return

    ### COMMUNICATION
    def delete_queues(self):
        ''' Delete queues '''
        del self.q_in
        del self.q_out

    def define_queues(self):
        ''' Define queues '''
        self.q_in  = Queue()
        self.q_out = Queue()

    def queue_handshake(self):
        ''' Exchange handshake '''
        self.q_out.put(True)
        self.q_in.get(
            block   = True,
            timeout = 600,
        )

    ### SAVE/LOAD NEWORK STATE
    def save_network_state(self) -> None:
        ''' Saves the current state of all the network objects '''
        filepath = self.params.simulation.results_data_folder_sub_process
        os.makedirs(filepath, exist_ok=True)
        filename = f'{filepath}/network_state.brian'
        logging.info('Saving network state in %s', filename)
        self.network.store(filename= filename)
        logging.info('Saved network state')

    def load_network_state(self) -> None:
        ''' Loads and assigns the state of all the network objects '''
        filepath = self.params.simulation.results_data_folder_sub_process
        filename = f'{filepath}/network_state.brian'
        logging.info('Loading network state from %s', filename)
        self.network.restore(filename= filename, restore_random_state= True)

        # Mark synaptic groups as connected
        for syn in self.synaptic_groups_list:
            if not len(syn):
                continue
            syn._connect_called = True

        logging.info('Loaded network state')
        return True

    def found_saved_network_state(self) -> bool:
        filepath = self.params.simulation.results_data_folder_sub_process
        filename = f'{filepath}/network_state.brian'
        return os.path.isfile(filename)

    ### Functionalities
    def _set_seed(self):
        '''
        Set the seed for the random number generators
        NOTE: Both seeds must be set to guarantee repeatability
        '''

        if not self.seed_assigned:
            self.seed_val = (
                self.params.simulation.seed_value
                if self.params.simulation.set_seed
                else np.random.randint(0, 1e9)
            )
            self.seed_assigned = True
            logging.info('Seed = %i', self.seed_val)

        # Brian2 seed
        b2.seed(self.seed_val)

        # Python seed
        random.seed(self.seed_val)

        # Numpy seed
        self.randstate = np.random.RandomState(
            np.random.MT19937(
                np.random.SeedSequence(self.seed_val)
            )
        )

        return

    def update_network_parameters_from_dict(self, new_pars: dict) -> None:
        '''
        Update targeted fields in the network from an input dictionary.
        NOTE: Scan hierarchically to ensure that the latest setter definition is used
        '''
        objects_to_scan = [self, self.params] + self.params.parameters_objects_list

        for key in list(new_pars.keys()):
            for obj in objects_to_scan:
                if is_writeable_property(obj, key):
                    setattr(obj, key, new_pars.pop(key))
                    break

        assert new_pars == {}, f'Cannot assign all parameters:\n {new_pars}'

    #### Callback method (virtual, model-specific)
    def step_function(self, _curtime):
        '''
        When used (include_callback = True), called at every integration step.\n
        Exchanges data with external thread via input and output queues.
        '''

    ### Data retrieval
    def get_parameters(self) -> dict:
        ''' Return the parameters of the simulation '''
        return self.params.get_parameters()

    def get_variable_parameters(self) -> dict:
        ''' Method to collect the variable parameters of the system '''
        return self.params.get_variable_parameters()

# TEST
def main():
    ''' Test case '''
    logging.info('TEST: SNN Core ')

    core = SnnCore(
        network_name = 'snn_core_test',
        params_name  = 'pars_simulation_test',
        q_in         = Queue(),
        q_out        = Queue(),
    )

    return core

if __name__ == '__main__':
    main()
