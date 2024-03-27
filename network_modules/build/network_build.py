'''
Functions to specify the connectivity within the network
'''
import logging
import copy
import numpy as np
import numexpr as ne

import brian2 as b2
from network_modules.equations import parameter_setting
from network_modules.parameters.network_module import SnnNetworkModule
from network_modules.connectivity.network_connectivity_axis import SnnConnectivityAxis
from network_modules.connectivity.network_connectivity_limbs import SnnConnectivityLimbs

from network_modules.equations import parameter_setting

class SnnBuild(SnnConnectivityAxis, SnnConnectivityLimbs):
    '''
    Class to build and initialize the networks for the simulations
    '''

    ### TOPOLOGY
    def define_network_topology(self) -> None:
        '''
        Define the network's topology and variable values
        '''

        # Neuronal identifiers
        self._define_neuronal_identifiers()

        # Connections
        self._define_connectivity()

        # Assign internal neuronal and synaptic parameters
        self._assign_network_parameters()

        # Varying weights depending on connected pools
        self._assign_synaptic_weights()

        # Initialize variables
        self._initialize_variables()

        # Save initialized network state
        self.save_network_state()

        logging.info('Defined newtwork topology')

    ### CONNECTIVITY
    def _define_connectivity(self) -> None:
        '''
        Define the overall network connectivity.
        Modulates the connectivity based on the selected gait.
        Optionally save the connectivity matrices.
        '''

        # Set seed again for repeatability
        self._set_seed()

        # Define from configuration files or from matrices
        if self.params.simulation.load_connectivity_indices:
            self.define_connectivity_from_indices()
        else:
            self.define_axial_connectivity_from_yaml()
            self.define_limbs_connectivity_from_yaml()
            self.define_connectivity_indices()
            self.save_connectivity_indices()

        # Remove unused synaptic groups from the network
        self._remove_unused_synaptic_groups()

        # Compute shuffled synaptic indices (fod gait transitions)
        self._compute_shuffled_syn_indices()

        # Modulate connectivity based on current gait
        self.update_limb_connectivity()

        return

    ### WEIGHTS
    def _assign_synaptic_weights_by_list(
        self,
        syn_weights_list: list[dict],
        std_val         : float,
    ) -> None:
        ''' Assign synaptic weights by list of parameters '''

        net_modules = self.params.topology.network_modules

        for syn_weights_pars in syn_weights_list:
            extra_cond = syn_weights_pars.get('extra_cond', None)
            source_mod = net_modules.get_sub_module_from_full_name(syn_weights_pars['source_name'])
            target_mod = net_modules.get_sub_module_from_full_name(syn_weights_pars['target_name'])

            if not source_mod.include or not target_mod.include:
                continue

            for syn in self.synaptic_groups_list:

                # Check if connected groups are matching
                source_group = self.neuron_groups_list[source_mod.neuron_group]
                target_group = self.neuron_groups_list[target_mod.neuron_group]

                if not syn.source == source_group or not syn.target == target_group:
                    continue

                # Assign weights
                parameter_setting.set_synaptic_parameters_by_neural_ind_ranges(
                    syn_group         = syn,
                    inds_limits_syn_i = source_mod.indices_limits,
                    inds_limits_syn_j = target_mod.indices_limits,
                    std_value         = std_val,
                    parameters        = syn_weights_pars,
                    extra_cond        = extra_cond,
                )

    def _assign_synaptic_weights(self) -> None:
        ''' Different weight scaling depending on the connected pools '''

        # Set seed again for repeatability
        self._set_seed()

        # Assign weights defined by the configuration files
        self._assign_synaptic_weights_by_list(
            syn_weights_list = self.params.synapses.syn_weights_list,
            std_val          = self.params.neurons.std_val,
        )

    ### PARAMETERS
    def _assign_network_parameters(self) -> None:
        ''' Assign internal neural and synaptic parameters '''

        # Set seed again for repeatability
        self._set_seed()

        std_val     = self.params.neurons.std_val
        net_modules = self.params.topology.network_modules

        # Shared neuronal params
        for pars_group in self.params.neurons.shared_neural_params:
            neuron_group_ind = pars_group['neuron_group']

            if neuron_group_ind >= len(self.neuron_groups_list):
                continue

            neuron_group = self.neuron_groups_list[neuron_group_ind]
            parameter_setting.set_neural_parameters_by_neural_inds(
                ner_group  = neuron_group,
                inds_ner   = range(neuron_group.N),
                std_value  = std_val,
                parameters = pars_group
            )

        # Variable neuronal params
        for pars_mod in self.params.neurons.variable_neural_params_list:
            target_mod = net_modules.get_sub_module_from_full_name(pars_mod['mod_name'])

            if not target_mod.include:
                continue

            neuron_group_ind = target_mod.neuron_group
            ind_min, ind_max = target_mod.indices_limits

            parameter_setting.set_neural_parameters_by_neural_inds(
                ner_group  = self.neuron_groups_list[neuron_group_ind],
                inds_ner   = range(ind_min, ind_max+1),
                std_value  = std_val,
                parameters = pars_mod
            )

        # Shared synaptic parameters
        for pars_group in self.params.synapses.shared_syn_params:
            neuron_group_ind = pars_group['neuron_group']

            if neuron_group_ind >= len(self.neuron_groups_list):
                continue

            neuron_group     = self.neuron_groups_list[neuron_group_ind]
            parameter_setting.set_neural_parameters_by_neural_inds(
                ner_group  = neuron_group,
                inds_ner   = range(neuron_group.N),
                std_value  = std_val,
                parameters = pars_group
            )

        # Variable synaptic parameters
        for pars_group in self.params.synapses.variable_syn_params:
            neuron_group_ind = pars_group['neuron_group']

            if neuron_group_ind >= len(self.neuron_groups_list):
                continue

            neuron_group     = self.neuron_groups_list[neuron_group_ind]
            parameter_setting.set_neural_parameters_by_neural_inds(
                ner_group  = neuron_group,
                inds_ner   = range(neuron_group.N),
                std_value  = std_val,
                parameters = pars_group
            )

    ### DRIVE
    def assign_drives(self) -> None:
        '''
        Defines the external input to the reticulospinal neurons
        '''
        # Excitation to reticulospinal neurons
        sim_pars    = self.params.simulation
        top_pars    = self.params.topology
        drv_pars    = self.params.drive
        gait_drives = self.params.drive.gait_drives

        module_rs = top_pars.network_modules['rs']

        axis_mul = sim_pars.stim_a_mul
        limb_mul = sim_pars.stim_l_mul

        axis_off = sim_pars.stim_a_off
        limb_off = sim_pars.stim_l_off

        l_mul = sim_pars.stim_l_multiplier
        r_mul = sim_pars.stim_r_multiplier
        f_mul = sim_pars.stim_f_multiplier
        e_mul = sim_pars.stim_e_multiplier

        l_off = sim_pars.stim_lr_off
        f_off = sim_pars.stim_fe_off

        network_current = getattr(self.pop, 'I_ext')

        # Axis
        if top_pars.include_reticulospinal_axial:

            inds_rs_axial_pools_l = module_rs['axial'].indices_pools_sides[0]
            inds_rs_axial_pools_r = module_rs['axial'].indices_pools_sides[1]

            drives_ax = sim_pars.gains_drives_axis * gait_drives['drives_axis'] * axis_mul

            tot_offset_l = axis_off + l_off
            tot_offset_r = axis_off

            tot_curr_l = (drives_ax + tot_offset_l) * l_mul * b2.pamp
            tot_curr_r = (drives_ax + tot_offset_r) * r_mul * b2.pamp

            for pool_ind_ax in range(module_rs.axial.pools):

                rs_inds_l = inds_rs_axial_pools_l[pool_ind_ax]
                rs_inds_r = inds_rs_axial_pools_r[pool_ind_ax]

                network_current[rs_inds_l] = tot_curr_l[pool_ind_ax]
                network_current[rs_inds_r] = tot_curr_r[pool_ind_ax]

        # Limbs
        if top_pars.include_reticulospinal_limbs:

            inds_rs_limbs_pools_f = module_rs['limbs'].indices_pools_sides[0]
            inds_rs_limbs_pools_e = module_rs['limbs'].indices_pools_sides[1]

            drives_lb = sim_pars.gains_drives_limbs * gait_drives['drives_limbs'] * limb_mul

            def _limb_current(d_drive_lb, l_drive_lb, flex: bool, side: int):
                ''' Current to provide to the limb '''
                lr_mul = l_mul if side == 0 else r_mul   # Left/Right multiplier
                lr_off = l_off if side == 0 else 0       # Left/Right offset
                fe_mul = f_mul if flex == 1 else e_mul   # Flexor/Extensor multiplier
                fe_off = f_off if flex == 1 else 0       # Flexor/Extensor offset
                tot_offset = limb_off + lr_off + fe_off + d_drive_lb + l_drive_lb
                return ( drive_lb + tot_offset ) * lr_mul * fe_mul * b2.pamp

            # Offset drives for flexor and extensor of LF, RF, LH, RH
            d_drives = drv_pars.dfac * np.array( [ [1, 0], [3, 2], [4, 3], [2, 1] ] )
            l_drives = drv_pars.lfac * np.array( [ [4, 3], [2, 1], [1, 0], [3, 2] ] )

            for pool_ind_lb, (d_drive, l_drive, drive_lb) in enumerate( zip(d_drives, l_drives, drives_lb)):

                rs_inds_f = inds_rs_limbs_pools_f[pool_ind_lb]
                rs_inds_e = inds_rs_limbs_pools_e[pool_ind_lb]

                network_current[rs_inds_f] = _limb_current(d_drive[0], l_drive[0], flex= 1, side= pool_ind_lb%2)
                network_current[rs_inds_e] = _limb_current(d_drive[1], l_drive[1], flex= 0, side= pool_ind_lb%2)

        if top_pars.include_motor_neurons_limbs:
            params_mn = top_pars.network_modules['mn']

            inds_mn_limbs_pools_f = params_mn['limbs'].indices_pools_sides[0]
            inds_mn_limbs_pools_e = params_mn['limbs'].indices_pools_sides[1]

            for inds_pool_f, inds_pool_e in zip(inds_mn_limbs_pools_f, inds_mn_limbs_pools_e):
                network_current[ inds_pool_f ] = sim_pars.stim_f_mn_off * b2.pamp
                network_current[ inds_pool_e ] = sim_pars.stim_e_mn_off * b2.pamp

        return

    ### INITIALIZATION
    def _initialize_variables(self) -> None:
        ''' Initialize internal variables of the neural and synaptic models '''

        # Set seed again for repeatability
        self._set_seed()

        # Starting point of the simulation
        self.initial_time = self.network.t

        # Neuronal model
        for parameter, initial_value in self.neuronal_initial_values.items():
            setattr(self.pop, parameter, initial_value)

        self.syn_ex.delay = self.params.synapses.syndel
        self.syn_in.delay = self.params.synapses.syndel

        # Muscle cells
        if self.params.topology.include_muscle_cells:

            for parameter, initial_value in self.muscle_initial_values.items():
                setattr(self.mc_pop, parameter, initial_value)

            if len(self.syn_mc_ex)>0:
                self.syn_mc_ex.delay = self.params.synapses.syndel
            if len(self.syn_mc_in)>0:
                self.syn_mc_in.delay = self.params.synapses.syndel

        # Excitation to reticulospinal neurons
        self.assign_drives()

    ### UTILS
    def _remove_unused_synaptic_groups(self):
        '''
        Removes synaptic groups with no connections
        ['syn_ex', 'syn_in', 'syn_mc_ex', 'syn_mc_in']
        '''
        removed_list = []
        for syn in self.synaptic_groups_list:
            if syn is None or len(syn)==0:
                self.network.remove(syn)
                removed_list.append(syn)

        # Update synaptic groups
        self.excluded_synaptic_groups_names += [syn.name for syn in removed_list]
        self.synaptic_groups_list = [
            syn
            for syn in self.synaptic_groups_list
            if syn not in removed_list
        ]
        return

    def _compute_shuffled_syn_indices_interlimb(self) -> None:
        '''
        Computes the indices of the synapses linking the limbs and stores
        them shuffled (allows for repeatable gait changes)
        '''

        # Set seed again for repeatability
        self._set_seed()

        # Silencing
        inds_flx = self.params.topology.network_modules['cpg']['limbs'].indices_pools_sides[0]
        inds_ext = self.params.topology.network_modules['cpg']['limbs'].indices_pools_sides[1]

        # Auxiliary functions
        def _find_and_shuffle_synapses(
            syn: b2.Synapses,
            inds_i: list,
            inds_j: list) -> np.ndarray:
            '''
            Finds indices of synapses linking indices in inds_i to indices in inds_j
            Returns shuffled synaptic indices
            '''
            target_syn_inds = np.intersect1d(
                np.where(np.isin(syn.i, inds_i))[0],
                np.where(np.isin(syn.j, inds_j))[0]
            )
            return self.randstate.permutation(target_syn_inds)

        def _get_all_shuffled_synapses(
            syn: b2.Synapses,
            inds_source: list[list],
            inds_target: list[list]) -> list[np.ndarray] :
            '''
            Returns shuffled synaptic indices as a 2D list representing connections
            between the ind_source and inds_target
            '''
            return [
                [
                    _find_and_shuffle_synapses(syn, source_i, target_j)
                    for target_j in inds_target
                ]
                for source_i in inds_source
            ]

        # Excitatory
        self.params.simulation.interlimb_syn_inds_ex_shuffled = {
            'ex_f2f' : _get_all_shuffled_synapses(self.syn_ex, inds_flx, inds_flx),
            'ex_f2e' : _get_all_shuffled_synapses(self.syn_ex, inds_flx, inds_ext),
            'ex_e2f' : _get_all_shuffled_synapses(self.syn_ex, inds_ext, inds_flx),
            'ex_e2e' : _get_all_shuffled_synapses(self.syn_ex, inds_ext, inds_ext),
        }

        # Inhibitory
        self.params.simulation.interlimb_syn_inds_in_shuffled = {
            'in_f2f' : _get_all_shuffled_synapses(self.syn_in, inds_flx, inds_flx),
            'in_f2e' : _get_all_shuffled_synapses(self.syn_in, inds_flx, inds_ext),
            'in_e2f' : _get_all_shuffled_synapses(self.syn_in, inds_ext, inds_flx),
            'in_e2e' : _get_all_shuffled_synapses(self.syn_in, inds_ext, inds_ext),
        }

        return

    def _compute_shuffled_syn_indices(self) -> None:
        '''
        Computes the indices of the synapses and stores
        them shuffled (allows for repeatable gait changes)
        '''

        # Set seed again for repeatability
        self._set_seed()

        # Compute for every synaptic group
        self.params.simulation.syn_inds_shuffled = {
            syn.name : self.randstate.permutation(len(syn))
            for syn in self.synaptic_groups_list
        }

        # Compute for inter-limb connections
        self._compute_shuffled_syn_indices_interlimb()

        return


    ## FUNCTIONALITIES
    def update_limb_connectivity(self) -> None:
        '''
        Modify connectivity based on the selected gait.
        '''
        logging.info('Modulating limb connectivity for %s pattern', self.params.simulation.gait)

        ## Adjust inter-limb connections
        gait = self.params.simulation.gait

        p_ex_max     = self.params.topology.pars_limb_conn.p_ex_inter_lb_max
        p_in_max     = self.params.topology.pars_limb_conn.p_in_inter_lb_max
        p_conn_limbs = self.params.topology.pars_limb_conn.p_connections_limbs

        # Silence connections
        ex_active = np.array([])
        ex_silent = np.array([])
        in_active = np.array([])
        in_silent = np.array([])

        for i in range(self.params.topology.limbs):
            for j in range(self.params.topology.limbs):

                if i == j:
                    continue

                # Excitatory
                if p_ex_max:
                    for conn_type, syn_inds in self.params.simulation.interlimb_syn_inds_ex_shuffled.items():

                        if not syn_inds:
                            continue

                        if p_conn_limbs[gait][conn_type] is None:
                            p_ex = 0
                        else:
                            p_ex = p_conn_limbs[gait][conn_type][i][j]

                        silence_ratio_ex = 1 - p_ex / p_ex_max
                        n_silence_ex = round( silence_ratio_ex * len(syn_inds[i][j]) )

                        ex_silent = np.concatenate( [ex_silent, syn_inds[i][j][:n_silence_ex] ] )
                        ex_active = np.concatenate( [ex_active, syn_inds[i][j][n_silence_ex:] ] )

                # Inhibitory
                if p_in_max:
                    for conn_type, syn_inds in self.params.simulation.interlimb_syn_inds_in_shuffled.items():

                        if not syn_inds:
                            continue

                        if p_conn_limbs[gait][conn_type] is None:
                            p_in = 0
                        else:
                            p_in = p_conn_limbs[gait][conn_type][i][j]

                        silence_ratio_in = 1 - p_in / p_in_max
                        n_silence_in = round( silence_ratio_in * len(syn_inds[i][j]) )

                        in_silent = np.concatenate( [in_silent,  syn_inds[i][j][:n_silence_in] ] )
                        in_active = np.concatenate( [in_active,  syn_inds[i][j][n_silence_in:] ] )

        self.toggle_syn_silencing_by_syn_inds(self.syn_ex, ex_silent, silenced= True )
        self.toggle_syn_silencing_by_syn_inds(self.syn_ex, ex_active, silenced= False)
        self.toggle_syn_silencing_by_syn_inds(self.syn_in, in_silent, silenced= True )
        self.toggle_syn_silencing_by_syn_inds(self.syn_in, in_active, silenced= False)
        return

    def modulate_connectivity(
        self,
        syn_group      : b2.Synapses,
        syn_modulation: float,
        module_src    : SnnNetworkModule,
        module_trg    : SnnNetworkModule,
        **kwargs
    ) -> None:
        '''
        Modify connectivity between source and target network modules.

        This function modulates the connectivity between a source and a target network module by adjusting the strength of synaptic connections based on the specified modulation factor.

        Args:
            syn_group (b2.Synapses): The synaptic group to be modulated.
            syn_modulation (float): The factor by which to modulate the connectivity. A value of 1.0 maintains the original connectivity, values greater than 1.0 enhance connectivity, and values less than 1.0 reduce connectivity.
            module_src (SnnNetworkModule): The source network module.
            module_trg (SnnNetworkModule): The target network module.
            **kwargs: Additional keyword arguments for specifying indices and other options.

        Returns:
            None
        '''
        logging.info(
            'Modulating connectivity between %s and %s',
            module_src.name,
            module_trg.name,
        )

        i_copies : list[int] = kwargs.get('i_copies', None)
        i_sides  : list[int] = kwargs.get('i_sides', None)
        i_pools  : list[int] = kwargs.get('i_pools', None)

        j_copies : list[int] = kwargs.get('j_copies', None)
        j_sides  : list[int] = kwargs.get('j_sides', None)
        j_pools  : list[int] = kwargs.get('j_pools', None)

        def __get_indices(
                module: SnnNetworkModule,
                copies: list[int],
                sides : list[int],
                pools: list[int],
            ):
            '''
            Get indices for the given network module based on specified parameters.

            This internal function calculates and returns indices for a given network module based on provided parameters for copies, sides, and pools.
            It considers the priority of different iterators and combinations to determine the appropriate indices.

            Args:
                module (SnnNetworkModule): The network module for which indices are being obtained.
                copies (list[int]): List of copy indices to consider. Set to None if not used.
                sides (list[int]): List of side indices to consider. Set to None if not used.
                pools (list[int]): List of pool indices to consider. Set to None if not used.

            Returns:
                np.ndarray: Flattened array of indices for the network module based on specified parameters.
            '''

            available_iterators_priority_and_keys = {
                'copies': [1, copies],
                'sides' : [2, sides],
                'pools' : [3, pools],
            }
            available_indices_and_priority = {
                'all'               : [1, module.indices],
                'copies'            : [2, module.indices_copies],
                'sides'             : [2, module.indices_sides],
                'pools'             : [2, module.indices_pools],
                'sides_copies'      : [3, module.indices_sides_copies],
                'pools_copies'      : [3, module.indices_pools_copies],
                'pools_sides'       : [3, module.indices_pools_sides],
                'pools_sides_copies': [4, module.indices_pools_sides_copies],
            }

            if pools is None:
                # Remove pools from available iterators
                available_iterators_priority_and_keys.pop('pools')
                available_indices_and_priority.pop('pools', None)
                available_indices_and_priority.pop('pools_copies', None)
                available_indices_and_priority.pop('pools_sides', None)
                available_indices_and_priority.pop('pools_sides_copies', None)

            if copies is None:
                # Remove copies from available iterators
                available_iterators_priority_and_keys.pop('copies')
                available_indices_and_priority.pop('copies', None)
                available_indices_and_priority.pop('sides_copies', None)
                available_indices_and_priority.pop('pools_copies', None)
                available_indices_and_priority.pop('pools_sides_copies', None)

            if sides is None:
                # Remove sides from available iterators
                available_iterators_priority_and_keys.pop('sides')
                available_indices_and_priority.pop('sides', None)
                available_indices_and_priority.pop('sides_copies', None)
                available_indices_and_priority.pop('pools_sides', None)
                available_indices_and_priority.pop('pools_sides_copies', None)

            # Select indices with the highest priority
            indices_key = max(
                available_indices_and_priority,
                key = lambda k: available_indices_and_priority[k][0]
            )
            indices_all : np.ndarray = available_indices_and_priority[indices_key][1]

            # Get list of iterators, sorted based on priority
            used_keys = sorted(
                available_iterators_priority_and_keys,
                key = lambda k: available_iterators_priority_and_keys[k][0]
            )

            # Return flattened indices
            n_keys = len(used_keys)

            if n_keys == 0:
                # No iterators
                indices_selected = indices_all

            if n_keys == 1:
                # 1-D iterator
                iterator_0 = available_iterators_priority_and_keys[used_keys[0]][1]
                indices_selected = indices_all[iterator_0]

            if n_keys == 2:
                # 2-D iterator
                iterator_0 = available_iterators_priority_and_keys[used_keys[0]][1]
                iterator_1 = available_iterators_priority_and_keys[used_keys[1]][1]
                indices_selected = indices_all[iterator_0, iterator_1]

            if n_keys == 3:
                # 3-D iterator
                iterator_0 = available_iterators_priority_and_keys[used_keys[0]][1]
                iterator_1 = available_iterators_priority_and_keys[used_keys[1]][1]
                iterator_2 = available_iterators_priority_and_keys[used_keys[2]][1]
                indices_selected = indices_all[iterator_0, iterator_1, iterator_2]

            return indices_selected.flatten()

        # Get indices for source and target modules
        inds_i = __get_indices(module_src, i_copies, i_sides, i_pools)
        inds_j = __get_indices(module_trg, j_copies, j_sides, j_pools)

        # Finds indices of synapses linking indices in inds_i to indices in inds_j
        target_syn_inds = np.intersect1d(
            np.where(np.isin( syn_group.i, inds_i))[0],
            np.where(np.isin( syn_group.j, inds_j))[0]
        )

        # Find indices of those connections in the shuffled indices
        syn_inds_shuffled = self.params.simulation.syn_inds_shuffled[syn_group.name]
        target_syn_inds_shuffled = np.where(np.isin(syn_inds_shuffled, target_syn_inds))[0]

        # Get number of connections
        n_connections           = len(target_syn_inds)
        n_connections_modulated = round(syn_modulation * n_connections)

        # Set the silenced field of the target indices
        self.toggle_syn_silencing_by_syn_inds(syn_group, syn_inds_shuffled[ target_syn_inds_shuffled[:n_connections_modulated] ], silenced= False)
        self.toggle_syn_silencing_by_syn_inds(syn_group, syn_inds_shuffled[ target_syn_inds_shuffled[n_connections_modulated:] ], silenced= True)

        return


# TEST
def main():
    ''' Test case '''

    import matplotlib.pyplot as plt
    from queue import Queue
    import network_modules.plotting.plots_snn as snn_plotting

    logging.info('TEST: SNN Build ')

    build = SnnBuild(
        network_name = 'snn_core_test',
        params_name  = 'pars_simulation_test',
        q_in         = Queue(),
        q_out        = Queue(),
    )

    build.define_network_topology()

    # Plotting
    sub_modules_0 = build.params.topology.network_leaf_modules[0]
    sub_modules_1 = build.params.topology.network_leaf_modules[1]

    plt.figure('Connectivity matrix')
    build.get_wmat()
    snn_plotting.plot_connectivity_matrix(
        pop_i                  = build.pop,
        pop_j                  = build.pop,
        w_syn                  = build.wmat,
        network_modules_list_i = sub_modules_0,
        network_modules_list_j = sub_modules_0,
    )

    plt.figure('Connectivity matrix - Muscle cells')
    build.get_wmat_mc()
    snn_plotting.plot_connectivity_matrix(
        pop_i                  = build.pop,
        pop_j                  = build.mc_pop,
        w_syn                  = build.wmat_mc,
        network_modules_list_i = sub_modules_0,
        network_modules_list_j = sub_modules_1,
    )

    plt.figure('Neurons Identifiers')
    snn_plotting.plot_neuronal_identifiers(
        pop                  = build.pop,
        network_modules_list = sub_modules_0,
        identifiers_list     = [
            'i',
            'ner_id',
            'side_id',
            'y_neur',
            'pool_id',
            'limb_id',
        ]
    )

    plt.show()

    return build

if __name__ == '__main__':
    main()
