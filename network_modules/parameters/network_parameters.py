'''
Parameters for the network simulation
This module is intended as a template to be extended for every specific implementation
'''

import os
import sys
import dill
import copy
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import copy
import logging
import numpy as np

from typing import Union, Callable

from network_modules.parameters.pars_utils      import SnnPars
from network_modules.parameters.pars_simulation import SnnParsSimulation
from network_modules.parameters.pars_topology   import SnnParsTopology
from network_modules.parameters.pars_neurons    import SnnParsNeurons
from network_modules.parameters.pars_synapses   import SnnParsSynapses
from network_modules.parameters.pars_drive      import SnnParsDrive
from network_modules.parameters.pars_mechanics  import SnnParsMechanics
from network_modules.parameters.pars_monitor    import SnnParsMonitor

class SnnParameters():
    '''
    Class used to store and retrieve parameters and
    hyperparameters regarding the simulation
    '''

    def __init__(
        self,
        parsname    : str,
        results_path: str,
        new_pars    : dict = None,
        **kwargs
    ) -> None:
        '''
        Core hyperparameters of the simulation
        '''

        pars_path = kwargs.pop('pars_path', None)

        if new_pars is None:
            new_pars = {}

        params_to_update = new_pars | kwargs

        # Simulation
        self.simulation = SnnParsSimulation(
            parsname     = parsname,
            results_path = results_path,
            new_pars     = params_to_update,
            pars_path    = pars_path,
        )

        # Topology
        self.topology = SnnParsTopology(
            parsname  = self.simulation.pars_topology_filename,
            new_pars  = params_to_update,
            pars_path = pars_path,
        )

        # Drives
        self.drive = SnnParsDrive(
            parsname  = self.simulation.pars_drive_filename,
            sim_pars  = self.simulation,
            top_pars  = self.topology,
            new_pars  = params_to_update,
            pars_path = pars_path,
        )

        # Neuronal models
        self.neurons = SnnParsNeurons(
            parsname  = self.simulation.pars_neurons_filename,
            new_pars  = params_to_update,
            pars_path = pars_path,
        )

        # Synaptic models
        self.synapses = SnnParsSynapses(
            parsname  = self.simulation.pars_synapses_filename,
            new_pars  = params_to_update,
            pars_path = pars_path,
        )

        # Mechanics
        # TODO: Avoid getting mechanics parameters when the simulation is performed in open loop
        self.mechanics = SnnParsMechanics(
            parsname  = self.simulation.pars_mech_filename,
            new_pars  = params_to_update,
            pars_path = pars_path,
        )

        # Monitors
        self.monitor = SnnParsMonitor(
            parsname  = self.simulation.pars_monitor_filename,
            sim_pars  = self.simulation,
            top_pars  = self.topology,
            new_pars  = params_to_update,
            pars_path = pars_path,
        )

        # Parameters classes
        self.parameters_objects_list :list[SnnPars] = [
            self.simulation,
            self.topology,
            self.drive,
            self.neurons,
            self.synapses,
            self.mechanics,
            self.monitor
        ]

        # DEFINE MECHANICAL AND NEURONAL COORDINATES
        self.__define_coordinates()

        # DEFINE VARIABLE PARAMETERS
        self.__define_variable_parameters_units()

        # DEFINE INPUT/OUTPUT GAINS
        self.__define_auxiliary_indices_mc_axial()
        self.__define_auxiliary_indices_ps_axial()
        self.__define_callback_mc_gain_axial()
        self.__define_callback_mc_gain_limbs()
        self.__define_callback_ps_gain_axial()

        # CONSISTENCY CHECKS
        if params_to_update != {}:
            raise ValueError(f'Unable to update parameters:\n {params_to_update}')

        if not self.simulation.include_callback and self.simulation.include_online_act:
            raise ValueError('Callback must be included to compute the pools activations online')

        if sorted(self.neurons.synaptic_labels) != sorted(self.synapses.synaptic_labels):
            raise ValueError('The synapse labels must match the neuron labels')

        mech = self.mechanics

        if self.topology.segments_per_limb and self.topology.segments_per_limb != mech.mech_n_lb_joints_act:
            raise ValueError('The modeled limb segments must match the active limb joints')

        if mech.mech_n_lb_joints_lead > 1:
            raise ValueError('Only one limb joint can be selected as leader DOF')

        if mech.mech_n_lb_joints_lead == 0 and mech.mech_n_lb_joints_foll > 0:
            raise ValueError('One limb joint must be selected to have follower joints')

        if mech.mech_n_lb_joints != mech.mech_n_lb_joints_mov + mech.mech_n_lb_joints_fixd:
            raise ValueError('The limb segments must match the DOFs of the mechanical model')

        if len(mech.mech_activation_delays) != mech.mech_n_lb_joints_foll:
            raise ValueError('The number of activation delays must match the number of follower joints')

    ## NEURAL-MECHANICS GEOMETRY MAPPING
    def __define_neur_positions(self):
        ''' Defines the neural positions of the neurons '''

        directly_mapped_modules = ['cpg', 'rs', 'mn', 'ps', 'es', 'mc']

        for group in self.topology.neuron_groups_ids:
            for neuron in self.topology.neurons_lists[group]:

                # Ex: 'cpg' in 'cpg.axial.ex.V2a'.split('.')
                selected_module =  [
                    (mod in neuron.neuron_name.split('.'))
                    for mod in directly_mapped_modules
                ]

                if not any(selected_module):
                    continue

                module = neuron.module
                neur_num = neuron.index_pool_side_copy[3]
                p0, p1   = np.array( module.pools_positions_limits[neuron.pool_id], dtype= float)

                neuron.position_neur = p0 + (p1 - p0) * neur_num / module.n_pool_side

    def __define_mech_positions(self):
        ''' Defines the mechanical positions of the neurons '''

        mech_axial_joints_position = self.mechanics.mech_axial_joints_position
        mech_limbs_joints_position = self.mechanics.mech_limbs_joints_position

        directly_mapped_modules = []

        for group in self.topology.neuron_groups_ids:
            for neuron in self.topology.neurons_lists[group]:

                # Ex: 'ps' in 'ps.axial.ps'.split('.')
                selected_module =  [
                    (mod in neuron.neuron_name.split('.'))
                    for mod in directly_mapped_modules
                ]

                if not any(selected_module):
                    continue

                if 'axial' in neuron.neuron_name.split('.'):
                    neur_num = neuron.index - neuron.module.indices_limits[0]
                    neur_num_copy_side = neur_num % neuron.module.n_copy_side
                    neur_fraction_copy_side = neur_num_copy_side / neuron.module.n_copy_side

                    p0 = mech_axial_joints_position[0]
                    p1 = mech_axial_joints_position[-1]
                    neuron.position_mech = p0 + (p1 - p0) * neur_fraction_copy_side

                if 'limbs' in neuron.neuron_name.split('.'):
                    neuron.position_mech = mech_limbs_joints_position[neuron.limb_id - 1]

    def __get_adjacent_lb2_pos_old(self, pre_lb_ind: int, post_lb_ind: int):
        ''' Get previous and successive limb pair position '''

        length_axial = float(self.topology.length_axial)
        neur_lb2_pos = np.array(self.topology.limbs_pairs_y_positions, dtype= float)
        mech_lb2_pos = np.array(self.mechanics.mech_limbs_pairs_positions, dtype= float)
        mech_ax_pos  = np.array(self.mechanics.mech_axial_joints_position, dtype= float)

        ####
        # Offset for origin and tail  in the mechanical and neuronal coordinates
        ####

        # Coordinates of the first and last joint
        neur_frst_joint = neur_lb2_pos[0]  if neur_lb2_pos.any() else 0
        neur_last_joint = neur_lb2_pos[-1] if neur_lb2_pos.any() else length_axial
        mech_frst_joint = mech_lb2_pos[0]  if mech_lb2_pos.any() else mech_ax_pos[0]
        mech_last_joint = mech_lb2_pos[-1] if mech_lb2_pos.any() else mech_ax_pos[-1]

        # Neur coordinates start first
        if neur_frst_joint != 0 and mech_frst_joint == 0:
            mech_origin_offset = - neur_frst_joint
        else:
            mech_origin_offset = 0

        # Neur coordinates end last
        if neur_last_joint != length_axial and mech_last_joint == length_axial:
            mech_tail_offset = length_axial - neur_last_joint
        else:
            mech_tail_offset = 0

        # Mech coordinates start first
        if mech_frst_joint != 0 and neur_frst_joint == 0:
            neur_origin_offset = - mech_frst_joint
        else:
            neur_origin_offset = 0

        # Mech coordinates end last
        if mech_last_joint != length_axial and neur_last_joint == length_axial:
            neur_tail_offset = length_axial - mech_last_joint
        else:
            neur_tail_offset = 0

        ####
        # Inter-girdles ranges
        ####

        neur_lb_pos_prev = (
            neur_lb2_pos[-(pre_lb_ind+1)]
            if pre_lb_ind is not None
            else neur_origin_offset
        )
        neur_lb_pos_post = (
            neur_lb2_pos[post_lb_ind]
            if post_lb_ind is not None
            else length_axial + neur_tail_offset
        )

        mech_lb_pos_prev = (
            mech_lb2_pos[-(pre_lb_ind+1)]
            if pre_lb_ind is not None
            else mech_origin_offset
        )
        mech_lb_pos_post = (
            mech_lb2_pos[post_lb_ind]
            if post_lb_ind is not None
            else length_axial + mech_tail_offset
        )

        return neur_lb_pos_prev, neur_lb_pos_post, mech_lb_pos_prev, mech_lb_pos_post

    def __get_adjacent_lb2_pos_new(self, pre_lb_ind: int, post_lb_ind: int):
        ''' Get previous and successive limb pair position '''

        length_axial = float(self.topology.length_axial)
        neur_lb2_pos = np.array(self.topology.limbs_pairs_y_positions, dtype= float)
        mech_lb2_pos = np.array(self.mechanics.mech_limbs_pairs_positions, dtype= float)
        mech_ax_pos  = np.array(self.mechanics.mech_axial_joints_position, dtype= float)

        neur_lb_pos_prev = (
            neur_lb2_pos[-(pre_lb_ind+1)]
            if pre_lb_ind is not None
            else 0
        )
        neur_lb_pos_post = (
            neur_lb2_pos[post_lb_ind]
            if post_lb_ind is not None
            else length_axial
        )

        mech_lb_pos_prev = (
            mech_lb2_pos[-(pre_lb_ind+1)]
            if pre_lb_ind is not None
            else mech_ax_pos[0]
        )
        mech_lb_pos_post = (
            mech_lb2_pos[post_lb_ind]
            if post_lb_ind is not None
            else mech_ax_pos[-1]
        )

        return neur_lb_pos_prev, neur_lb_pos_post, mech_lb_pos_prev, mech_lb_pos_post

    def __get_adjacent_lb2_pos(self, position: float, pos_type: str):
        ''' Get previous and successive limb pair position '''

        if pos_type == 'neur':
            lb2_pos = np.array(self.topology.limbs_pairs_y_positions, dtype= float)
        elif pos_type == 'mech':
            lb2_pos = np.array(self.mechanics.mech_limbs_pairs_positions, dtype= float)
        else:
            raise ValueError('pos_type should be either "neur" or "mech"')

        # Previous and successive limb index
        pre_lb_ind  = next( (x for x, val in enumerate(lb2_pos[::-1]) if position >= val), None)
        post_lb_ind = next( (x for x, val in enumerate(lb2_pos[::1])  if position <  val), None)

        # Previous and successive limb position
        NEW_MAPPING = False

        if NEW_MAPPING:
            neur_lb_pos_prev, neur_lb_pos_post, mech_lb_pos_prev, mech_lb_pos_post = (
                self.__get_adjacent_lb2_pos_new(pre_lb_ind, post_lb_ind)
            )
        else:
            neur_lb_pos_prev, neur_lb_pos_post, mech_lb_pos_prev, mech_lb_pos_post = (
                self.__get_adjacent_lb2_pos_old(pre_lb_ind, post_lb_ind)
            )

        return neur_lb_pos_prev, neur_lb_pos_post, mech_lb_pos_prev, mech_lb_pos_post

    def __neur2mech_pos(self, pos_neur):
        ''' Map neural position to mechanical position '''
        neur_lb_pos_prev, neur_lb_pos_post, mech_lb_pos_prev, mech_lb_pos_post = (
            self.__get_adjacent_lb2_pos(pos_neur, 'neur')
        )
        neur_lb_range = neur_lb_pos_post - neur_lb_pos_prev
        mech_lb_range = mech_lb_pos_post - mech_lb_pos_prev
        lb_fraction   = (pos_neur - neur_lb_pos_prev) / neur_lb_range
        return mech_lb_pos_prev + lb_fraction * (mech_lb_range)

    def __mech2neur_pos(self, pos_mech):
        ''' Map mechanical position to neural position '''
        neur_lb_pos_prev, neur_lb_pos_post, mech_lb_pos_prev, mech_lb_pos_post = (
            self.__get_adjacent_lb2_pos(pos_mech, 'mech')
        )
        mech_lb_range = mech_lb_pos_post - mech_lb_pos_prev
        neur_lb_range = neur_lb_pos_post - neur_lb_pos_prev
        lb_fraction   = (pos_mech - mech_lb_pos_prev) / mech_lb_range
        return neur_lb_pos_prev + lb_fraction * (neur_lb_range)

    def __map_neur_to_mech_positions(self):
        ''' Assign mechanical positions from the neural ones, when not assigned '''

        for group in self.topology.neuron_groups_ids:
            for neuron in self.topology.neurons_lists[group]:
                if not np.isnan(neuron.position_mech):
                    continue
                neuron.position_mech = self.__neur2mech_pos(neuron.position_neur)

    def __map_mech_to_neur_positions(self):
        ''' Assign neuronal positions from the mechanical ones, when not assigned '''

        for group in self.topology.neuron_groups_ids:
            for neuron in self.topology.neurons_lists[group]:
                if not np.isnan(neuron.position_neur):
                    continue
                neuron.position_neur = self.__mech2neur_pos(neuron.position_mech)

    def __define_coordinates(self):
        ''' Defines neuronal an mechanical coordinates '''

        # ROADMAP:
        # 1) Define neuronal positions of all neurons
        # 2) Define motor neurons and muscle cells mechanical positions
        # 3) Map neuronal positions of neurons to mechanical positions
        # 4) Map mechanical positions of motor neurons and muscle cells to neural positions
        # 5) Assign newly defined mechanical and neural positions to all neurons

        self.__define_neur_positions()
        self.__define_mech_positions()
        self.__map_neur_to_mech_positions()
        self.__map_mech_to_neur_positions()

        for neuron_group in self.topology.neuron_groups_ids:
            self.topology.neurons_y_neur[neuron_group] = np.array(
                [ner.position_neur for ner in self.topology.neurons_lists[neuron_group]]
            )
            self.topology.neurons_y_mech[neuron_group] = np.array(
                [ner.position_mech for ner in self.topology.neurons_lists[neuron_group]]
            )

            assert not np.any( np.isnan( self.topology.neurons_y_neur[neuron_group] ) ), \
                'All neuronal positions should be assigned'
            assert not np.any( np.isnan( self.topology.neurons_y_mech[neuron_group] ) ), \
                'All mechanical positions should be assigned'

        logging.info("Defined neuronal and mechanical coordinates of neurons.")

    ## INPUT/OUTPUT GAINS
    def __define_auxiliary_indices_mc_axial(self) -> None:
        ''' Defines auxiliary MC indices for the callback function '''

        if not self.topology.include_muscle_cells_axial:
            return

        net_modules = self.topology.network_modules

        # Indices of the mc neurons whose y coordinate is closest to the joints
        self.output_mc_axial_inds = np.array(
            [
                np.argmin(
                    np.abs(
                        self.topology.neurons_y_mech[1][:net_modules['mc']['axial'].n_side] - j_pos
                    )
                ) + side_off
                for j_pos in self.mechanics.mech_axial_joints_position
                for side_off in [0, net_modules['mc']['axial'].n_side]
            ]
        )

    def __define_auxiliary_indices_ps_axial(self) -> None:
        ''' Defines auxiliary PS indices for the callback function '''

        if not self.topology.include_proprioception_axial:
            return

        net_modules = self.topology.network_modules
        ind_0_ps    = net_modules['ps']['axial'].indices[0]
        n_side_ps   = net_modules['ps']['axial'].n_side

        # Indices of the ps neurons whose y coordinate is closest to the joints
        input_ps_axial_inds = ind_0_ps + np.array(
            [
                np.argmin(
                    np.abs(
                        self.topology.neurons_y_mech[0][ ind_0_ps : ind_0_ps + n_side_ps] - j_pos
                    )
                )
                for j_pos in self.mechanics.mech_axial_joints_position
            ]
        )

        # Pools of the ps neurons whose y coordinate is closest to the joints
        self.input_ps_axial_pool_inds = np.array(
            [
                self.topology.neurons_lists[0][ner_ind].pool_id
                for ner_ind in input_ps_axial_inds
            ]
        )

        return

    def __define_callback_mc_gain_axial(self, mc_scaling= 1.0) -> None:
        ''' Defines the mc_axial_gains for the callback function '''

        if not self.topology.include_muscle_cells_axial:
            return

        if mc_scaling != 1.0:
            logging.info(f'Applying scaling {mc_scaling} to mc_gain_axial')

        mc_gain_ax        = self.simulation.mc_gain_axial * mc_scaling
        mech_axial_joints = self.mechanics.mech_axial_joints

        # Float
        if isinstance(mc_gain_ax, (int, float)):
            mc_gain_axial_ext = mc_gain_ax * np.ones(2 * mech_axial_joints)

        # Gain for each joint
        elif len(mc_gain_ax) == mech_axial_joints:
            mc_gain_axial_ext = np.repeat(mc_gain_ax, 2)

        # Gain for each side of each joint
        elif len(mc_gain_ax) == 2 * mech_axial_joints:
            mc_gain_axial_ext = np.array(mc_gain_ax)

        # Error
        else:
            raise ValueError(
                f'mc_gain_axial should be a float, a list of length {mech_axial_joints}, '
                f'or a list of length {2*mech_axial_joints}'
            )

        logging.info(
            f'Updating mc_gain_axial_ext to {mc_gain_axial_ext}'
        )

        # Update
        self.mc_gain_axial_callback = mc_gain_axial_ext
        return

    def __define_callback_mc_gain_limbs(self, mc_scaling= 1.0) -> None:
        ''' Defines the mc_gain_limbs for the callback function '''

        if not self.topology.include_muscle_cells_limbs:
            return

        if mc_scaling != 1.0:
            logging.info(f'Applying scaling {mc_scaling} to mc_gain_limbs')

        mc_gain_lb        = self.simulation.mc_gain_limbs * mc_scaling
        mech_limbs_joints = self.mechanics.mech_limbs_joints

        # Float
        if isinstance(mc_gain_lb, (int, float)):
            mc_gain_limbs_ext = mc_gain_lb * np.ones(2 * mech_limbs_joints)

        # Gain for each joint
        elif len(mc_gain_lb) == mech_limbs_joints:
            mc_gain_limbs_ext = np.repeat(mc_gain_lb, 2)

        # Gain for each side of each joint
        elif len(mc_gain_lb) == 2 * mech_limbs_joints:
            mc_gain_limbs_ext = np.array(mc_gain_lb)

        # Error
        else:
            raise ValueError(
                f'mc_gain_limbs should be a float, a list of length {mech_limbs_joints}, '
                f'or a list of length {2*mech_limbs_joints}'
            )

        # Update
        self.mc_gain_limbs_callback = mc_gain_limbs_ext
        return

    def __define_callback_ps_gain_axial(self, ps_scaling= 1.0) -> None:
        ''' Defines the ps_gain_axial for the callback function '''

        if not self.topology.include_proprioception_axial:
            return

        if ps_scaling != 1.0:
            logging.info(f'Applying scaling {ps_scaling} to ps_gain_axial')

        net_module_ps = self.topology.network_modules['ps']['axial']
        ps_gain_ax    = self.simulation.ps_gain_axial * ps_scaling

        neur_axial_joints = self.topology.network_modules['ps']['axial'].pools
        mech_axial_joints = self.mechanics.mech_axial_joints

        # Float
        if isinstance(ps_gain_ax, (int, float)):
            ps_gain_axial_pools_sides = ps_gain_ax * np.ones(2 * net_module_ps.pools)

        # Gain for each pool
        elif len(ps_gain_ax) == net_module_ps.pools:
            ps_gain_axial_pools_sides = np.concatenate((ps_gain_ax, ps_gain_ax))

        # Gain for each side of each pool
        elif len(ps_gain_ax) == 2 * net_module_ps.n_pool:
            ps_gain_axial_pools_sides = np.array(ps_gain_ax)

        # Gain for each joint
        elif len(ps_gain_ax) == mech_axial_joints:
            gains_mapped = np.zeros(neur_axial_joints)

            for joint in range(mech_axial_joints):
                prev_gain = ps_gain_ax[ joint ]
                post_gain = ps_gain_ax[ joint+1 ] if joint < mech_axial_joints-1 else 0

                prev_pool = self.input_ps_axial_pool_inds[joint]
                post_pool = self.input_ps_axial_pool_inds[joint+1] if joint < mech_axial_joints-1 else neur_axial_joints

                for pool_id in range(prev_pool, post_pool):
                    gains_mapped[pool_id] = (
                        prev_gain + (pool_id - prev_pool) / (post_pool - prev_pool) * (post_gain - prev_gain)
                    )

            # Expand ps gain to all the pools of both sides
            ps_gain_axial_pools_sides = np.concatenate( [gains_mapped, gains_mapped] )

        # Gain for each side of each joint
        elif len(ps_gain_ax) == 2 * mech_axial_joints:
            gains_mapped = np.zeros(2*neur_axial_joints)

            for joint in range(mech_axial_joints):
                prev_gain_l = ps_gain_ax[ 2*joint ]
                post_gain_l = ps_gain_ax[ 2*(joint+1) ] if joint < mech_axial_joints-1 else 0

                prev_gain_r = ps_gain_ax[ 1 + 2*joint ]
                post_gain_r = ps_gain_ax[ 1 + 2*(joint+1) ] if joint < mech_axial_joints-1 else 0

                prev_pool = self.input_ps_axial_pool_inds[joint]
                post_pool = self.input_ps_axial_pool_inds[joint+1] if joint < mech_axial_joints-1 else neur_axial_joints

                for pool_id in range(prev_pool, post_pool):
                    gains_mapped[pool_id] = (
                        prev_gain_l + (pool_id - prev_pool) / (post_pool - prev_pool) * (post_gain_l - prev_gain_l)
                    )
                    gains_mapped[neur_axial_joints + pool_id] = (
                        prev_gain_r + (pool_id - prev_pool) / (post_pool - prev_pool) * (post_gain_r - prev_gain_r)
                    )

            ps_gain_axial_pools_sides = gains_mapped

        # Error
        else:
            raise ValueError(
                f'ps_gain_axial should be a float, a list of length {net_module_ps.pools}, '
                f'or a list of length {2*net_module_ps.pools}, '
                f'or a list of length {mech_axial_joints}, '
                f'or a list of length {2*mech_axial_joints}'
            )

        # Update
        self.ps_gain_axial_callback = np.repeat(
            ps_gain_axial_pools_sides,
            repeats = net_module_ps.n_pool  // 2
        )
        return

    ## VARIABLE PARAMETERS
    def __define_variable_parameters_units(self) -> dict:
        ''' Method to define the variable parameters units of the system '''

        # Available groups
        neuron_groups = set(
            [
                sub_part.neuron_group
                for sub_part in self.topology.network_modules.sub_parts_list
                if sub_part.include
            ]
        )

        # Neuronal parameters units
        self.variable_ner_pars_units_list = [
            {
                key:val for key,val in vnpu_type.items()
                if vnpu_type['neuron_group'] in neuron_groups
            }
            for vnpu_type in self.neurons.variable_neural_params_units
        ]

        # Synaptic parameters units
        self.variable_syn_pars_units_list = [
            {
                key:val for key,val in vspu_type.items()
                if vspu_type['neuron_group'] in neuron_groups
            }
            for vspu_type in self.synapses.variable_syn_params_units
        ]

        return self.variable_ner_pars_units_list, self.variable_syn_pars_units_list

    ## GET PARAMETERS
    def get_variable_parameters(self) -> dict:
        ''' Method to collect the variable parameters of the system '''

        def new_sortedkeys(dic: dict, dic_id: int) -> dict:
            ''' Prepend ID to preserve lexicographic ordering '''
            keys = sorted(dic)
            for dic_key in keys:
                if dic_key in ['synaptic_group', 'neuron_group']:
                    dic.pop(dic_key)
                    continue

                dic[f"{dic_id}_{dic_key}"] = dic.pop(dic_key)

            return dic

        # Pre-append ID to distinguish parameters for the differen populations
        variable_params_units = {}

        # Neuronal parameters
        variable_neur_pars_units = [
            new_sortedkeys(vnpu_type, 0)
            for vnpu_type in self.variable_ner_pars_units_list
        ]

        # Update
        for variable_neur_type_pars_units in variable_neur_pars_units:
            variable_params_units.update(variable_neur_type_pars_units)

        # Synaptic parameters
        variable_syn_params_units = [
            new_sortedkeys(vspu_type, vspu_type['synaptic_group']+1)
            for vspu_type in self.variable_syn_pars_units_list
        ]

        # Update
        for variable_syn_type_pars_units in variable_syn_params_units:
            variable_params_units.update(variable_syn_type_pars_units)

        return variable_params_units

    def get_parameters(self) -> dict:
        ''' Return the minimum set of parameters to run the simulation '''
        return self.__dict__

    ## EXTEND PROPERTIES
    # GAITS
    @SnnParsSimulation.gaitflag.setter
    def gaitflag(self, value: int):
        SnnParsSimulation.gaitflag.fset(self.simulation, value)

        self.drive.update_gait_drives(
            self.simulation.gait,
            self.topology.ascending_feedback_str
        )
    @SnnParsSimulation.gait.setter
    def gait(self, value: str):
        SnnParsSimulation.gait.fset(self.simulation, value)

        self.drive.update_gait_drives(
            self.simulation.gait,
            self.topology.ascending_feedback_str
        )

    # DRIVES
    def __drives_setter(
        self,
        value       : Union[int, float, np.ndarray],
        drive_setter: Callable,
        drive_name  : str,
        n_pools     : int,
    ):
        ''' Sets the drives of the simulation '''
        if type(value) not in (int, float, list, np.ndarray):
            raise TypeError(f'{drive_name} should be a int, float, list or numpy array')

        if isinstance(value, list):
            value = np.array(value)
        if isinstance(value, np.ndarray):
            assert value.shape == n_pools, \
                f'{drive_name} should have length {n_pools}'

        drive_setter(self.simulation, value)

    @SnnParsSimulation.stim_a_mul.setter
    def stim_a_mul(self, value: Union[float, np.ndarray]):
        self.__drives_setter(
            value        = value,
            drive_setter = SnnParsSimulation.stim_a_mul.fset,
            drive_name   = 'stim_a_mul',
            n_pools      = self.topology.network_modules['rs']['axial'].pools,
        )
    @SnnParsSimulation.stim_a_off.setter
    def stim_a_off(self, value: Union[float, np.ndarray]):
        self.__drives_setter(
            value        = value,
            drive_setter = SnnParsSimulation.stim_a_off.fset,
            drive_name   = 'stim_a_off',
            n_pools      = self.topology.network_modules['rs']['axial'].pools,
        )
    @SnnParsSimulation.stim_l_mul.setter
    def stim_l_mul(self, value: Union[float, np.ndarray]):
        self.__drives_setter(
            value        = value,
            drive_setter = SnnParsSimulation.stim_l_mul.fset,
            drive_name   = 'stim_l_mul',
            n_pools      = self.topology.network_modules['rs']['limbs'].pools,
        )
    @SnnParsSimulation.stim_l_off.setter
    def stim_l_off(self, value: Union[float, np.ndarray]):
        self.__drives_setter(
            value        = value,
            drive_setter = SnnParsSimulation.stim_l_off.fset,
            drive_name   = 'stim_l_off',
            n_pools      = self.topology.network_modules['rs']['limbs'].pools,
        )

    # GAINS
    def __gains_setter(
        self,
        value      : Union[float, np.ndarray],
        gain_setter: Callable,
        gain_name  : str,
        n_joints   : int,
    ):
        ''' Sets the gain of the model '''
        if type(value) not in (int, float, list, np.ndarray):
            raise TypeError(f'{gain_name} should be a float, list or numpy array')

        if isinstance(value, list):
            value = np.array(value)
        if isinstance(value, np.ndarray):
            assert value.shape in ( (n_joints,),(2*n_joints,) ), \
                f'{gain_name} should have the length {n_joints} or {2*n_joints}'

            # Repeat values (motor control has 2 signals per joint)
            if value.shape == (n_joints,):
                value = value.repeat(2)

        gain_setter(self.simulation, value)

    @SnnParsSimulation.mc_gain_axial.setter
    def mc_gain_axial(self, value: Union[float, list, np.ndarray]):
        self.__gains_setter(
            value       = value,
            gain_setter = SnnParsSimulation.mc_gain_axial.fset,
            gain_name   = 'mc_gain_axial',
            n_joints    = self.mechanics.mech_axial_joints
        )
        self.__define_callback_mc_gain_axial()

    @SnnParsSimulation.mc_gain_limbs.setter
    def mc_gain_limbs(self, value: Union[float, np.ndarray]):
        self.__gains_setter(
            value       = value,
            gain_setter = SnnParsSimulation.mc_gain_limbs.fset,
            gain_name   = 'mc_gain_limbs',
            n_joints    = self.mechanics.mech_limbs_joints
        )
        self.__define_callback_mc_gain_limbs()

    @SnnParsSimulation.ps_gain_axial.setter
    def ps_gain_axial(self, value: Union[float, np.ndarray]):
        self.__gains_setter(
            value       = value,
            gain_setter = SnnParsSimulation.ps_gain_axial.fset,
            gain_name   = 'ps_gain_axial',
            n_joints    = self.mechanics.mech_axial_joints
        )
        self.__define_callback_ps_gain_axial()

    @SnnParsSimulation.ps_gain_limbs.setter
    def ps_gain_limbs(self, value: Union[float, np.ndarray]):
        self.__gains_setter(
            value       = value,
            gain_setter = SnnParsSimulation.ps_gain_limbs.fset,
            gain_name   = 'ps_gain_limbs',
            n_joints    = self.mechanics.mech_limbs_joints
        )
    @SnnParsSimulation.es_gain_axial.setter
    def es_gain_axial(self, value: Union[float, np.ndarray]):
        self.__gains_setter(
            value       = value,
            gain_setter = SnnParsSimulation.es_gain_axial.fset,
            gain_name   = 'es_gain_axial',
            n_joints    = self.mechanics.mech_axial_joints
        )
    @SnnParsSimulation.es_gain_limbs.setter
    def es_gain_limbs(self, value: Union[float, np.ndarray]):
        self.__gains_setter(
            value       = value,
            gain_setter = SnnParsSimulation.es_gain_limbs.fset,
            gain_name   = 'es_gain_limbs',
            n_joints    = self.mechanics.mech_limbs_joints
        )

def main():
    ''' Test case '''
    import matplotlib.pyplot as plt

    print('TEST: Network Parameters ')
    par = SnnParameters(
        # parsname     = 'pars_simulation_farms_4limb_1dof_unweighted',
        parsname     = 'pars_simulation_farms_zebrafish_simplified',
        results_path = 'simulation_results_test',
    )

    positions_mech = par.topology.neurons_y_mech[0]
    positions_neur = par.topology.neurons_y_neur[0]

    plt.figure('Neural and Mechanical positions')
    plt.plot(positions_neur, label='Neur')
    plt.plot(positions_mech, label='Mech')
    plt.legend()

    positions_mech_mapped = [par._SnnParameters__neur2mech_pos(pos) for pos in positions_neur]
    positions_neur_mapped = [par._SnnParameters__mech2neur_pos(pos) for pos in positions_mech_mapped]

    error = np.linalg.norm(positions_mech - positions_mech_mapped)
    print(f'Error between original and reconstructed mech coordinates: {error: .4f}')

    error = np.linalg.norm(positions_neur - positions_neur_mapped)
    print(f'Error between original and reconstructed neur coordinates: {error: .4f}')

    plt.figure('Neural and Mapped Neural positions')
    plt.plot(positions_neur, label='Neur')
    plt.plot(positions_neur_mapped, label='Mapped Neur')
    plt.legend()

    plt.show()

    return par

if __name__ == '__main__':
    main()