'''
Functions to specify the connectivity within the network
'''
import os
import dill
import logging
import numpy as np

from brian2 import NeuronGroup, Synapses
from network_modules.equations import connections, parameter_setting
from network_modules.core.network_core import SnnCore

PRINTAUX = '-' * 3 + ' '

class SnnConnectivityCore(SnnCore):
    '''
    Class to define the connectivity for the network
    '''

    ### Define connections
    def define_connections_from_specs(
        self,
        connection_specs: dict,
        extracond       : str = '',
        **kwargs,
    ):
        ''' Define the connections based on the identifiers of the neurons '''
        logging.info('%s %s', PRINTAUX, connection_specs['name'])

        # SYNAPSE
        synapse = getattr(self, connection_specs['synapse'])

        # CONNECTION TYPE
        connect_type = connection_specs['type']
        if connect_type not in [
            'connect',
            'connect_identity',
            'gaussian_identity',
            'sat_gaussian_identity',
            'trapezoidal_identity',
        ]:
            raise ValueError('Wrong connection type')

        # EXTRA PARAMETERS
        pools_list = kwargs.pop('pools_list', None)

        pre_ind_limits  = kwargs.pop('pre_ind_limits', None)
        post_ind_limits = kwargs.pop('post_ind_limits', None)

        pre_y_neur_limits  = kwargs.pop('pre_y_neur_limits', None)
        post_y_neur_limits = kwargs.pop('post_y_neur_limits', None)

        pre_y_mech_limits  = kwargs.pop('pre_y_mech_limits', None)
        post_y_mech_limits = kwargs.pop('post_y_mech_limits', None)

        pre_pool_id  = kwargs.pop('pre_pool_id', None)
        post_pool_id = kwargs.pop('post_pool_id', None)

        pre_limb_id  = kwargs.pop('pre_limb_id', None)
        post_limb_id = kwargs.pop('post_limb_id', None)

        y_neur_off = kwargs.pop('post_y_neur_offset', None)
        y_mech_off = kwargs.pop('post_y_mech_offset', None)

        assert kwargs == {}, f'Not all elements are recognized: {kwargs.keys()}'

        # EXTRA CONDITIONS
        extra_conditions_list = [ connection_specs['cond_str'], extracond ]

        def _get_ind_lim_cond(indtype: str, indlimits: list[int]):
            ''' Conditions on indices '''
            assert indtype in ['i', 'j'], 'Invalid indtype'
            return (
            f'( {indtype} >= {indlimits[0]} and {indtype} <= {indlimits[-1]} )'
            if indlimits is not None
            else ''
        )

        extra_conditions_list.append( _get_ind_lim_cond('i', pre_ind_limits) )
        extra_conditions_list.append( _get_ind_lim_cond('j', post_ind_limits) )

        def _get_y_lim_cond(ytype: str, syntarget: str, ylimits: list[float]):
            ''' Conditions on y coordinates '''
            assert ytype in ['neur', 'mech'], 'Invalid ytype'
            assert syntarget in ['pre', 'post'], 'Invalid syntarget'
            return (
                f'( y_{ytype}_{syntarget} >= {float(ylimits[0])} * metre and'
                + f'y_{ytype}_{syntarget} <= {float(ylimits[-1])} * metre )'
                if pre_y_neur_limits is not None
                else ''
            )

        extra_conditions_list.append( _get_y_lim_cond('neur',  'pre',  pre_y_neur_limits) )
        extra_conditions_list.append( _get_y_lim_cond('neur', 'post', post_y_neur_limits) )
        extra_conditions_list.append( _get_y_lim_cond('mech',  'pre',  pre_y_mech_limits) )
        extra_conditions_list.append( _get_y_lim_cond('mech', 'post', post_y_mech_limits) )

        def _get_equality_cond(parameter_str: str, parameter_val: int):
            ''' Conditions on equality of a parameter '''
            return (
            f'( {parameter_str} == {parameter_val} )'
            if parameter_val is not None
            else ''
        )

        extra_conditions_list.append( _get_equality_cond( 'pool_id_pre',  pre_pool_id) )
        extra_conditions_list.append( _get_equality_cond('pool_id_post', post_pool_id) )
        extra_conditions_list.append( _get_equality_cond( 'limb_id_pre',  pre_limb_id) )
        extra_conditions_list.append( _get_equality_cond('limb_id_post', post_limb_id) )

        # Join all conditions
        extra_conditions = ' and '.join([ cond for cond in extra_conditions_list if cond != '' ])

        # CONNECTION
        seg_h = self.params.topology.height_segment

        if connect_type == 'connect':
            connections.connect(
                syn             = synapse,
                pools_list      = pools_list,
                prob            = connection_specs['parameters']['amp'],
                extraconditions = extra_conditions
            )

        if connect_type == 'connect_identity':
            connections.connect_byidentity(
                syn             = synapse,
                condlist        = connection_specs['cond_list'],
                prob            = connection_specs['parameters']['amp'],
                extraconditions = extra_conditions,
            )

        if connect_type == 'gaussian_identity':
            connections.gaussian_asymmetric_connect_byidentity(
                syn             = synapse,
                condlist        = connection_specs['cond_list'],
                y_type          = connection_specs['parameters']['y_type'],
                amp             = connection_specs['parameters']['amp'],
                sigma_up        = connection_specs['parameters']['sigma_up'] * seg_h,
                sigma_dw        = connection_specs['parameters']['sigma_dw'] * seg_h,
                extraconditions = extra_conditions,
            )

        if connect_type == 'sat_gaussian_identity':

            # Select neuronal or mechanics coordinates
            y_type          = connection_specs['parameters']['y_type']
            range_fractions = connection_specs['parameters']['limits']

            if y_type not in ['y_neur', 'y_mech']:
                raise ValueError('Invalide y_type')

            elif y_type == 'y_neur':
                y_off        = y_neur_off
                y_min, y_max = post_y_neur_limits

            elif y_type == 'y_mech':
                y_off        = y_mech_off
                y_min, y_max = post_y_mech_limits

            # Translate fractions into limits for the coordinates
            def _get_ylim(y0, fraction, range):
                return y0 + fraction * range if fraction is not None else np.NaN

            if y_off is None:
                # Fractions of the whole range
                y_min_0   = y_min
                y_min_1   = y_min
                y_range_0 = y_max - y_min
                y_range_1 = y_max - y_min
            else:
                # Fractions of the lower and upper range
                y_min_0   = y_min
                y_min_1   = y_off
                y_range_0 = y_off - y_min
                y_range_1 = y_max - y_off

            y_limits = [
                _get_ylim(y_min_0, range_fractions[0], y_range_0),
                _get_ylim(y_min_0, range_fractions[1], y_range_0),
                _get_ylim(y_min_1, range_fractions[2], y_range_1),
                _get_ylim(y_min_1, range_fractions[3], y_range_1),
            ]

            # Connect
            connections.satgaussian_connect_byidentity(
                syn             = synapse,
                pre_ilimits     = pre_ind_limits,
                post_ylimits    = y_limits,
                segment_height  = self.params.topology.height_segment_row,
                condlist        = connection_specs['cond_list'],
                y_type          = connection_specs['parameters']['y_type'],
                amp             = connection_specs['parameters']['amp'],
                sigma_up        = connection_specs['parameters']['sigma_up'] * seg_h,
                sigma_dw        = connection_specs['parameters']['sigma_dw'] * seg_h,
                extraconditions = extra_conditions,
            )

        if connect_type == 'trapezoidal_identity':

            # Select neuronal or mechanics coordinates
            y_type = connection_specs['parameters']['y_type']

            if y_type not in ['y_neur', 'y_mech']:
                raise ValueError('Invalide y_type')

            elif y_type == 'y_neur':
                post_ylimits = post_y_neur_limits

            elif y_type == 'y_mech':
                post_ylimits = post_y_mech_limits

            # Connect
            connections.trapezoidal_connect_byidentity(
                syn             = synapse,
                pre_ilimits     = pre_ind_limits,
                post_ylimits    = post_ylimits,
                segment_height  = self.params.topology.height_segment_row,
                condlist        = connection_specs['cond_list'],
                y_type          = connection_specs['parameters']['y_type'],
                amp             = connection_specs['parameters']['amp'],
                sigma_up        = connection_specs['parameters']['sigma_up'] * seg_h,
                sigma_dw        = connection_specs['parameters']['sigma_dw'] * seg_h,
                extraconditions = extra_conditions,
            )

    ### Silencing
    def toggle_ner_silencing_by_neural_inds(
        self,
        pop       : NeuronGroup,
        inds_ner  : list[int],
        silenced  : bool = False,
        extra_cond: tuple[list[str], str] = None,
    ) -> None:
        '''
        Assigns silenced to the synapses objects linking neurons between
        the specified indices
        '''
        silence_par = {'silenced_ner': [int(silenced), ''] }

        parameter_setting.set_neural_parameters_by_neural_inds(
            ner_group  = pop,
            inds_ner   = inds_ner,
            std_value  = self.params.neurons.std_val,
            parameters = silence_par,
            extra_cond = extra_cond,
        )

    def toggle_ner_silencing_by_neural_inds_range(
        self,
        pop        : NeuronGroup,
        limits_list: list[list[int]],
        silenced   : bool = False,
        extra_cond : tuple[list[str], str] = None,
    ) -> None:
        '''
        Assigns silenced to the neurons between the specified indices
        '''
        for limits in limits_list:
            self.toggle_ner_silencing_by_neural_inds(
                pop        = pop,
                inds_ner   = range(limits[0], limits[-1]),
                silenced   = silenced,
                extra_cond = extra_cond,
            )

    def toggle_syn_silencing_by_neural_inds_range(
        self,
        syns_list   : list[Synapses],
        ilimits_list: list[int],
        jlimits_list: list[int],
        silenced    : bool = False,
        extra_cond  : tuple[list[str], str] = None,
    ) -> None:
        '''
        Assigns silenced to the synapses objects linking neurons between
        the specified indices
        '''
        silence_par = {'silenced_syn': [int(silenced), ''] }
        for syn in syns_list:
            for ilims, jlims in zip(ilimits_list, jlimits_list):
                parameter_setting.set_synaptic_parameters_by_neural_ind_ranges(
                    syn_group         = syn,
                    inds_limits_syn_i = ilims,
                    inds_limits_syn_j = jlims,
                    std_value         = self.params.neurons.std_val,
                    parameters        = silence_par,
                    extra_cond        = extra_cond,
                )

    def toggle_syn_silencing_by_neural_inds(
        self,
        syns_list : list[Synapses],
        inds_ner_i: list[int],
        inds_ner_j: list[int],
        silenced  : bool = False,
        extra_cond: tuple[list[str], str] = None,
    ) -> None:
        '''
        Assigns silenced to the synapses objects linking neurons between
        the specified indices
        '''
        silence_par = {'silenced_syn': [int(silenced), ''] }
        for syn in syns_list:
            parameter_setting.set_synaptic_parameters_by_neural_ind(
                syn_group  = syn,
                inds_syn_i = inds_ner_i,
                inds_syn_j = inds_ner_j,
                std_value  = self.params.neurons.std_val,
                parameters = silence_par,
                extra_cond = extra_cond,
            )

    def toggle_syn_silencing_by_syn_inds(
        self,
        syn       : Synapses,
        syn_inds  : np.ndarray,
        silenced  : bool = False,
        extra_cond: tuple[list[str], str] = None,
    ) -> None:
        '''
        Assigns silenced to the synapses objects linking neurons between
        the specified indices
        '''
        silence_par = {'silenced_syn': [int(silenced), ''] }
        parameter_setting.set_synaptic_parameters_by_synaptic_inds(
            syn_group  = syn,
            inds_syn   = syn_inds,
            std_value  = self.params.neurons.std_val,
            parameters = silence_par,
            extra_cond = extra_cond,
        )

    def toggle_silencing_by_neural_inds_range(
        self,
        pop       : NeuronGroup,
        syns      : list[Synapses],
        limits    : list,
        silenced  : bool = False,
        extra_cond: tuple[list[str], str] = None,
    ) -> None:
        '''
        Assigns silenced to neurons and synapses objects linking neurons between
        the specified indices [[ilimits0,jlimits0],[ilimits1,jlimits1]...]
        '''
        pop_lims = [0, len(pop)]

        # SILENCE NEURONS
        pop[ limits[0] : limits[-1]+1 ].silenced_ner = int(silenced)

        # SILENCE CONNECTION from, to and between neurons in the interval
        silence_par = {'silenced_syn': [int(silenced), ''] }
        for syn in syns:
            for ilims, jlims in [ [pop_lims, limits], [limits, pop_lims] ]:
                parameter_setting.set_synaptic_parameters_by_neural_ind_ranges(
                    syn_group         = syn,
                    inds_limits_syn_i = ilims,
                    inds_limits_syn_j = jlims,
                    std_value         = self.params.neurons.std_val,
                    parameters        = silence_par,
                    extra_cond        = extra_cond,
                )

    def set_syn_weight_by_neural_ind(
        self,
        syns_list : list[Synapses],
        inds_ner_i: list[int],
        inds_ner_j: list[int],
        silenced  : bool = False,
        extra_cond: tuple[list[str], str] = None,
    ) -> None:
        '''
        Assigns silenced to the synapses objects linking neurons between
        the specified indices
        '''
        silence_par = {'silenced_syn': [int(silenced), ''] }
        for syn in syns_list:
            parameter_setting.set_synaptic_parameters_by_neural_ind(
                syn_group  = syn,
                inds_syn_i = inds_ner_i,
                inds_syn_j = inds_ner_j,
                std_value  = self.params.neurons.std_val,
                parameters = silence_par,
                extra_cond = extra_cond,
            )

    ### Loading connectivity indices
    def load_connectivity_indices(self) -> None:
        ''' Loads the connectivity indices (sources, targets) of the network '''

        wmat_file = self.params.simulation.connectivity_indices_file
        logging.info('Loading connectivity indices from %s', wmat_file)

        with open(wmat_file, "rb") as infile:
            self.syn_indices : dict[str, dict[str, np.ndarray]] = dill.load(infile)

        return

    ### Saving connectivity indices
    def save_connectivity_indices(self, file_path: str = None) -> None:
        '''
        Saves the connectivity indices (sources, targets) of the network and muscle cells
        '''
        conn_inds_file = (
            self.params.simulation.connectivity_indices_file
            if file_path is None
            else file_path
        )
        logging.info('Saving connectivity indices to %s', conn_inds_file)

        os.makedirs( os.path.dirname(conn_inds_file), exist_ok=True)
        with open(conn_inds_file, "wb") as outfile:
            dill.dump(self.syn_indices, outfile)

        return

    ### Getting connectivity indices
    def define_connectivity_indices(self) -> None:
        ''' Get indices (sources, targets) for the synaptic connections '''

        self.syn_indices : dict[str, dict[str, np.ndarray]] = {}
        for syn in self.synaptic_groups_list:
            self.syn_indices[syn.name] = {
              'i' : np.array( syn.i ),
              'j' : np.array( syn.j ),
            }

    ### Getting connectivity matrices
    def get_wmat(self, recompute: bool = True, active_links_only: bool = True) -> np.ndarray:
        ''' Creates connectivity matrix of the network '''

        if recompute or self.wmat is None or not self.wmat.any():
            # Create a sparse matrix to store the connections
            self.wmat = np.zeros((len(self.pop), len(self.pop)))
            # Insert the values from the Synapses object
            if active_links_only and self.params.simulation.include_silencing:
                self.wmat[self.syn_ex.i[:], self.syn_ex.j[:]] = + ( 1-self.syn_ex.silenced_syn[:] )
                self.wmat[self.syn_in.i[:], self.syn_in.j[:]] = - ( 1-self.syn_in.silenced_syn[:] )
            else:
                self.wmat[self.syn_ex.i[:], self.syn_ex.j[:]] = + 1
                self.wmat[self.syn_in.i[:], self.syn_in.j[:]] = - 1

        return self.wmat

    def get_wmat_mc(self, recompute: bool = False) -> np.ndarray:
        ''' Creates connectivity matrix for the muscel cells of the network '''

        if not self.params.topology.include_muscle_cells or self.mc_pop is None:
            return None

        if recompute or self.wmat_mc is None or not self.wmat_mc.any():
            # Create a sparse matrix to store the connections
            self.wmat_mc = np.zeros((len(self.pop), len(self.mc_pop)))

            # Insert the values from the Synapses object
            if len(self.syn_mc_ex):
                self.wmat_mc[self.syn_mc_ex.i[:], self.syn_mc_ex.j[:]] = + 1

            if len(self.syn_mc_in):
                self.wmat_mc[self.syn_mc_in.i[:], self.syn_mc_in.j[:]] = - 1

        return self.wmat_mc

    ### Defining connectivity
    def _define_connectivity_from_indices_net(self) -> None:
        '''
        Defines the network connectivity based on input connectivity indices
        '''
        logging.info('Defining connectivity_net from connectivity indices')

        # Excitatory
        if len(self.syn_indices['syn_ex']['i']) and len(self.syn_indices['syn_ex']['j']):
            self.syn_ex.connect(
                i=self.syn_indices['syn_ex']['i'],
                j=self.syn_indices['syn_ex']['j']
            )

        # Inhibitory
        if len(self.syn_indices['syn_in']['i']) and len(self.syn_indices['syn_in']['j']):
            self.syn_in.connect(
                i=self.syn_indices['syn_in']['i'],
                j=self.syn_indices['syn_in']['j']
            )


        return

    def _define_connectivity_from_indices_mc(self) -> None:
        '''
        Defines the muscle cells connectivity based on input connectivity indices
        '''
        if not self.params.topology.include_muscle_cells:
            return

        logging.info('Defining connectivity_mc from connectivity indices')

        # Excitatory
        if len(self.syn_indices['syn_mc_ex']['i']) and len(self.syn_indices['syn_mc_ex']['j']):
            self.syn_mc_ex.connect(
                i=self.syn_indices['syn_mc_ex']['i'],
                j=self.syn_indices['syn_mc_ex']['j']
            )

        # Inhibitory
        if len(self.syn_indices['syn_mc_in']['i']) and len(self.syn_indices['syn_mc_in']['j']):
            self.syn_mc_in.connect(
                i=self.syn_indices['syn_mc_in']['i'],
                j=self.syn_indices['syn_mc_in']['j']
            )

        return

    def define_connectivity_from_indices(self) -> None:
        '''
        Defines the connectivity based on input weight matrices
        '''
        self.load_connectivity_indices()
        self._define_connectivity_from_indices_net()
        self._define_connectivity_from_indices_mc()

# TEST
def main():
    ''' Test case '''

    from queue import Queue

    logging.info('TEST: SNN Connectivity ')

    connectivity = SnnConnectivityCore(
        network_name = 'snn_core_test',
        params_name  = 'pars_simulation_test',
        q_in         = Queue(),
        q_out        = Queue(),
    )

    return connectivity

if __name__ == '__main__':
    main()