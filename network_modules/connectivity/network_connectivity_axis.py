'''
Functions to specify the connectivity within the network
'''
import logging

from network_modules.connectivity.network_connectivity_core import SnnConnectivityCore

PRINTAUX = '-'*3 + ' '

class SnnConnectivityAxis(SnnConnectivityCore):
    '''
    Class to define the connectivity for the axial network
    '''

    ## NEURON_GROUP_0 --> NEURON_GROUP_0
    def _define_axial_net_connectivity_from_yaml(self) -> None:
        ''' Define connections between neurons in the network '''

        if not self.params.topology.include_cpg_axial:
            logging.warning('NOT INCLUDED: AXIAL NETWORK')
            return

        logging.info('DEFINING AXIAL NETWORK')
        net_modules                        = self.params.topology.network_modules
        extracond_trunk_tail_discontinuity = self._trunk_tail_discontinuity_condition()

        # AXIAL-TO-AXIAL CONNECTIONS
        logging.info('Defining axial to axial connections')

        connectivity_ax2ax = self.params.topology.connectivity_axial.get('ax2ax')
        for connection_specs in connectivity_ax2ax:
            self.define_connections_from_specs(
                connection_specs = connection_specs,
                extracond        = extracond_trunk_tail_discontinuity,
            )

        # RETICULOSPINAL CONNECTIONS
        if not self.params.topology.include_reticulospinal_axial:
            logging.warning('NOT INCLUDED: reticulospinal connections to axial network')
        else:
            logging.info('Defining reticulospinal connections to axial network')

            connectivity_rs2ax = self.params.topology.connectivity_axial.get('rs2ax')
            if len(connectivity_rs2ax) != net_modules['rs']['axial'].pools:
                raise ValueError('Connectivity and Pools mismatch')

            rs2ax_pars = zip(
                connectivity_rs2ax,
                net_modules['rs']['axial'].pools_positions_limits
            )
            for rs_pool_id, (connection_specs, rs_pool_ylimits) in enumerate(rs2ax_pars):
                self.define_connections_from_specs(
                    connection_specs   = connection_specs,
                    pre_ind_limits     = net_modules['rs']['axial'].indices,
                    pre_pool_id        = rs_pool_id,
                    post_y_neur_limits = rs_pool_ylimits,
                )

        # ASCENDING FEEDBACK
        if not self.params.topology.ascfb or not self.params.topology.include_reticulospinal_axial:
            logging.warning('NOT INCLUDED: ascending feedback connections from the axial network')
        else:
            logging.info('Defining ascending feedback connections from the axial network')

            connectivity_ax2rs = self.params.topology.connectivity_axial.get('ax2rs')

            extraconditions = [
                f'''
                (
                    ( y_neur_pre >= {float(y_lim[0])} * metre and y_neur_pre <= {float(y_lim[-1])} * metre )
                    and
                    ( pool_id_post == {post_pool_id} )
                )
                '''.replace('\n', ' ').replace('    ', '')
                for post_pool_id, y_lim in enumerate(net_modules['rs']['axial'].pools_positions_limits)
            ]
            extraconditions = f'''( {' or '.join( extraconditions )} )'''

            for connection_specs in connectivity_ax2rs:
                self.define_connections_from_specs(
                    connection_specs = connection_specs,
                    post_ind_limits  = net_modules['rs']['axial'].indices,
                    extracond        = extraconditions,
                )

        # MOTOR NEURONS
        if not self.params.topology.include_motor_neurons_axial:
            logging.warning('NOT INCLUDED: motor neuron connections for the axial network')
        else:
            logging.info('Defining motor neuron connections for the axial network')

            connectivity_ax2mn = self.params.topology.connectivity_axial.get('ax2mn')
            for connection_specs in connectivity_ax2mn:
                self.define_connections_from_specs(
                    connection_specs = connection_specs,
                    extracond        = extracond_trunk_tail_discontinuity
                )

        # PROPRIOSENSORY NEURONS TO AXIS
        if not self.params.topology.include_proprioception_axial:
            logging.warning('NOT INCLUDED: propriosensory neuron connections to axial network')
        else:
            logging.info('Defining propriosensory neuron connections to axial network')

            connectivity_ps2ax = self.params.topology.connectivity_axial.get('ps2ax')
            for connection_specs in connectivity_ps2ax:
                self.define_connections_from_specs(
                    connection_specs = connection_specs,
                    extracond        = extracond_trunk_tail_discontinuity
                )

        # EXTEROSENSORY NEURONS
        if not self.params.topology.include_exteroception_axial:
            logging.warning('NOT INCLUDED: exterosensory neuron connections to axial network')
        else:
            logging.info('Defining exterosensory neuron connections to axial network')

            connectivity_es2ax = self.params.topology.connectivity_axial.get('es2ax')
            for connection_specs in connectivity_es2ax:
                self.define_connections_from_specs(
                    connection_specs = connection_specs,
                    extracond        = extracond_trunk_tail_discontinuity
                )

    ## NEURON_GROUP_0 --> NEURON_GROUP_1
    def _define_axial_mc_connectivity_from_yaml(self) -> None:
        ''' Connectivity between the axial network to motor cells '''

        if not self.params.topology.include_motor_neurons_axial \
                or not self.params.topology.include_muscle_cells_axial:
            logging.warning('NOT INCLUDED: axial muscle cells connections')
        else:
            logging.info('Defining axial muscle cells connections')

            extracond_trunk_tail_discontinuity = self._trunk_tail_discontinuity_condition()
            connectivity_mn2mc                 = self.params.topology.connectivity_axial.get('mn2mc')
            for connection_specs in connectivity_mn2mc:
                self.define_connections_from_specs(
                    connection_specs = connection_specs,
                    extracond        = extracond_trunk_tail_discontinuity,
                )

    ## ALL CONNECTIONS
    def define_axial_connectivity_from_yaml(self) -> None:
        ''' Defines the axial connectivity for all neuron groups '''
        self._define_axial_net_connectivity_from_yaml()
        self._define_axial_mc_connectivity_from_yaml()

    ## FUNCTIONALITIES
    def _trunk_tail_discontinuity_condition(self):
        ''' Weak connections between trunk and tail '''

        if self.params.topology.trunk_tail_discontinuity_flag == 1:
            seg_row_h = float( self.params.topology.height_segment_row )
            y_limb    = float( self.params.topology.limbs_pairs_y_positions[1] )

            # Above girdle
            y_neur_pre_f  = y_limb - 3 * seg_row_h
            y_neur_post_f = y_limb + 3 * seg_row_h

            # Below girdle
            y_neur_pre_s  = y_limb + 0 * seg_row_h
            y_neur_post_s = y_limb - 3 * seg_row_h

            tol = 0.01

            # Overall condition
            extracond_trunk_tail_discontinuity_flag = f'''
            (
                (
                    (y_neur_pre  < ({y_neur_pre_f}  + {tol}*{seg_row_h}) * metre) and
                    (y_neur_post < ({y_neur_post_f} + {tol}*{seg_row_h}) * metre)
                )
                or
                (
                    (y_neur_pre  > ({y_neur_pre_s}  - {tol}*{seg_row_h}) * metre) and
                    (y_neur_post > ({y_neur_post_s} - {tol}*{seg_row_h}) * metre)
                )
            )
            '''.replace('\n','').replace('    ','')
        else:
            extracond_trunk_tail_discontinuity_flag = ''

        return extracond_trunk_tail_discontinuity_flag

# TEST
def main():
    ''' Test case '''

    import numpy as np
    import matplotlib.pyplot as plt
    from queue import Queue
    import network_modules.plotting.plots_snn as plotting
    from network_modules.parameters.network_parameters import SnnParameters

    logging.info('TEST: SNN Connectivity Axis ')

    connectivity_axial = SnnConnectivityAxis(
        network_name = 'snn_core_test',
        params_name  = 'pars_simulation_test',
        q_in         = Queue(),
        q_out        = Queue(),
    )

    connectivity_axial._define_neuronal_identifiers()
    connectivity_axial.define_axial_connectivity_from_yaml()

    # PLOTTING
    sub_modules_0 = [
        sub_sub_mod
        for module in connectivity_axial.params.topology.network_modules.sub_parts_list
        for sub_mod in module.sub_parts_list
        for sub_sub_mod in sub_mod.sub_parts_list
        if sub_sub_mod.neuron_group == 0
    ]
    sub_modules_1 = [
        sub_sub_mod
        for module in connectivity_axial.params.topology.network_modules.sub_parts_list
        for sub_mod in module.sub_parts_list
        for sub_sub_mod in sub_mod.sub_parts_list
        if sub_sub_mod.neuron_group == 1
    ]

    connectivity_axial.params.topology.plotting()

    connectivity_axial.get_wmat()
    if np.any(connectivity_axial.wmat):
        plt.figure('Connectivity matrix')
        plotting.plot_connectivity_matrix(
            pop_i                  = connectivity_axial.pop,
            pop_j                  = connectivity_axial.pop,
            w_syn                  = connectivity_axial.wmat,
            network_modules_list_i = sub_modules_0,
            network_modules_list_j = sub_modules_0,
        )

    connectivity_axial.get_wmat_mc()
    if np.any(connectivity_axial.wmat_mc):
        plt.figure('Connectivity matrix - Muscle cells')
        plotting.plot_connectivity_matrix(
            pop_i                  = connectivity_axial.pop,
            pop_j                  = connectivity_axial.mc_pop,
            w_syn                  = connectivity_axial.wmat_mc,
            network_modules_list_i = sub_modules_0,
            network_modules_list_j = sub_modules_1,
        )

    plt.show()

    return connectivity_axial

if __name__ == '__main__':
    main()
