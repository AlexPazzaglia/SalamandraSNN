'''
Functions to specify the connectivity within the network
'''
import logging
import numpy as np

from network_modules.equations import connections
from network_modules.connectivity.network_connectivity_core import SnnConnectivityCore

PRINTAUX = '-'*3 + ' '

class SnnConnectivityLimbs(SnnConnectivityCore):
    '''
    Class to define the connectivity for the limb network
    '''

    ## NEURON_GROUP_0 --> NEURON_GROUP_0
    def _define_limbs_net_connectivity_from_yaml(self) -> None:
        ''' Define connections between neurons in the network '''

        if not self.params.topology.include_cpg_limbs:
            logging.warning('NOT INCLUDED: LIMBS NETWORK')
            return

        logging.info('DEFINING LIMBS NETWORK')

        # LIMB TO AXIS
        if not self.params.topology.include_cpg_axial:
            logging.warning('NOT INCLUDED: limb to axial connections')
        else:
            logging.info('Defining limb to axial connections')

            connectivity_lb2ax = self.params.topology.connectivity_limbs.get('lb2ax')
            for connection_specs in connectivity_lb2ax:

                for limb_id, limb_pos in enumerate(self.params.topology.limbs_y_positions):
                    self.define_connections_from_specs(
                        connection_specs   = connection_specs,
                        pre_limb_id        = limb_id + 1,
                        post_y_neur_limits = [ 0, self.params.topology.length_axial ],
                        post_y_neur_offset = limb_pos,
                    )

        # LIMB TO LIMB
        logging.info('Defining limb to limb connections')

        extracond_samegirdle = connections.concond_equal_to_n_distant_girdle(
            direction= 0,
            exclude_recurrent= False
        )

        # INTRA-LIMB
        # Intra DOF
        connectivity_lb2lb_intra_intra = self.params.topology.connectivity_limbs.get('lb2lb_intra_limb_intra_dof')
        for connection_specs in connectivity_lb2lb_intra_intra:
            self.define_connections_from_specs(
                connection_specs = connection_specs,
                extracond        = extracond_samegirdle,
            )

        # Inter DOF
        connectivity_lb2lb_intra_inter = self.params.topology.connectivity_limbs.get('lb2lb_intra_limb_inter_dof')

        for connection_specs in connectivity_lb2lb_intra_inter:
            intra_lb_connections = connection_specs['parameters']['intra_lb_connections']

            extracond_list = []
            for dof_strt in range(self.params.topology.segments_per_limb):
                for dof_stop in range(self.params.topology.segments_per_limb):

                    if intra_lb_connections[dof_strt][dof_stop] == 1:

                        dof_cond_aux = '( abs(side_id_{}) // 2 == {} )'

                        extracond_list.append(
                            '( '
                            + extracond_samegirdle                      + ' and '
                            + dof_cond_aux.format( 'pre', dof_strt + 1) + ' and '
                            + dof_cond_aux.format('post', dof_stop + 1) +
                            ')'
                        )

            extracond = ' or '.join(extracond_list)
            extracond = f'( {extracond} )' if extracond != '' else ''

            self.define_connections_from_specs(
                connection_specs = connection_specs,
                extracond        = extracond,
            )

        # INTER-LIMB (NOTE: GAIT-DEPENDENT)
        extracond_onegirdle = connections.concond_less_than_n_distant_girdle(
            range= 2,
            exclude_recurrent= True,
        )

        connectivity_lb2lb_inter_limb_intra_dof = self.params.topology.connectivity_limbs.get('lb2lb_inter_limb_intra_dof')
        for connection_specs in connectivity_lb2lb_inter_limb_intra_dof:
            connection_specs['parameters']['amp'] = getattr(
                self.params.topology.pars_limb_conn,
                connection_specs['parameters']['amp']
            )
            self.define_connections_from_specs(
                connection_specs = connection_specs,
                extracond        = extracond_onegirdle,
            )

        # RETICULOSPINAL TO LIMBS
        if not self.params.topology.include_reticulospinal_limbs:
            logging.warning('NOT INCLUDED: reticulospinal connections to limb networks')
        else:

            connectivity_rs2lb = self.params.topology.connectivity_limbs.get('rs2lb')
            for connection_specs in connectivity_rs2lb:

                self.define_connections_from_specs( connection_specs = connection_specs )

        # LIMBS TO RETICULOSPINAL
        if not self.params.topology.ascfb_lb or not self.params.topology.include_reticulospinal_limbs:
            logging.warning('NOT INCLUDED: ascending feedback connections from the limb networks')
        else:

            connectivity_lb2rs = self.params.topology.connectivity_limbs.get('lb2rs')
            for connection_specs in connectivity_lb2rs:
                self.define_connections_from_specs( connection_specs = connection_specs )

        # LIMBS TO MOTOR NEURONS
        if not self.params.topology.include_motor_neurons_limbs:
            logging.warning('NOT INCLUDED: motor neuron connections for the limb networks')
        else:

            connectivity_lb2mn = self.params.topology.connectivity_limbs.get('lb2mn')
            for connection_specs in connectivity_lb2mn:
                self.define_connections_from_specs( connection_specs = connection_specs )


        # PROPRIOSENSORY TO LIMBS
        if not self.params.topology.include_proprioception_limbs:
            logging.warning('NOT INCLUDED: propriosensory neuron connections to limb networks')
        else:

            connectivity_ps2lb = self.params.topology.connectivity_limbs.get('ps2lb')
            for connection_specs in connectivity_ps2lb:
                self.define_connections_from_specs( connection_specs = connection_specs )


        # EXTEROSENSORY NEURONS TO LIMBS
        if not self.params.topology.include_exteroception_limbs:
            logging.warning('NOT INCLUDED: exterosensory neuron connections to limb networks')
        else:
            logging.info('Defining exterosensory neuron connections to limb networks')
            logging.warning('UNDEFINED exterosensory neuron connections to limb networks')

    ## NEURON_GROUP_0 --> NEURON_GROUP_1
    def _define_limbs_mc_connectivity_from_yaml(self) -> None:
        ''' Connectivity from the axial network to motor cells '''

        if not self.params.topology.include_motor_neurons_limbs \
                or not self.params.topology.include_muscle_cells_limbs:
            logging.warning('NOT INCLUDED: limb muscle cells connections')
        else:

            connectivity_mn2mc = self.params.topology.connectivity_limbs.get('mn2mc')
            for connection_specs in connectivity_mn2mc:
                self.define_connections_from_specs( connection_specs = connection_specs )

    ## ALL CONNECTIONS
    def define_limbs_connectivity_from_yaml(self) -> None:
        ''' Defines the limb connectivity for all neuron groups '''
        self._define_limbs_net_connectivity_from_yaml()
        self._define_limbs_mc_connectivity_from_yaml()

# TEST
def main():
    ''' Test case '''

    import numpy as np
    import matplotlib.pyplot as plt
    from queue import Queue
    import network_modules.plotting.plots_snn as plotting

    logging.info('TEST: SNN Connectivity Limbs ')

    connectivity_limbs = SnnConnectivityLimbs(
        network_name = 'snn_core_test',
        params_name  = 'pars_simulation_test',
        q_in         = Queue(),
        q_out        = Queue(),
    )

    connectivity_limbs._define_neuronal_identifiers()
    connectivity_limbs.define_limbs_connectivity_from_yaml()

    # PLOTTING
    sub_modules_0 = [
        sub_sub_mod
        for module in connectivity_limbs.params.topology.network_modules.sub_parts_list
        for sub_mod in module.sub_parts_list
        for sub_sub_mod in sub_mod.sub_parts_list
        if sub_sub_mod.neuron_group == 0
    ]
    sub_modules_1 = [
        sub_sub_mod
        for module in connectivity_limbs.params.topology.network_modules.sub_parts_list
        for sub_mod in module.sub_parts_list
        for sub_sub_mod in sub_mod.sub_parts_list
        if sub_sub_mod.neuron_group == 1
    ]

    connectivity_limbs.params.topology.plotting()

    connectivity_limbs.get_wmat()
    if np.any(connectivity_limbs.wmat):
        plt.figure('Connectivity matrix')
        plotting.plot_connectivity_matrix(
            pop_i                  = connectivity_limbs.pop,
            pop_j                  = connectivity_limbs.pop,
            w_syn                  = connectivity_limbs.wmat,
            network_modules_list_i = sub_modules_0,
            network_modules_list_j = sub_modules_0,
    )

    connectivity_limbs.get_wmat_mc()
    if np.any(connectivity_limbs.wmat_mc):
        plt.figure('Connectivity matrix - Muscle cells')
        plotting.plot_connectivity_matrix(
            pop_i                  = connectivity_limbs.pop,
            pop_j                  = connectivity_limbs.mc_pop,
            w_syn                  = connectivity_limbs.wmat_mc,
            network_modules_list_i = sub_modules_0,
            network_modules_list_j = sub_modules_1,
        )

    plt.show()

    return connectivity_limbs