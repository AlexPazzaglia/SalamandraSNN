'''
Network simulation using farms and 1DOF limbs
'''

import logging
import numpy as np
from typing import Any

from brian2.units.allunits import pamp, msecond
from brian2.core.variables import VariableView
from network_modules.experiment.network_experiment import SnnExperiment

class SalamandraSimulation(SnnExperiment):
    '''
    Class used to run a simulation in closed loop with multi-dof limbs
    '''

    def update_network_parameters_from_dict(self, new_pars: dict) -> None:
        ''' Update network parameters from a dictionary '''

        # Transection points
        sig_ex_up         = new_pars.pop('sig_ex_up')
        sig_ex_dw         = new_pars.pop('sig_ex_dw')
        sig_in_up         = new_pars.pop('sig_in_up')
        sig_in_dw         = new_pars.pop('sig_in_dw')

        message = (
            f'Set max synaptic range of PS neurons to: '
            f'sig_ex_up = {sig_ex_up}, '
            f'sig_ex_dw = {sig_ex_dw}, '
            f'sig_in_up = {sig_in_up}, '
            f'sig_in_dw = {sig_in_dw}'
        )
        logging.info(message)

        self.set_max_synaptic_range(
            syn_group_names      = ['syn_ex'],
            range_up             = sig_ex_up,
            range_dw             = sig_ex_dw,
            included_populations = [['ps', 'cpg'], ['ps', 'mn']],
        )
        self.set_max_synaptic_range(
            syn_group_names      = ['syn_in'],
            range_up             = sig_in_up,
            range_dw             = sig_in_dw,
            included_populations = [['ps', 'cpg'], ['ps', 'mn']],
        )

        # Update network parameters
        super().update_network_parameters_from_dict(new_pars)


    # Callback function
    def step_function(self, curtime: VariableView) -> None:
        '''
        When used (include_callback = True), called at every integration step.\n
        Exchanges data with external thread via input and output queues.
        '''

        # Send handshake
        self.q_out.put(True)

        # Proprioceptive feedback
        self.get_mechanics_input()

        # Final handshake
        self.queue_handshake()

        return