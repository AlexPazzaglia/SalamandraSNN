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
        ps_weight = new_pars.pop('ps_weight')

        message = f'Scale PS synaptic strength by factor {ps_weight}'
        logging.info(message)

        syn_weights_list = [

            # PS2cpg
            {
                'source_name' : 'ps.axial',
                'target_name' : 'cpg.axial',
                'weight_ampa' : [0.35 * ps_weight, ''],
                'weight_nmda' : [0.35 * ps_weight, ''],
                'weight_glyc' : [0.33 * ps_weight, ''],
            },

            # PS2mn
            {
                'source_name' : 'ps.axial',
                'target_name' : 'mn.axial',
                'weight_ampa' : [0.35 * ps_weight, ''],
                'weight_nmda' : [0.35 * ps_weight, ''],
                'weight_glyc' : [0.33 * ps_weight, ''],
            },

        ]

        self._assign_synaptic_weights_by_list(
            syn_weights_list = syn_weights_list,
            std_val         = 0.0,
        )

        # Update network parameters
        super().update_network_parameters_from_dict(new_pars)


    # Callback function
    def step_function(self, curtime: VariableView) -> None:
        '''
        When used (include_callback = True), called at every integration step.\n
        Exchanges data with external thread via input and output queues.
        '''

        # Silence feedback
        self.step_feedback_toggle( curtime, self.params.simulation.duration * 0/2, active= False )
        self.step_feedback_toggle( curtime, self.params.simulation.duration * 1/2, active= True )

        # Send handshake
        self.q_out.put(True)

        # Proprioceptive feedback
        self.get_mechanics_input()

        # Final handshake
        self.queue_handshake()

        return