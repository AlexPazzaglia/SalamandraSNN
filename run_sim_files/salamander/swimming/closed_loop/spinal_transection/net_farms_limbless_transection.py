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

    def __init__(self, *args, **kwargs) -> None:
        ''' Initialize the simulation '''
        self.n_transections         = None
        self.transected_populations = None
        self.transection_points     = None
        super().__init__(*args, **kwargs)


    def update_network_parameters_from_dict(self, new_pars: dict) -> None:
        ''' Update network parameters from a dictionary '''

        # PS weight
        ps_weight                   = new_pars.pop('ps_weight')

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

        # Transection points
        self.n_transections         = new_pars.pop('n_transections')
        self.transected_populations = new_pars.pop('transected_populations')
        self.transection_points     = np.linspace(
            np.amin( self.params.topology.neurons_y_mech[0] ),
            float(self.params.topology.length_axial),
            self.n_transections + 1,
            endpoint= False
        )[1:]

        for transection_point in self.transection_points:
            message = (
                f'Applying transection at {transection_point:.4f} m '
                f'involving populations {self.transected_populations}'
            )
            logging.info(message)
            self.apply_axial_transection(
                transection_s        = transection_point,
                transection_f        = transection_point,
                included_populations = self.transected_populations,
            )

        # Update network parameters
        super().update_network_parameters_from_dict(new_pars)


    # Callback function
    def step_function(self, curtime: VariableView) -> None:
        '''
        When used (include_callback = True), called at every integration step.\n
        Exchanges data with external thread via input and output queues.
        '''

        # if self.step_desired_time(curtime, target_time= 60 * msecond):
        #     print(curtime)

        # Gait transition
        # self.step_gait_transition(curtime, self.params.simulation.duration * 0/5, 'swim')
        # self.step_gait_transition(curtime, self.params.simulation.duration * 1/5, 'trot')
        # self.step_gait_transition(curtime, self.params.simulation.duration * 2/5, 'diag')
        # self.step_gait_transition(curtime, self.params.simulation.duration * 3/5, 'lat')
        # self.step_gait_transition(curtime, self.params.simulation.duration * 4/5, 'amble')

        # # Turn right, Turn left
        # self.step_update_turning(curtime, self.params.simulation.duration*1/3, +0.1)
        # self.step_update_turning(curtime, self.params.simulation.duration*2/3, -0.1)

        # Silence feedback
        self.step_feedback_toggle( curtime, self.params.simulation.duration * 0/2, active= False )
        self.step_feedback_toggle( curtime, self.params.simulation.duration * 1/2, active= True )

        # # Ramp feedback
        # self.step_ramp_ps_gains_axis(curtime, 0, 10.0)

        # # Silence drive
        # self.step_drive_toggle( curtime, self.params.simulation.duration * 1/2, active= False )

        # # Ramp drive
        # self.step_ramp_current_axis(curtime, 0, 4.0)

        # Send handshake
        self.q_out.put(True)

        # Proprioceptive feedback
        self.get_mechanics_input()

        # Final handshake
        self.queue_handshake()

        return