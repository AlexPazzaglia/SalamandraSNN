'''
Network simulation using farms and 1DOF limbs
'''

import numpy as np
from typing import Any

from brian2.units.allunits import pamp
from brian2.core.variables import VariableView
from network_modules.experiment.network_experiment import SnnExperiment

class SalamandraSimulation(SnnExperiment):
    '''
    Class used to run a simulation in closed loop with multi-dof limbs
    '''

    # Callback function

    def step_function(self, curtime: VariableView) -> None:
        '''
        When used (include_callback = True), called at every integration step.\n
        Exchanges data with external thread via input and output queues.
        '''

        # Gait transition
        # self.step_gait_transition(curtime, self.params.simulation.duration * 0/5, 'swim')
        # self.step_gait_transition(curtime, self.params.simulation.duration * 1/5, 'trot')
        # self.step_gait_transition(curtime, self.params.simulation.duration * 2/5, 'diag')
        # self.step_gait_transition(curtime, self.params.simulation.duration * 3/5, 'lat')
        # self.step_gait_transition(curtime, self.params.simulation.duration * 4/5, 'amble')

        # # Turn right, Turn left
        # self.step_update_turning(curtime, self.params.simulation.duration*1/3, +0.1)
        # self.step_update_turning(curtime, self.params.simulation.duration*2/3, -0.1)

        # # Silence feedback
        # self.step_feedback_toggle( curtime, self.params.simulation.duration * 0/2, active= False )
        # self.step_feedback_toggle( curtime, self.params.simulation.duration * 1/2, active= True )

        # Silence drive
        # self.step_drive_toggle( curtime, self.params.simulation.duration * 1/2, active= False )

        if self.params.simulation.include_online_act:
            # Online activity estimation
            self.step_compute_pools_activation_limbs_online(curtime)

            # Online frequency  and duty cycle estimation
            self.step_compute_frequency_limbs_online(curtime)
            self.step_compute_duty_limbs_online(curtime)

        # Send handshake
        self.q_out.put(True)

        # Proprioceptive feedback
        self.get_mechanics_input()

        # Final handshake
        self.queue_handshake()

        return
