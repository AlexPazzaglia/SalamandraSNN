'''
Network simulation in open loop with a single 1DOF limb
'''

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

        # if self.step_desired_time(curtime, 0.9*self.params.simulation.duration):
        #     print(curtime)

        # # Drive current
        # self.step_ramp_current_limbs(curtime, 0.0, 3.0)

        # Online activity estimation
        self.step_compute_pools_activation_limbs_online(curtime)

        # Online frequency  and duty cycle estimation
        self.step_compute_frequency_limbs_online(curtime)
        self.step_compute_duty_limbs_online(curtime)