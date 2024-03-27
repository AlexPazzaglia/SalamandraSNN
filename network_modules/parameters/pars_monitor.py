''' Monitor parameters '''
import logging
from network_modules.parameters.pars_utils import SnnPars
from network_modules.parameters.pars_simulation import SnnParsSimulation
from network_modules.parameters.pars_topology import SnnParsTopology


class SnnParsMonitor(SnnPars):
    ''' Monitoring parameters for plots and recording '''

    def __init__(
        self,
        parsname : str,
        sim_pars : SnnParsSimulation,
        top_pars : SnnParsTopology,
        new_pars : dict = None,
        pars_path: str = None,
        **kwargs
    ):
        if pars_path is None:
            pars_path = 'network_parameters/parameters_monitor'

        super().__init__(
            pars_path = pars_path,
            parsname  = parsname,
            new_pars  = new_pars,
            pars_type = 'parameters_monitor',
            **kwargs
        )

        self.pars = self._substitute_monitor_rate(self.pars, sim_pars)

        # Define monitors
        self.spikes           : dict = self.pars.pop('monitor_spikes')
        self.pools_activation : dict = self.pars.pop('monitor_poolsactivation')
        self.freq_evolution   : dict = self.pars.pop('monitor_freqevolution')
        self.ipl_evolution    : dict = self.pars.pop('monitor_iplevolution')
        self.states           : dict = self.pars.pop('monitor_states')
        self.muscle_cells     : dict = self.pars.pop('monitor_musclecells')
        self.connectivity     : dict = self.pars.pop('monitor_connectivity')
        self.farms_simulation : dict = self.pars.pop('monitor_farmsim')
        self.online_metrics   : dict = self.pars.pop('monitor_online_metrics')

        # Update values based on core parameters
        if not top_pars.include_muscle_cells_axial and not top_pars.include_muscle_cells_limbs:
            self.muscle_cells['active'] = False

        if not sim_pars.include_callback:
            self.farms_simulation['active'] = False

        if not sim_pars.include_online_act:
            self.online_metrics['active'] = False

        if not top_pars.include_cpg_axial:
            self.freq_evolution   ['active']              = False
            self.ipl_evolution    ['active']              = False
            self.pools_activation ['plotpars']['animate'] = False
            self.states           ['plotpars']['animate'] = False

        # Consistency checks
        self.consistency_checks()

    def _substitute_monitor_rate(
        self,
        pars_monitor: dict[str,dict],
        sim_pars    : SnnParsSimulation,
    ) -> dict[str, dict]:
        ''' Substitute the monitor rate with a multiple of the used timestep '''
        for monitor_vals in pars_monitor.values():
            if 'rate' in monitor_vals:
                monitor_vals['rate'] = monitor_vals['rate'] * sim_pars.timestep
        return pars_monitor


# TEST
def main():
    ''' Return test parameters '''
    logging.info('TEST: Monitor Parameters')

    sim_pars = SnnParsSimulation(parsname= 'pars_simulation_test')
    top_pars = SnnParsTopology(parsname= 'pars_topology_test')

    pars = SnnParsMonitor(
        parsname= 'pars_monitor_test',
        sim_pars= sim_pars,
        top_pars= top_pars,
    )
    return pars

if __name__ == '__main__':
    main()
