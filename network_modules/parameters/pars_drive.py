''' Parameters of the drive to the populations'''
import logging
import numpy as np
from typing import Union

from network_modules.parameters.pars_utils import SnnPars
from network_modules.parameters.pars_simulation import SnnParsSimulation
from network_modules.parameters.pars_topology import SnnParsTopology

ALL_DRIVES_T = tuple[dict[str, Union[str, list[dict]]]]
class SnnParsDrive(SnnPars):
    ''' Defines the drive to the network '''

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
            pars_path = 'network_parameters/parameters_drive'

        super().__init__(
            pars_path = pars_path,
            parsname  = parsname,
            new_pars  = new_pars,
            pars_type = 'parameters_drive',
            **kwargs
        )

        # Check connectivity and applied offsets
        scheme  = top_pars.limb_connectivity_scheme
        schemes = top_pars.pars_limb_conn.limb_connectivity_schemes
        schemes_with_offset = self.pars.pop('limb_connectivity_schemes_with_drive_offset')

        if len( schemes ) > 0 and \
            top_pars.limb_connectivity_scheme not in schemes:
            raise ValueError('Connectivity scheme is not listed')

        # Additional factors for lateral and diagonal
        lfac_max = self.pars.pop('lfac_max')
        dfac_max = self.pars.pop('dfac_max')

        self.__lfac_max = lfac_max if scheme in schemes_with_offset else 0
        self.__dfac_max = dfac_max if scheme in schemes_with_offset else 0

        # Get drives
        self.gaits_drives_all : ALL_DRIVES_T = tuple(self.pars.pop('drives'))
        self.update_gait_drives(sim_pars.gait, top_pars.ascending_feedback_str)

        # Consistency checks
        self.consistency_checks()

    def update_gait_drives(
        self,
        gait        : str,
        ascending_fb: str,
    ):
        '''
        Updates drives based on selected gait, feedback strategy, gains
        '''
        # Target gait
        gait_ind_list = [
            i for i, gait_drives in enumerate(self.gaits_drives_all)
            if gait_drives['gait'] == gait
        ]
        if len(gait_ind_list) != 1:
            raise ValueError('Cannot determine the target gait drive')

        gait_index = gait_ind_list[0]

        # Target mode
        gait_modes = self.gaits_drives_all[gait_index]['modes']

        mode_ind_list = [
            i for i, gait_mode in enumerate(gait_modes)
            if gait_mode['feedback'] == ascending_fb
            or gait_mode['feedback'] == 'Any'
        ]
        if len(gait_ind_list) != 1:
            raise ValueError('Cannot determine the target gait mode')

        mode_index = mode_ind_list[0]

        # Get drives
        self.gait_drives : dict = gait_modes[mode_index]
        self.gait_drives['drives_axis']  = np.array(self.gait_drives['drives_axis'])
        self.gait_drives['drives_limbs'] = np.array(self.gait_drives['drives_limbs'])

        self.lfac = self.lfac_max if self.gait_drives['lfac'] else 0
        self.dfac = self.dfac_max if self.gait_drives['dfac'] else 0

    ## PROPERTIES
    lfac_max : float = SnnPars.read_only_attr('lfac_max')
    dfac_max : float = SnnPars.read_only_attr('dfac_max')

# TEST
def main():
    ''' Return test parameters '''
    logging.info('TEST: Drive Parameters')

    sim_pars = SnnParsSimulation(parsname= 'pars_simulation_test')
    top_pars = SnnParsTopology(parsname= 'pars_topology_test')

    pars = SnnParsDrive(
        parsname= 'pars_drive_test',
        sim_pars= sim_pars,
        top_pars= top_pars,
    )
    return pars

if __name__ == '__main__':
    main()
