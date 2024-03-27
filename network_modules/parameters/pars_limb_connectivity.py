''' Limb connectivity parameter '''
import logging
import numpy as np
from network_modules.parameters.pars_utils import SnnPars

PROB_T = dict[str, dict[str, list[list[float]]]]
class SnnParsLimbConnectivity(SnnPars):
    '''
    Defines the connectivity between the limbs according to the chosen scheme
    '''

    def __init__(
        self,
        parsname           : str,
        connectivity_scheme: str,
        new_pars           : dict = None,
        pars_path          : str = None,
        **kwargs
    ):

        if parsname in ['None', ''] or connectivity_scheme in ['None', '']:
            self.__limb_connectivity_schemes = ()
            self.__p_connections_limbs       = {}
            self.__p_ex_inter_lb_max         = 0
            self.__p_in_inter_lb_max         = 0
            return

        if pars_path is None:
            pars_path = 'network_parameters/parameters_limb_connectivity'

        super().__init__(
            pars_path      = pars_path,
            parsname       = parsname,
            new_pars       = new_pars,
            keys_to_update = [connectivity_scheme],
            pars_type      = 'parameters_limb_connectivity',
            **kwargs
        )

        # Parameters
        self.__limb_connectivity_schemes = tuple( self.pars.keys() )
        self.pars : dict = self.pars[connectivity_scheme]

        self.__p_connections_limbs : PROB_T = self.pars.pop('p_connections_limbs')
        self.__p_connections_limbs = self._substitute_probability_values(
            p_connections= self.p_connections_limbs,
            pars= self.pars,
        )

        # Max values
        self.__p_ex_inter_lb_max : float = np.amax(
            [
                np.amax( p_connections_limbs_gait[conn_type] )
                for p_connections_limbs_gait in self.p_connections_limbs.values()
                for conn_type in ['ex_f2f','ex_e2e','ex_f2e','ex_e2f',]
                if p_connections_limbs_gait[conn_type] is not None
            ],
            initial= 0.0
        )

        self.__p_in_inter_lb_max : float = np.amax(
            [
                np.amax( p_connections_limbs_gait[conn_type] )
                for p_connections_limbs_gait in self.p_connections_limbs.values()
                for conn_type in ['in_f2f','in_e2e','in_f2e','in_e2f',]
                if  p_connections_limbs_gait[conn_type] is not None
            ],
            initial= 0.0
        )

    def _substitute_probability_values(
        self,
        p_connections: PROB_T,
        pars         : dict,
    ) -> PROB_T:

        for gait_prob in p_connections.values():
            for pool_prob in gait_prob.values():

                if pool_prob is None:
                    continue

                n_limbs = len(pool_prob)

                for i_limb in range(n_limbs):
                    for j_limb in range(n_limbs):

                        prob = pars.get(
                            pool_prob[i_limb][j_limb],
                            None
                        )

                        if prob is None:
                            continue

                        pool_prob[i_limb][j_limb] = prob

        return p_connections

    # PROPERTIES
    limb_connectivity_schemes : tuple[str] = SnnPars.read_only_attr('limb_connectivity_schemes')
    p_connections_limbs       : PROB_T     = SnnPars.read_only_attr('p_connections_limbs')
    p_ex_inter_lb_max         : float      = SnnPars.read_only_attr('p_ex_inter_lb_max')
    p_in_inter_lb_max         : float      = SnnPars.read_only_attr('p_in_inter_lb_max')

# TEST
def main():
    ''' Return test parameters '''
    logging.info('TEST: Limb Connectivity Parameters')
    pars = SnnParsLimbConnectivity(
        parsname= 'pars_limb_connectivity_test',
        connectivity_scheme= 'reference',
    )
    return pars

if __name__ == '__main__':
    main()
