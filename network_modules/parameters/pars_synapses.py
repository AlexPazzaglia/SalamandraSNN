'''
Synaptic parameters

This module defines the SnnParsSynapses class for configuring parameters of synaptic models.
'''
import logging
from brian2 import meter, second
from network_modules.parameters.pars_utils import SnnPars

UNITS_T  = dict[str, str]
VALUES_T = dict[str, tuple[float, str]]
WEIGHT_T = dict[str, tuple[float, str]]

class SnnParsSynapses(SnnPars):
    '''
    Parameters for the synapses.

    This class defines parameters for synaptic models, including common and specific parameters for synapses.

    There are two types of parameters:
    - Common parameters are shared among neurons/synapses.
    - Specific parameters differ depending on the synapse type.
      They require an additional variable to be added to the models.

    Args:
        parsname (str): The name of the parameters.
        new_pars (dict, optional): New parameters to add. Default is None.
        pars_path (str, optional): Path to the parameter file. Default is None.
        **kwargs: Additional keyword arguments passed to the superclass constructor.

    Attributes:
        synaptic_labels_ex (tuple[str]): Labels for excitatory synaptic connections.
        synaptic_labels_in (tuple[str]): Labels for inhibitory synaptic connections.
        synaptic_labels (tuple[str]): All synaptic connection labels.
        conduction_speed (float): Conduction speed of synapses.
        synaptic_delay_nominal (float): Nominal synaptic delay.
        synaptic_delay_muscle (float): Synaptic delay for muscle synapses.
        include_conduction_delay (float): Flag indicating whether to include conduction delay.
        syndel (float): Effective synaptic delay based on parameters.

        shared_syn_params (tuple[VALUES_T]): Shared synaptic parameters.
        variable_syn_params_units (tuple[UNITS_T]): Units of variable synaptic parameters.
        variable_syn_params (tuple[VALUES_T]): List of variable synaptic parameters.
        syn_weights_list (tuple[WEIGHT_T]): List of synaptic weight parameters.
    '''

    def __init__(
        self,
        parsname : str,
        new_pars : dict = None,
        pars_path: str = None,
        **kwargs
    ):
        if pars_path is None:
            pars_path = 'network_parameters/parameters_synapses'

        super().__init__(
            pars_path = pars_path,
            parsname  = parsname,
            new_pars  = new_pars,
            pars_type = 'parameters_synapses',
            **kwargs
        )

        # Define parameters
        self.__synaptic_labels_ex = tuple(self.pars.pop('synaptic_labels_ex'))
        self.__synaptic_labels_in = tuple(self.pars.pop('synaptic_labels_in'))
        self.__synaptic_labels    = self.__synaptic_labels_ex + self.__synaptic_labels_in

        self.__conduction_speed         = float(self.pars.pop('conduction_speed'))       * meter / second
        self.__synaptic_delay_nominal   = float(self.pars.pop('synaptic_delay_nominal')) * second
        self.__synaptic_delay_muscle    = float(self.pars.pop('synaptic_delay_muscle'))  * second
        self.__include_conduction_delay = self.pars.pop('include_conduction_delay')

        # COMMON
        self.__shared_syn_params = tuple( self.pars.pop('shared_syn_params') )

        # VARIABLE
        self.__variable_syn_params_units = tuple( self.pars.pop('variable_syn_params_units') )
        self.__variable_syn_params       = tuple( self.pars.pop('variable_syn_params') )

        # SYNAPTIC WEIGHTS
        self.__syn_weights_list = tuple( self.pars.pop('syn_weights_list') )

        # DERIVED PARAMETERS
        if self.include_conduction_delay:
            self.__syndel = f'''
                {self.synaptic_delay_nominal} * second
                + abs(y_neur_pre - y_neur_post) / {self.conduction_speed} * second
            '''.replace('\n', '').replace('    ','')

        else:
            self.__syndel = self.synaptic_delay_nominal

        # Consistency checks
        self.consistency_checks()

    ## PROPERTIES
    synaptic_labels_ex           : tuple[str] = SnnPars.read_only_attr('synaptic_labels_ex')
    synaptic_labels_in           : tuple[str] = SnnPars.read_only_attr('synaptic_labels_in')
    synaptic_labels              : tuple[str] = SnnPars.read_only_attr('synaptic_labels')
    conduction_speed             : float      = SnnPars.read_only_attr('conduction_speed')
    synaptic_delay_nominal       : float      = SnnPars.read_only_attr('synaptic_delay_nominal')
    synaptic_delay_muscle        : float      = SnnPars.read_only_attr('synaptic_delay_muscle')
    include_conduction_delay     : float      = SnnPars.read_only_attr('include_conduction_delay')
    syndel                       : float      = SnnPars.read_only_attr('syndel')

    shared_syn_params            : tuple[VALUES_T] = SnnPars.read_only_attr('shared_syn_params')
    variable_syn_params_units    : tuple[UNITS_T]  = SnnPars.read_only_attr('variable_syn_params_units')
    variable_syn_params          : tuple[VALUES_T] = SnnPars.read_only_attr('variable_syn_params')
    syn_weights_list             : tuple[WEIGHT_T] = SnnPars.read_only_attr('syn_weights_list')

# TEST
def main():
    ''' Return test parameters '''
    logging.info('TEST: Synapses Parameters')
    pars = SnnParsSynapses(
        parsname= 'pars_synapses_test',
    )
    return pars

if __name__ == '__main__':
    main()
