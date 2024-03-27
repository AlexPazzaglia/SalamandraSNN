'''
Neuronal parameters

This module defines the SnnParsNeurons class for configuring parameters of neuronal models.
'''
import logging
from brian2 import second
from network_modules.parameters.pars_utils import SnnPars

UNITS_T  = dict[str, str]
VALUES_T = dict[str, list[float, str]]

class SnnParsNeurons(SnnPars):
    '''
    Parameters for the neuronal models.

    This class defines parameters for neuronal models, including common and specific parameters for neurons and synapses.

    There are two types of parameters:
    - Common parameters are shared among neurons/synapses.
    - Specific parameters differ depending on the neuron/synapse type.
      They require an additional variable to be added to the models.

    Args:
        parsname (str): The name of the parameters.
        new_pars (dict, optional): New parameters to add. Default is None.
        pars_path (str, optional): Path to the parameter file. Default is None.
        **kwargs: Additional keyword arguments passed to the superclass constructor.

    Attributes:
        neuron_type_network (str): Neuron type for the network.
        neuron_type_muscle (str): Neuron type for muscles.
        synaptic_labels (tuple[str]): Labels for synaptic connections.
        n_adaptation_variables (int): Number of adaptation variables.
        std_val (float): Standard deviation of stochastic parameters.

        shared_neural_params (tuple[VALUES_T]): Shared neural parameters.
        variable_neural_params_list (tuple[VALUES_T]): List of variable neural parameters.
        variable_neural_params_units (tuple(UNITS_T)): Units of variable neural parameters.
    '''

    def __init__(
        self,
        parsname : str,
        new_pars : dict = None,
        pars_path: str = None,
        **kwargs
    ):
        if pars_path is None:
            pars_path = 'network_parameters/parameters_neurons'

        super().__init__(
            pars_path = pars_path,
            parsname  = parsname,
            new_pars  = new_pars,
            pars_type = 'parameters_neurons',
            **kwargs
        )

        # Define parameters
        self.__neuron_type_network    = self.pars.pop('neuron_type_network')
        self.__neuron_type_muscle     = self.pars.pop('neuron_type_muscle')
        self.__synaptic_labels        = tuple(self.pars.pop('synaptic_labels'))
        self.__n_adaptation_variables = self.pars.pop('n_adaptation_variables')

        # COMMON
        self.__shared_neural_params = tuple( self.pars.pop('shared_neural_params') )

        # VARIABLE
        self.__variable_neural_params_units = tuple( self.pars.pop('variable_neural_params_units') )
        self.__variable_neural_params_list  = tuple( self.pars.pop('variable_neural_params_list') )
        self.__variable_neural_params_dict  = {
            variable_pars['mod_name'] : variable_pars
            for variable_pars in self.variable_neural_params_list
        }

        # Standard deviation of stochastic parameters
        std_aux        = self.shared_neural_params[0].get('std_val')
        self.__std_val = std_aux[0] if std_aux is not None else 0

        # Consistency checks
        self.consistency_checks()

    # PROPERTIES
    neuron_type_network          : str        = SnnPars.read_only_attr('neuron_type_network')
    neuron_type_muscle           : str        = SnnPars.read_only_attr('neuron_type_muscle')
    synaptic_labels              : tuple[str] = SnnPars.read_only_attr('synaptic_labels')
    n_adaptation_variables       : int        = SnnPars.read_only_attr('n_adaptation_variables')
    std_val                      : float      = SnnPars.read_only_attr('std_val')

    shared_neural_params         : tuple[VALUES_T]     = SnnPars.read_only_attr('shared_neural_params')
    variable_neural_params_list  : tuple[VALUES_T]     = SnnPars.read_only_attr('variable_neural_params_list')
    variable_neural_params_dict  : dict[str, VALUES_T] = SnnPars.read_only_attr('variable_neural_params_dict')
    variable_neural_params_units : tuple[UNITS_T]      = SnnPars.read_only_attr('variable_neural_params_units')

# TEST
def main():
    ''' Return test parameters '''
    logging.info('TEST: Neurons Parameters')
    pars = SnnParsNeurons(
        parsname= 'pars_neurons_test',
    )
    return pars

if __name__ == '__main__':
    main()
