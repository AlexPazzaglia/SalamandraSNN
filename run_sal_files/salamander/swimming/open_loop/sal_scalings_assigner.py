import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_modules.parameters.network_parameters import SnnParameters

def get_sal_run_params(
    default_parameters: SnnParameters,
    sal_problem       : dict,
    sampled_values    : list[float],
):
    ''' Get run parameters for sensitivity analysis '''

    sal_variables_keys : list[str] = sal_problem['names']

    pars_top    = default_parameters.topology
    pars_ner    = default_parameters.neurons

    # Replace neural variables
    params_run = [
        {
            'scalings' : {
                'neural_params' : [
                    {
                        'neuron_group_ind': 0,
                        'var_name'        : var_name,
                        'indices'         : pars_top.network_modules['cpg']['axial'].indices,
                        'nominal_value'   : pars_ner.variable_neural_params_dict['cpg.axial'][var_name],
                        'scaling'         : sampled_values_run[var_ind],
                    }
                    for var_ind, var_name in enumerate(sal_variables_keys)
                ],
            },
        }
        for sampled_values_run in sampled_values
    ]

    return params_run