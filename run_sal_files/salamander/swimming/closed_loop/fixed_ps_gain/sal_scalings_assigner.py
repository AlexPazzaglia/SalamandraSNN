import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_modules.parameters.network_parameters import SnnParameters

SYN_GROUPS = {
    'weight_ampa': 0,
    'weight_nmda': 0,
    'weight_glyc': 1,
}

def get_sal_run_params(
    default_parameters: SnnParameters,
    sal_problem       : dict,
    sampled_values    : list[float],
):
    ''' Get run parameters for sensitivity analysis '''

    sal_variables_keys : list[str] = sal_problem['names']

    ner_variable_keys : list[str] = [
        var_name
        for var_name in sal_variables_keys
        if var_name not in ['weight_ampa', 'weight_nmda', 'weight_glyc']
    ]
    ner_variable_inds : list[int] = [
        sal_variables_keys.index(var_name)
        for var_name in ner_variable_keys
    ]

    syn_variable_keys : list[str] = [
        var_name
        for var_name in sal_variables_keys
        if var_name in ['weight_ampa', 'weight_nmda', 'weight_glyc']
    ]
    syn_variable_inds : list[int] = [
        sal_variables_keys.index(var_name)
        for var_name in syn_variable_keys
    ]

    # Replace neural variables
    pars_top    = default_parameters.topology
    pars_ner    = default_parameters.neurons

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
                    for var_ind, var_name in zip(ner_variable_inds, ner_variable_keys)
                ],
                'synaptic_params' : [
                    {
                        'syn_group_ind': SYN_GROUPS[var_name],
                        'indices_i'    : pars_top.network_modules['cpg']['axial'].indices,
                        'indices_j'    : pars_top.network_modules['cpg']['axial'].indices,
                        'var_name'     : var_name,
                        'nominal_value': [0.35, ''],
                        'scaling'      : sampled_values_run[var_ind],
                    }
                    for var_ind, var_name in zip(syn_variable_inds, syn_variable_keys)
                ],
            },
        }
        for sampled_values_run in sampled_values
    ]

    return params_run