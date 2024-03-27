'''
Neuronal and Synaptic Behavior Manipulation Module

This module provides functions for defining and manipulating neuronal and synaptic behavior.

Functions:
    set_synaptic_parameters_by_synaptic_inds(syn, syn_inds, std_value, parameters=None, extra_cond=None):
        Set synaptic connection parameters for specified synaptic indices.

    set_neural_parameters_by_array(pool, ids, parameters):
        Set neuron parameters using arrays of values and units for specified indices.

    set_neural_parameters_by_neural_inds(pool, ids, std_value, parameters=None, deterministic=False):
        Set neuron parameters for specified indices with options for distribution.

    set_synaptic_parameters_by_neural_ind(syn, i_inds, j_inds, std_value, parameters=None, extra_cond=None):
        Set synaptic connection parameters for specified neuronal index ranges.

    set_synaptic_parameters_by_neural_ind_ranges(syn, i_limits, j_limits, std_value, parameters=None, extra_cond=None):
        Set synaptic connection parameters for specified ranges of neuronal indices.

Notes:
    - These functions allow configuration of neuronal and synaptic behavior.
    - Use the provided functions to set parameters for specific subsets of populations.
    - Gaussian and uniform distributions can be applied to parameter values as needed.
'''
import copy

import numpy as np
import numexpr as ne
import brian2 as b2

# -------- [ UTILITY FUNCTIONS ] --------
def _get_value_to_set(
    value        : float,
    unit         : str,
    std_value    : float = 0,
    deterministic: bool  = False,
) -> str:
    '''
    Get the value to set for a given value and unit.

    Args:
        value (float): The value to set.
        unit (str): Measurement unit of the value.
        std_value (float, optional): Standard deviation value for distribution. Default is 0.
        deterministic (bool, optional): Flag indicating deterministic value or distribution. Default is False.

    Returns:
        str: Value to set, including unit.

    Notes:
        - Handles fixed values, Gaussian, and uniform distributions based on input parameters.
    '''

    unit_mul = f'*{unit}' if unit != '' else ''

    # Random
    if isinstance(value, list) and not deterministic:
        if len(value) == 1:
            # Normal distribution
            valuetoset = 'randn()*{std}*{M}{un} + {M}{un}'.format(
                M   = value[0],
                std = std_value,
                un  = unit_mul,
            )

        elif len(value) == 2:
            # Uniform distribution
            valuetoset = 'rand()*({M}-{m}){un} + {m}{un}'.format(
                m  = value[0],
                M  = value[1],
                un = unit_mul
            )

    # Deterministic
    else:
        valuetoset = f"{np.mean(value) }{unit_mul}"

    return valuetoset

def _apply_scalings_to_parameters(
    parameters: dict[str, float],
    pars_keys : list[str],
    scalings  : list[float]
) -> None:
    '''
    Auxiliary function to scale the values of a dict according to the provided scalings
    '''

    assert len(pars_keys) == len(scalings), \
        'Number of parameters and scalings must be the same.'

    pars_scaled = copy.deepcopy(parameters)

    for i, parkey in enumerate(pars_keys):

        if not isinstance(pars_scaled[parkey][0], list):
            # Deterministic values
            pars_scaled[parkey][0] = scalings[i] * pars_scaled[parkey][0]

        elif len(pars_scaled[parkey][0]) == 1:
            # Gaussian distributed values
            pars_scaled[parkey][0] = [scalings[i] * pars_scaled[parkey][0][0]]

        elif len(pars_scaled[parkey][0]) == 2:
            # Uniformly distributed values
            mindpoint = (pars_scaled[parkey][0][1] + pars_scaled[parkey][0][0])/2.0
            semi_range = (pars_scaled[parkey][0][1] - pars_scaled[parkey][0][0])
            pars_scaled[parkey][0] = [
                scalings[i] * mindpoint - semi_range,
                scalings[i] * mindpoint + semi_range
            ]

    return pars_scaled

def _evaluate_neuronal_extra_conditions(
    ner_group: b2.NeuronGroup,
    var_names: list[str],
    cond_str : str,
) -> np.ndarray:
    '''
    Evaluate expression and return array of indices that satisfy it.

    Args:
        pop (b2.NeuronGroup): The neuronal population object.
        var_names (list[str]): List of variable names.
        cond_str (str): Condition expression.

    Returns:
        np.ndarray: Array of indices satisfying the condition.
    '''
    ner_vars = {
        var_name : np.array(getattr(ner_group, var_name))
        for var_name in var_names
        if hasattr(ner_group, var_name)
    }
    cond_str_parsed = cond_str.replace(' and ', ' & ').replace(' or ', ' | ').replace(' not ', ' ~ ')
    ner_inds_mask = ne.evaluate(cond_str_parsed, local_dict=ner_vars)
    return np.where(ner_inds_mask)

def _evaluate_synaptic_extra_conditions(
    syn_group: b2.Synapses,
    var_names: list[str],
    cond_str : str,
) -> np.ndarray:
    '''
    Evaluate expression and return array of indices that satisfy it.

    Args:
        syn (b2.Synapses): The synaptic object.
        var_names (list[str]): List of variable names.
        cond_str (str): Condition expression.

    Returns:
        np.ndarray: Array of indices satisfying the condition.
    '''
    syn_vars = {
        var_name : np.array(getattr(syn_group, var_name))
        for var_name in var_names
        if hasattr(syn_group, var_name)
    }
    cond_str_parsed = cond_str.replace(' and ', ' & ').replace(' or ', ' | ').replace(' not ', ' ~ ')
    syn_inds_mask = ne.evaluate(cond_str_parsed, local_dict=syn_vars)
    return np.where(syn_inds_mask)

# \------- [ UTILITY FUNCTIONS ] --------

# -------- [ FUNCTIONS TO ASSIGN PARAMETERS' VALUES ] --------
def set_neural_parameters_by_array(
    ner_group : b2.NeuronGroup,
    ner_inds  : list[int],
    parameters: dict[str, tuple[np.ndarray, str]],
) -> b2.NeuronGroup:
    '''
    Set neuron parameters using arrays of values and units for specified indices.

    Args:
        pool (b2.NeuronGroup): The neuron group to configure.
        ids (list[int]): List of neuron indices to set parameters for.
        parameters (dict[str, tuple[np.ndarray, str]]): Dictionary specifying parameter values and units.
                                                       Format: {'parameter1': ([value_array], unit1)}.

    Returns:
        b2.NeuronGroup: The modified neuron group.
    '''

    if not isinstance(ner_inds, (list, range, np.ndarray) ) \
        or not len(ner_inds) \
            or not parameters:
        return ner_group

    for param, data in parameters.items():

        if not hasattr(ner_group, param):
            continue

        value_arr  = np.array(data[0])
        value_unit = getattr(b2.units, data[1]) if data[1] != '' else 1

        # Assign values
        pool_attr = getattr(ner_group, param)
        pool_attr[ner_inds] = value_arr * value_unit
        setattr(ner_group, param, pool_attr)

    return ner_group

def set_neural_parameters_by_neural_inds(
    ner_group    : b2.NeuronGroup,
    inds_ner     : list[int],
    std_value    : float,
    parameters   : dict = None,
    deterministic: bool = False,
    extra_cond   : tuple[list[str], str] = None,
) -> b2.NeuronGroup:
    '''
    Set neuron parameters for specified indices with options for distribution.

    Args:
        pool (b2.NeuronGroup): The neuron group to configure.
        ids (list[int]): List of neuron indices to set parameters for.
        std_value (float): Standard deviation value used for parameter configurations.
        parameters (dict, optional): Dictionary specifying parameter values and units.
                                    Format: {'parameter1': [value1, unit1]}.
                                    Numbers are treated as fixed quantities.
                                    Lists of one element indicate Gaussian distribution with std_value.
                                    Lists of two elements indicate uniform distribution between values.
        deterministic (bool, optional): Flag indicating deterministic value or distribution. Default is False.

    Returns:
        b2.NeuronGroup: The modified neuron group.
    '''

    if not isinstance(inds_ner, (list, range, np.ndarray) ) \
        or not len(inds_ner) \
            or not parameters:
        return ner_group

    # Check extra conditions
    if extra_cond is not None:
        ner_inds_mask = _evaluate_neuronal_extra_conditions(
            ner_group = ner_group,
            var_names = extra_cond[0],
            cond_str  = extra_cond[1],
        )

        inds_ner = np.intersect1d(
            inds_ner,
            ner_inds_mask,
        )

    # Substitute the desired values
    for param, data in parameters.items():

        if not isinstance(data, list) or not hasattr(ner_group, param):
            continue

        # Get value to set
        valuetoset = _get_value_to_set(
            value         = data[0],
            unit          = data[1],
            std_value     = std_value,
            deterministic = deterministic,
        )

        # Assign values
        pool_attr = getattr(ner_group, param)
        pool_attr[inds_ner] = valuetoset
        setattr(ner_group, param, pool_attr)

    return ner_group

def set_scaled_neural_parameters_by_neural_inds(
    ner_group        : b2.NeuronGroup,
    inds_ner         : list[int],
    pars_vals_nominal: dict[str, float],
    pars_keys        : list[str],
    pars_scalings    : list[float],
    pars_std_value   : float = 0.0,
    extra_cond       : tuple[list[str], str] = None,
) -> b2.NeuronGroup:
    '''
    Auxiliary function to scale and assign the neuronal
    parameters to the desired population subset, according to the provided scalings
    '''
    # Copy and scale
    scaled_pars = _apply_scalings_to_parameters(
        parameters = pars_vals_nominal,
        pars_keys  = pars_keys,
        scalings   = pars_scalings
    )

    # Assign
    return set_neural_parameters_by_neural_inds(
        ner_group  = ner_group,
        inds_ner   = inds_ner,
        std_value  = pars_std_value,
        parameters = scaled_pars,
        extra_cond = extra_cond,
    )

def set_synaptic_parameters_by_synaptic_inds(
    syn_group : b2.Synapses,
    inds_syn  : list[int],
    std_value : float,
    parameters: dict =None,
    extra_cond: tuple[list[str], str] = None,
) -> b2.Synapses:
    '''
    Set synaptic connection parameters for specified neuronal index ranges.

    Args:
        syn (b2.Synapses): The synaptic object to configure.
        syn_inds (list[int]): List of synaptic indices.
        std_value (float): Standard deviation value used for parameter configurations.
        parameters (dict, optional): Dictionary specifying parameter values and units.
                                    Format: {'parameter1': [value1, unit1]}.
                                    Numbers are treated as fixed quantities.
                                    Lists of one element indicate Gaussian distribution with std_value.
                                    Lists of two elements indicate uniform distribution between values.
        extra_cond (tuple[list[str], str], optional): Additional conditions to evaluate for setting parameters.
                                                      Tuple format: (variable_names, condition_string).

    Returns:
        b2.Synapses: The modified synaptic object.
    '''

    if not isinstance(inds_syn, (list, range, np.ndarray) ) \
        or not len(inds_syn) \
            or not parameters:
        return syn_group

    # Check extra conditions
    if extra_cond is not None:
        syn_inds_mask = _evaluate_synaptic_extra_conditions(
            syn_group = syn_group,
            var_names = extra_cond[0],
            cond_str  = extra_cond[1],
        )

        inds_syn = np.intersect1d(
            inds_syn,
            syn_inds_mask,
        )

    # Substitute the desired values
    for param, data in parameters.items():

        if not isinstance(data, list) or not hasattr(syn_group, param):
            continue

        # Get value to set
        valuetoset = _get_value_to_set(
            value     = data[0],
            unit      = data[1],
            std_value = std_value,
        )

        # Assign values
        syn_attr = getattr(syn_group, param)
        syn_attr[inds_syn] = valuetoset
        setattr(syn_group, param, syn_attr)

    return syn_group

def set_synaptic_parameters_by_neural_ind(
    syn_group : b2.Synapses,
    inds_syn_i: list[int],
    inds_syn_j: list[int],
    std_value : float,
    parameters: dict = None,
    extra_cond: tuple[list[str], str] = None,
) -> b2.Synapses:
    '''
    Set synaptic connection parameters for specified neuronal indices.

    Args:
        syn (b2.Synapses): The synaptic object to configure.
        i_inds (list[int]): List of pre-synaptic neuron indices.
        j_inds (list[int]): List of post-synaptic neuron indices.
        std_value (float): Standard deviation value used for parameter configurations.
        parameters (dict, optional): Dictionary specifying parameter values and units.
                                    Format: {'parameter1': [value1, unit1]}.
                                    Numbers are treated as fixed quantities.
                                    Lists of one element indicate Gaussian distribution with std_value.
                                    Lists of two elements indicate uniform distribution between values.
        extra_cond (tuple[list[str], str], optional): Additional conditions to evaluate for setting parameters.
                                                      Tuple format: (variable_names, condition_string).

    Returns:
        b2.Synapses: The modified synaptic object.
    '''

    if not isinstance(inds_syn_i, (list, range, np.ndarray) ) \
        or not isinstance(inds_syn_j, (list, range, np.ndarray) ) \
            or not len(inds_syn_i) * len(inds_syn_j) \
                or not parameters:
        return syn_group

    # Check whic synapses are connecting neurons of interest
    syn_inds = np.intersect1d(
        np.where( np.isin(syn_group.i, inds_syn_i) ),
        np.where( np.isin(syn_group.j, inds_syn_j) )
    )

    return set_synaptic_parameters_by_synaptic_inds(
        syn_group  = syn_group,
        inds_syn   = syn_inds,
        std_value  = std_value,
        parameters = parameters,
        extra_cond = extra_cond,
    )

def set_synaptic_parameters_by_neural_ind_ranges(
    syn_group        : b2.Synapses,
    inds_limits_syn_i: list[int],
    inds_limits_syn_j: list[int],
    std_value        : float,
    parameters       : dict = None,
    extra_cond       : tuple[list[str], str] = None
) -> b2.Synapses:
    '''
    Set synaptic connection parameters for specified ranges of neuronal indices.

    Args:
        syn (b2.Synapses): The synaptic object to configure.
        i_limits (list[int]): Range of pre-synaptic neuron indices (inclusive).
        j_limits (list[int]): Range of post-synaptic neuron indices (inclusive).
        std_value (float): Standard deviation value used for parameter configurations.
        parameters (dict, optional): Dictionary specifying parameter values and units.
                                    Format: {'parameter1': [value1, unit1]}.
                                    Numbers are treated as fixed quantities.
                                    Lists of one element indicate Gaussian distribution with std_value.
                                    Lists of two elements indicate uniform distribution between values.
        extra_cond (tuple[list[str], str], optional): Additional conditions to evaluate for setting parameters.
                                                      Tuple format: (variable_names, condition_string).

    Returns:
        b2.Synapses: The modified synaptic object.
    '''

    if not isinstance(inds_limits_syn_i, (list, range, np.ndarray) ) \
        or not isinstance(inds_limits_syn_j, (list, range, np.ndarray) ) \
            or not len(inds_limits_syn_i) * len(inds_limits_syn_j) \
                or not parameters:
        return syn_group

    return set_synaptic_parameters_by_neural_ind(
        syn_group  = syn_group,
        inds_syn_i = range(inds_limits_syn_i[0], inds_limits_syn_i[-1] + 1),
        inds_syn_j = range(inds_limits_syn_j[0], inds_limits_syn_j[-1] + 1),
        std_value  = std_value,
        parameters = parameters,
        extra_cond = extra_cond,
    )

def set_scaled_synaptic_parameters_by_neural_ind(
    syn_group        : b2.Synapses,
    inds_syn_i       : list[int],
    inds_syn_j       : list[int],
    pars_vals_nominal: dict[str, float],
    pars_keys        : list[str],
    pars_scalings    : list[float],
    pars_std_value   : float = 0.0,
    extra_cond       : tuple[list[str], str] = None,
) -> b2.Synapses:
    '''
    Auxiliary function to scale and assign the synaptic
    parameters to the desired synaptic group, according to the provided scalings
    '''
    # Copy and scale
    scaled_pars = _apply_scalings_to_parameters(
        parameters = pars_vals_nominal,
        pars_keys  = pars_keys,
        scalings   = pars_scalings
    )

    # Assign
    return set_synaptic_parameters_by_neural_ind(
        syn_group  = syn_group,
        inds_syn_i = inds_syn_i,
        inds_syn_j = inds_syn_j,
        std_value  = pars_std_value,
        parameters = scaled_pars,
        extra_cond = extra_cond,
    )

# \-------- [ FUNCTIONS TO ASSIGN PARAMETERS' VALUES ] --------
