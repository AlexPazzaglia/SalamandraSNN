'''
Module to store the functions used to define and manipulate neuronal and synaptic behavior.
'''
import re

from network_modules.equations import synaptic_models

# Internal parameters common to all models, used to identify neuronal populations
NEURAL_IDENTIFIERS = '''
y_neur     :  metre (constant)    # Longitudinal position along the axis (neural coordinates)
y_mech     :  metre (constant)    # Longitudinal position along the axis (mechanical coordinates)
side_id    :      1 (constant)    # Left/Right + Axial/Limb DOF
ner_id     :      1 (constant)    # Neuron type (e.g. ex, in, mn, rs)
ner_sub_id :      1 (constant)    # Neuron sub-type (e.g. V2a, V0d)
limb_id    :      1 (constant)    # Limb index (0 for axis)
pool_id    :      1 (constant)    # Pool index
t_refr     : second (constant)    # Refractory period
'''

# -------- [ FUNCTIONS TO DEFINE NEURONAL MODELS ] --------
def _define_noise_term(unit= 'volt', tau_name='tau_memb') -> str:
    ''' Define noise term for the neuronal model '''
    noise_dv    = f'sigma * sqrt( 2 / {tau_name} ) * xi'
    noise_sigma = f'sigma : {unit} (constant)'

    return noise_dv, noise_sigma


def _define_neuronal_model_ad_if(
    syn_labels             : list[str],
    silencing              : bool = False,
    noise_term             : bool = False,
    syn_id                 : str  = None,
    additional_params_units: dict[str, str] = None,
) -> str:
    '''
    Adaptive Integrate and Fire neuron model with two adaptation variables

    Define the equations for the neurons in the population + Substitute shared values +
    Concatenate the list of neural parameter \n
    - syn_params = synaptic parameters for the entire population, whose value can be substituted\n
    - additional_params_units = parameters whose value changes between different pools,
    defined as constant\n
    '''

    if syn_id is None:
        syn_id = '1'

    # Define synaptic currents and parameters
    (
        currents_syn,
        params_syn,
        current_syn_sum
    ) = synaptic_models.define_synaptic_currents(
        syn_labels = syn_labels,
        syn_id     = syn_id,
        syn_dim    = 'volt',
        syn_type   = 'single_exp',
    )

    # Total input current
    if silencing:
        current_tot = f'''
        I_tot = ( {current_syn_sum} + R_memb * I_ext ) * (1 - silenced_ner) : volt
        silenced_ner : 1 (constant)
        '''
    else:
        current_tot = f'''
        I_tot = ( {current_syn_sum} + R_memb * I_ext ) : volt
        '''

    # Noise term
    noise_dv, noise_sigma = _define_noise_term() if noise_term else ('', '')

    # Equations
    eqs = f'''
    dv/dt  =  {noise_dv} + ( -(v- V_rest ) - R_memb*w1 - R_memb*w2 + I_tot ) / tau_memb: volt (unless refractory)
    dw1/dt = - w1 / tau1 : ampere (unless refractory)
    dw2/dt = - w2 / tau2 : ampere (unless refractory)

    {noise_sigma}
    {current_tot}
    I_ext : amp

    # Synaptic currents
    {currents_syn}

    # Neuronal parameters
    std_val : 1 (constant)
    tau_memb: second (constant)
    R_memb  : ohm (constant)
    V_rest  : volt (constant)
    V_reset : volt (constant)
    V_thres : volt (constant)
    tau1    : second (constant)
    tau2    : second (constant)
    delta_w1: ampere (constant)
    delta_w2: ampere (constant)

    # Neuronal identifiers
    {NEURAL_IDENTIFIERS}

    # Synaptic parameters
    {params_syn}
    '''

    # Add internal variables for the other parameters
    if not additional_params_units:
        additional_params_units = {}

    if len(additional_params_units):
        eqs += '\n# Additional parameters'

    for par, unit in additional_params_units.items():
        unit = unit if unit != '' else '1'
        eqs += f"\n{par} : {unit} (constant) "

    # Eliminate leading spaces
    eqs = re.sub(r'\n\s+', '\n', eqs)

    return eqs

def _define_neuronal_model_adex_if(
    syn_labels             : list[str],
    silencing              : bool = False,
    noise_term             : bool = False,
    syn_id                 : str  = None,
    additional_params_units: dict[str, str] = None,
) -> str:
    '''
    UNIFIED (LINEAR SYNAPSES)
    Adaptive Exponential Integrate and Fire neuron model

    Define the equations for the neurons in the population + Substitute shared values +
    Concatenate the list of neural parameter \n
    - syn_params = synaptic parameters for the entire population, whose value can be substituted\n
    - additional_params_units = parameters whose value changes between different pools,
    defined as constant\n
    '''

    if syn_id is None:
        syn_id = '1'

    # Define synaptic currents and parameters
    (
        currents_syn,
        params_syn,
        current_syn_sum
    ) = synaptic_models.define_synaptic_currents(
        syn_labels = syn_labels,
        syn_id     = syn_id,
        syn_dim    = 'volt',
        syn_type   = 'single_exp',
    )

    # Total input current
    if silencing:
        current_tot = f'''
        I_tot = ( {current_syn_sum} + R_memb * I_ext ) * (1 - silenced_ner) : volt
        silenced_ner : 1 (constant)
        '''
    else:
        current_tot = f'''
        I_tot = ( {current_syn_sum} + R_memb * I_ext ) : volt
        '''

    # Noise term
    noise_dv, noise_sigma = _define_noise_term() if noise_term else ('', '')

    # Internal neuronal current
    current_ner_str =  '(V_rest - v) + exp_term * delta_t * exp( (v - V_rh) / delta_t )'

    # Equations
    eqs = f'''
    dv/dt  = {noise_dv} + ( {current_ner_str} - R_memb * w1 + I_tot ) / tau_memb  : volt (unless refractory)
    dw1/dt = ( a_gain1 * (v - V_rest) - w1 ) / tau1                  : amp  (unless refractory)

    {noise_sigma}
    {current_tot}
    I_ext : amp

    # Synaptic currents
    {currents_syn}

    # Neuronal parameters
    std_val   :      1 (constant)
    tau_memb  : second (constant)
    R_memb    :    ohm (constant)
    V_rest    :   volt (constant)
    V_reset   :   volt (constant)
    V_thres   :   volt (constant)

    V_rh     :    volt (constant)
    delta_t  :    volt (constant)
    exp_term :       1 (constant)

    a_gain1   : siemens (constant)
    tau1      : second  (constant)
    delta_w1  :    amp  (constant)

    # Neuronal identifiers
    {NEURAL_IDENTIFIERS}

    # Synaptic parameters
    {params_syn}
    '''

    # Add internal variables for the other parameters
    if not additional_params_units:
        additional_params_units = {}

    if len(additional_params_units):
        eqs += '\n# Additional parameters'

    for par, unit in additional_params_units.items():
        unit = unit if unit != '' else '1'
        eqs += f"\n{par} : {unit} (constant) "

    # Eliminate leading spaces
    eqs = re.sub(r'\n\s+', '\n', eqs)

    return eqs

def _define_neuronal_model_izhikevich(
    syn_labels             : list[str],
    silencing              : bool = False,
    noise_term             : bool = False,
    syn_id                 : str  = None,
    additional_params_units: dict[str, str] = None,
) -> str:
    '''
    UNIFIED (LINEAR SYNAPSES)
    Izhikevic neuron model

    Define the equations for the neurons in the population + Substitute shared values +
    Concatenate the list of neural parameter \n
    - syn_params = synaptic parameters for the entire population, whose value can be substituted\n
    - additional_params_units = parameters whose value changes between different pools,
    defined as constant\n
    '''

    if syn_id is None:
        syn_id = '1'

    # Define synaptic currents and parameters
    (
        currents_syn,
        params_syn,
        current_syn_sum
    ) = synaptic_models.define_synaptic_currents(
        syn_labels = syn_labels,
        syn_id     = syn_id,
        syn_dim    = 'amp',
        syn_type   = 'double_exp',
    )

    # Total input current
    if silencing:
        current_tot = f'''
        I_tot = ( {current_syn_sum} + I_ext ) * (1 - silenced_ner) : amp
        silenced_ner : 1 (constant)
        '''
    else:
        current_tot = f'''
        I_tot = ( {current_syn_sum} + I_ext ) : amp
        '''

    # Noise term
    noise_dv, noise_sigma = _define_noise_term() if noise_term else ('', '')

    # Internal neuronal current
    current_ner_str =  'k_gain * (v - V_rest) * (v - V_crit)'

    # Equations
    eqs = f'''
    dv/dt  = {noise_dv} + ( {current_ner_str} - w1 + I_tot ) / C_memb  : volt (unless refractory)
    dw1/dt = (b_gain * (v - V_rest) - w1) / tau1          : amp  (unless refractory)

    {noise_sigma}
    {current_tot}
    I_ext : amp

    # Synaptic currents
    {currents_syn}

    # Neuronal parameters
    C_memb    : farad              (constant)
    k_gain    : ampere / volt**2   (constant)
    b_gain    : 1 / ohm            (constant)
    tau1      : second             (constant)
    delta_w1  : ampere             (constant)
    V_rest    : volt               (constant)
    V_crit    : volt               (constant)
    V_thres   : volt               (constant)
    V_reset   : volt               (constant)

    # Neuronal identifiers
    {NEURAL_IDENTIFIERS}

    # Synaptic parameters
    {params_syn}
    '''

    # Add internal variables for the other parameters
    if not additional_params_units:
        additional_params_units = {}

    if len(additional_params_units):
        eqs += '\n# Additional parameters'

    for par, unit in additional_params_units.items():
        unit = unit if unit != '' else '1'
        eqs += f"\n{par} : {unit} (constant) "

    # Eliminate leading spaces
    eqs = re.sub(r'\n\s+', '\n', eqs)

    return eqs

def _define_neuronal_model_muscle(
    syn_labels             : list[str],
    silencing              : bool = False,
    noise_term             : bool = False,
    syn_id                 : str  = None,
    additional_params_units: dict[str, str] = None,
) -> str:
    '''
    UNIFIED (LINEAR SYNAPSES)
    Define muscle activation model + Substitute with desired values.\n
    - commonparams_units = parameters shared among the entire population,
    whose value can be substituted\n
    - params = additional parameters whose value can be substituted\n
    '''

    if syn_id is None:
        syn_id = '_mc'

    # Define synaptic currents and parameters
    (
        currents_syn,
        params_syn,
        current_syn_sum
    ) = synaptic_models.define_synaptic_currents(
        syn_labels = syn_labels,
        syn_id     = syn_id,
        syn_dim    = '1',
        syn_type   = 'single_exp',
        v_dim      = '1',
    )

    # Total input current
    if silencing:
        current_tot = f'''
        I_tot = clip( ({current_syn_sum}) * (1 - silenced_ner), 0.05, 100 ) : 1
        silenced_ner : 1 (constant)
        '''
    else:
        current_tot = f'''
        I_tot = clip( ({current_syn_sum}), 0.05, 100 ) : 1
        '''

    # Noise term
    noise_term  = _define_noise_term('1', 'tau_mc_act') if noise_term else ('', '')
    noise_dv    = noise_term[0]
    noise_sigma = noise_term[1]

    # Equations
    eqs = f'''
    dv/dt = {noise_dv} + ( I_tot -v ) / tau_muscle : 1
    tau_muscle = tau_mc_act * (0.5 + 1.5 * v) * int( I_tot > v ) + tau_mc_deact / (0.5 + 1.5 * v) * int( I_tot <= v ) : second

    {noise_sigma}
    {current_tot}

    # Synaptic currents
    {currents_syn}

    # Neuronal parameters
    tau_mc_act   : second (constant)
    tau_mc_deact : second (constant)

    # Neuronal identifiers
    {NEURAL_IDENTIFIERS}

    # Synaptic parameters
    {params_syn}
    '''

    # Add internal variables for the other parameters
    if not additional_params_units:
        additional_params_units = {}

    for par, unit in additional_params_units.items():
        unit = unit if unit != '' else '1'
        eqs += f"\n{par} : {unit} (constant) "

    # Eliminate leading spaces
    eqs = re.sub(r'\n\s+', '\n', eqs)

    return eqs

def define_neuronal_model(
    neuronal_model_type    : str,
    syn_labels             : list[str],
    silencing              : bool = False,
    noise_term             : bool = False,
    syn_id                 : str  = None,
    additional_params_units: dict[str, str] = None,
) -> str:
    ''' Define equations for the selected neuronal model, if available '''

    available_neuronal_models = {
        'ad_if'     : _define_neuronal_model_ad_if,
        'adex_if'   : _define_neuronal_model_adex_if,
        'izhikevich': _define_neuronal_model_izhikevich,
        'muscle'    : _define_neuronal_model_muscle,
    }

    return available_neuronal_models[neuronal_model_type](
        syn_labels               = syn_labels,
        silencing               = silencing,
        noise_term              = noise_term,
        syn_id                  = syn_id,
        additional_params_units = additional_params_units,
    )

# \-------- [ FUNCTIONS TO DEFINE NEURONAL MODELS] --------

# -------- [ FUNCTIONS TO DEFINE MODEL'S INITIAL CONDITIONS ] --------
def _define_model_initial_values_ad_if(
        synlabels: list[str],
        syn_id   : str = None,
        rest_vals: bool = False,
    ) -> dict[str, str]:
    ''' Initialization values of the variables '''

    if syn_id is None:
        syn_id = '1'
    initial_values = {}

    for sid in synlabels:
        initial_values[f'g_{sid}{syn_id}_tot'] = '0'

    initial_values['not_refractory']  = 'True'
    initial_values['v']  = 'V_rest + rand() * (V_thres - V_rest)' if not rest_vals else 'V_rest'
    initial_values['w1'] = '0 * pamp + 2 * rand() * delta_w1'     if not rest_vals else '0 * pamp'
    initial_values['w2'] = '0 * pamp + 2 * rand() * delta_w2'     if not rest_vals else '0 * pamp'

    return initial_values

def _define_model_initial_values_adex_if(
        synlabels: list[str],
        syn_id   : str = None,
        rest_vals: bool = False,
    ) -> dict[str, str]:
    ''' Initialization values of the variables '''

    if syn_id is None:
        syn_id = '1'

    initial_values = {}

    for sid in synlabels:
        initial_values[f'g_{sid}{syn_id}_tot'] = '0'

    initial_values['not_refractory']  = 'True'
    initial_values['v']  = 'V_rest + rand() * (V_thres - V_rest)' if not rest_vals else 'V_rest'
    initial_values['w1'] = '0 * pamp + 2 * rand() * delta_w1'     if not rest_vals else '0 * pamp'

    return initial_values

def _define_model_initial_values_izhikevich(
    synlabels: list[str],
    syn_id   : str = None,
    rest_vals: bool = False,
) -> dict[str, str]:
    ''' Initialization values of the variables '''

    if syn_id is None:
        syn_id = '1'

    initial_values = {}

    for sid in synlabels:
        initial_values[f'g_{sid}{syn_id}_tot']   = '0'
        initial_values[f'g_{sid}{syn_id}_biexp'] = '0'

    initial_values['not_refractory']  = 'True'
    initial_values['v']  = 'V_rest + rand() * (V_thres - V_rest)' if not rest_vals else 'V_rest'
    initial_values['w1'] = '0 * pamp + 2 * rand() * delta_w1'     if not rest_vals else '0 * pamp'

    return initial_values

def _define_model_initial_values_muscle(
        synlabels: list[str],
        syn_id   : str = None,
        rest_vals: bool = False,
    ) -> dict[str, str]:
    ''' Initialization values of the variables '''

    if syn_id is None:
        syn_id = '_mc'

    initial_values = {}

    for sid in synlabels:
        initial_values[f'g_{sid}{syn_id}_tot'] = '0'

    initial_values['v']  = '0.05 + 0.95 * rand()' if not rest_vals else '0.05'

    return initial_values

def define_model_initial_values(
        neuronal_model_type: str,
        synlabels          : list[str],
        syn_id             : str = None,
        rest_vals          : bool = False,
    ) -> dict[str, str]:
    ''' Defini initial values for the selected neuronal type, if available '''

    available_neuronal_models = {
        'ad_if'     : _define_model_initial_values_ad_if,
        'adex_if'   : _define_model_initial_values_adex_if,
        'izhikevich': _define_model_initial_values_izhikevich,
        'muscle'    : _define_model_initial_values_muscle,
    }

    return available_neuronal_models[neuronal_model_type](
        synlabels = synlabels,
        syn_id    = syn_id,
        rest_vals = rest_vals,
    )

# \------- [ FUNCTIONS TO DEFINE MODEL'S INITIAL CONDITIONS ] --------

# -------- [ FUNCTIONS TO DEFINE MODEL'S IMPULSIVE BEHAVIOR ] --------
def define_reset_condition(
    n_adaptation_variables: int,
    extracond             : str = ''
) -> str:
    '''
    Define reset condition of the neuronal model
    '''

    adaptation_increments = '\n'.join(
        [f'w{i+1} = w{i+1} + delta_w{i+1}' for i in range(n_adaptation_variables) ]
    )

    reset = f'''
    v = V_reset
    {adaptation_increments}
    {extracond}
    '''

    # Eliminate leading spaces
    reset = re.sub(r'\n\s+', '\n', reset)

    return reset

def define_threshold_condition(extracond: str = '') -> str:
    '''
    Define threshold condition of the neuronal model (one line string)
    '''
    return f'v >= V_thres {extracond}'

def define_refractoriness(extracond: str = '') -> str:
    '''
    Define refractoriness condition of the neuronal model (one line string)
    '''
    return f't_refr {extracond}'

# \-------- [ FUNCTIONS TO DEFINE MODEL'S IMPULSIVE BEHAVIOR ] --------
