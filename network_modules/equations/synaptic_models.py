'''
Module to store the functions used to define and manipulate neuronal and synaptic behavior.
'''

# -------- [ FUNCTIONS TO DEFINE SYNAPTIC CURRENTS ] --------

def _define_synaptic_current_single_exponential(
    syn_id   : str,
    syn_label: str,
    syn_dim  : str,
    v_dim    : str = 'volt',
) -> tuple[str, str, str]:
    '''
    Single exponential synaptic current
    '''

    # Current name
    syn_current_name = f'I_{syn_label}'

    # Current and conductance equations
    s_lab = syn_label
    g_lab = syn_label + syn_id

    syn_current_eq = f'''
    I_{s_lab}         = w_{s_lab} * g_{g_lab}_tot * (E_{s_lab} - v)  : {syn_dim}
    dg_{g_lab}_tot/dt = - g_{g_lab}_tot / tau_{s_lab}                : 1
    '''

    # Internal parameters
    w_dim = f'{syn_dim} / {v_dim}' if syn_dim != v_dim else '1'

    syn_params = f'''
    w_{s_lab}   : {w_dim}  (constant)
    E_{s_lab}   : {v_dim}  (constant)
    tau_{s_lab} : second   (constant)
    '''

    return syn_current_name, syn_current_eq, syn_params

def _define_synaptic_current_double_exponential(
    syn_id   : str,
    syn_label: str,
    syn_dim  : str,
    v_dim    : str = 'volt',
) -> tuple[str, str, str]:
    '''
    Double exponential synaptic current
    '''

    # Current name
    syn_current_name = f'I_{syn_label}'

    # Current and conductance equations
    # NOTE:
    # g_biexp evolves as g(t) = exp(-t/tau_1) - exp(-t/tau_2)
    # g_biexp reaches maximum at t = log(tau_2/tau_1) * tau_1 * tau_2 / (tau_2 - tau_1)
    # on_pre = 'g_biexp += w_syn'

    s_lab = syn_label
    g_lab = syn_label + syn_id

    t_lab        = f'tau_{s_lab}'
    g_biexp_gain = f'({t_lab}_d  - {t_lab}_r) / ({t_lab}_r * {t_lab}_d)'
    g_tot_gain   = f'({t_lab}_d / {t_lab}_r) ** ({t_lab}_r / ({t_lab}_d - {t_lab}_r)'

    syn_current_eq = f'''
    I_{s_lab}           = w_{s_lab} * g_{g_lab}_biexp * (E_{s_lab} - v)  : {syn_dim}

    dg_{g_lab}_biexp/dt = ( {g_biexp_gain} ) * ( {g_tot_gain} ) * g_{g_lab}_tot - g_{g_lab}_biexp ) : 1
    dg_{g_lab}_tot/dt   = - g_{g_lab}_tot / {t_lab}_d                                               : 1

    '''

    # Internal parameters
    w_dim = f'{syn_dim} / {v_dim}' if syn_dim != v_dim else '1'

    syn_params = f'''
    w_{s_lab}      : {w_dim}    (constant)
    E_{s_lab}      : {v_dim}    (constant)
    tau_{s_lab}_r  : second     (constant)
    tau_{s_lab}_d  : second     (constant)
    '''

    return syn_current_name, syn_current_eq, syn_params

def define_synaptic_currents(
        syn_labels: list[str],
        syn_id    : str,
        syn_dim   : str,
        syn_type  : str,
        v_dim     : str = 'volt',
):
    ''' Define the synaptic currents for the model '''

    # Define the synaptic current model
    available_synaptic_currents_models = {
        'single_exp': _define_synaptic_current_single_exponential,
        'double_exp': _define_synaptic_current_double_exponential,
    }
    current_model = available_synaptic_currents_models[syn_type]

    # Define the synaptic currents
    currents_syn    = ''
    params_syn      = ''
    current_syn_sum = ''

    for syn_label in syn_labels:
        (
            syn_current_name,
            syn_current_eq,
            syn_params
        ) = current_model(
            syn_id    = syn_id,
            syn_label = syn_label,
            syn_dim   = syn_dim,
            v_dim     = v_dim,
        )

        current_syn_sum += f'+ {syn_current_name} '
        currents_syn    += syn_current_eq
        params_syn      += syn_params

    return currents_syn, params_syn, current_syn_sum

# \------- [ FUNCTIONS TO DEFINE SYNAPTIC CURRENTS ] --------

# -------- [ FUNCTIONS TO DEFINE SYNAPTIC EQUATIONS ] --------

def define_syn_equation(
        syn_labs               : list[str],
        silencing              : bool = False,
        weighted               : bool = False,
        additional_params_units: dict[str, str]= None
    ) -> str:
    '''
    UNIFIED (LINEAR)
    Synaptic equation with weighting or silencing option.
    Valid for excitatory and inhibitory
    '''

    syn_eq = 'link = 1 : 1'

    if silencing:
        syn_eq += '\nsilenced_syn : 1'

    if weighted:
        for sid in syn_labs:
            syn_eq += f'\nweight_{sid} : 1'

    # Add internal variables for additional parameters
    if not additional_params_units:
        additional_params_units = {}

    for par, unit in additional_params_units.items():
        unit = unit if unit != '' else '1'
        syn_eq += f"\n{par} : {unit} (constant) "

    return syn_eq

def syn_on_pre(
        syn_id: str,
        syn_labs: list[str],
        silencing: bool = False,
        weighted: bool = False,
        extracond: str = ''
    ) -> str:
    '''
    UNIFIED (LINEAR)
    Synaptic reset after input spike (valid for excitatory and inhibitory)
    '''

    syn_eq = ''
    for sid in syn_labs:
        syn_eq += f'\ng_{sid}{syn_id}_tot_post += int(not_refractory_post)'

        if weighted:
            syn_eq += f'* weight_{sid}'

        if silencing:
            syn_eq += '* (1 - silenced_syn) * (1 - silenced_ner_post)'

    syn_eq += extracond
    return syn_eq

# \-------- [ FUNCTIONS TO DEFINE SYNAPTIC EQUATIONS ] --------
