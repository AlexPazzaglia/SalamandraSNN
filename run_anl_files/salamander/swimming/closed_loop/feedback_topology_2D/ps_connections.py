''' Get the proprioceptive feedback connections '''

def get_ps_to_ax_ex(amp, sigma_up, sigma_dw):
    ''' Get the proprioceptive feedback gains '''
    return {
        'name'      : 'AX_ps -> AX_all Ipsi',
        'synapse'   : 'syn_ex',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_mech',
            'amp'     : amp,
            'sigma_up': sigma_up,
            'sigma_dw': sigma_dw,
        },
        'cond_list' : [
            [ '', 'ipsi', 'ax', 'ps', 'ax', ['ex', 'in']]
        ],
        'cond_str'  : ''
    }

def get_ps_to_mn_ex(amp, sigma_up, sigma_dw):
    ''' Get the proprioceptive feedback gains '''
    return {
        'name'      : 'AX_ps -> AX_mn Ipsi',
        'synapse'   : 'syn_ex',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_mech',
            'amp'     : amp,
            'sigma_up': sigma_up,
            'sigma_dw': sigma_dw,
        },
        'cond_list' : [
            [ '', 'ipsi', 'ax', 'ps', 'ax', 'mn']
        ],
        'cond_str'  : ''
    }

def get_ps_to_ax_in(amp, sigma_up, sigma_dw):
    ''' Get the proprioceptive feedback gains '''
    return {
        'name'      : 'AX_ps -> AX_all Contra',
        'synapse'   : 'syn_in',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_mech',
            'amp'     : amp,
            'sigma_up': sigma_up,
            'sigma_dw': sigma_dw,
        },
        'cond_list' : [
            ['', 'contra', 'ax', 'ps', 'ax', ['ex', 'in']]
        ],
        'cond_str'  : ''
    }

def get_ps_to_mn_in(amp, sigma_up, sigma_dw):
    ''' Get the proprioceptive feedback gains '''
    return {
        'name'      : 'AX_ps -> AX_mn Contra',
        'synapse'   : 'syn_in',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_mech',
            'amp'     : amp,
            'sigma_up': sigma_up,
            'sigma_dw': sigma_dw,
        },
        'cond_list' : [
            ['', 'contra', 'ax', 'ps', 'ax', 'mn']
        ],
        'cond_str'  : ''
    }
