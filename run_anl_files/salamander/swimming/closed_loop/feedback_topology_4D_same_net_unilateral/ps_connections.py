''' Get the proprioceptive feedback connections '''

def get_ps_to_ax_ex(amp, max_ps_range):
    ''' Get the proprioceptive feedback connections '''
    return {
        'name'      : 'AX_ps -> AX_all Ipsi',
        'synapse'   : 'syn_ex',
        'type'      : 'connect_identity',
        'parameters': {
            'amp'     : amp,
        },
        'cond_list' : [
            [ '', 'ipsi', 'ax', 'ps', 'ax', ['ex', 'in']]
        ],
        'cond_str'  : f'( abs( y_mech_post - y_mech_pre ) < {max_ps_range} * metre)'
    }

def get_ps_to_mn_ex(amp, max_ps_range):
    ''' Get the proprioceptive feedback connections '''
    return {
        'name'      : 'AX_ps -> AX_mn Ipsi',
        'synapse'   : 'syn_ex',
        'type'      : 'connect_identity',
        'parameters': {
            'amp'     : amp,
        },
        'cond_list' : [
            [ '', 'ipsi', 'ax', 'ps', 'ax', 'mn']
        ],
        'cond_str'  : f'( abs( y_mech_post - y_mech_pre ) < {max_ps_range} * metre)'
    }

def get_ps_to_ax_in(amp, max_ps_range):
    ''' Get the proprioceptive feedback connections '''
    return {
        'name'      : 'AX_ps -> AX_all Contra',
        'synapse'   : 'syn_in',
        'type'      : 'connect_identity',
        'parameters': {
            'amp'     : amp,
        },
        'cond_list' : [
            ['', 'contra', 'ax', 'ps', 'ax', ['ex', 'in']]
        ],
        'cond_str'  : f'( abs( y_mech_post - y_mech_pre ) < {max_ps_range} * metre)'
    }

def get_ps_to_mn_in(amp, max_ps_range):
    ''' Get the proprioceptive feedback connections '''
    return {
        'name'      : 'AX_ps -> AX_mn Contra',
        'synapse'   : 'syn_in',
        'type'      : 'connect_identity',
        'parameters': {
            'amp'     : amp,
        },
        'cond_list' : [
            ['', 'contra', 'ax', 'ps', 'ax', 'mn']
        ],
        'cond_str'  : f'( abs( y_mech_post - y_mech_pre ) < {max_ps_range} * metre)'
    }
