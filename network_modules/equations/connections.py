'''
Module to store the functions used to define and manipulate neuronal and synaptic behavior.
'''
from numpy import isnan, tanh
from functools import partial

import brian2 as b2

# -------- [ FUNCTIONS TO DEFINE CONNECTIVITY CONDITIONS ] --------

# BY POOLS INDICES

def concond_ilimit_jlimit(
    imin           : int,
    imax           : int,
    jmin           : int,
    jmax           : int,
    extraconditions: str = ''
) -> str:
    '''
    Condition for the desired range in i and j
    '''

    output = (
        f"( (i >= {imin}) and (i <= {imax}) and "\
        + f"(j >= {jmin}) and (j <= {jmax}) )"
    )

    if extraconditions != '':
        output = extraconditions + ' and ' + output

    return '(i!=j) and ' + output

def concond_ilimit_jpools(
    imin           : int,
    imax           : int,
    j_pools        : list,
    extraconditions: str = ''
) -> str:
    '''
    Condition for the desired range in i and multiple ranges in j\n
    Ex: imin, imax, ex_in_l[s:]
    '''

    condition_i = f"( i >= {imin} and i <= {imax} ) "

    condition_j = ' or '.join(
        [
            f"( (j >= {j_ind[0]}) and (j <= {j_ind[-1]}) )"
            for j_ind in j_pools
        ]
    )

    output = f'{condition_i} and ( {condition_j} )'

    if extraconditions != '':
        output = extraconditions + ' and ' + output

    return '(i!=j) and ' + output

def concond_ipools_jpools_directional(
    ipools         : list,
    jpools         : list,
    direction      : str,
    extraconditions: str = ''
) -> str:
    '''
    Condition for multiple ranges in i and j,
    with a desired direction (ascending, descending) \n
    Ex: lb_in_l, ex_in_r, 'd'
    '''

    if direction == 'u':
        direction = '>'      # i > j --> Upward-ordiented connection
    elif direction == 'd':
        direction = '<'      # i < j --> Downward-ordiented connection
    else:
        raise ValueError('Direction must be either \'u\' or \'d\' ')

    condition_i = ' or '.join(
        [
            f"( i >= {ipool[0]} and i <= {ipool[-1]} )"
            for ipool in ipools
        ]
    )

    condition_j = ' or '.join(
        [
            f"( j >= {jpool[0]} and j <= {jpool[-1]} )"
            for jpool in jpools
        ]
    )

    output = f"( y_neur_pre {direction} y_neur_post ) and ( {condition_i} ) and ( {condition_j} )"

    if extraconditions != '':
        output = extraconditions + ' and ' + output

    return '(i!=j) and ' + output

def concond_byidentity(
    condlist       : list,
    extraconditions: str = ''
) -> str:
    '''
    Condition based on direction, crossing, location and synaptic type\n
    - extraconditions (if included) contains additional
    constrains to be linked by and operators\n
    Ex: [ [ 'up', 'ipsi', 'ax','ex', 'ax', ['ex','in'] ] ]\n
    Note: elements of the list can also be lists themselves
    (will be linked by and operators)\n
    Note: equality considered with tolerance to avoid errors
    due to numerical approximations
    '''

    def exp_from_dict(cond_elem, dic: dict) -> str:

        if isinstance(cond_elem, list):
            exp = '( '

            for i, cond in enumerate(cond_elem):
                if i == 0:
                    exp += dic[cond]
                else:
                    exp += ' or ' + dic[cond]

            exp += ' )'

        else:
            exp = dic[cond_elem]

        return exp

    # DIRECTION ( Safe with sh > 0.01 mm )
    dir_dict = {
        'up': '(y_neur_pre > y_neur_post + 0.00001 * metre)', # greater
        'dw': '(y_neur_pre < y_neur_post - 0.00001 * metre)', # smaller
        'eq'  : '( abs( y_neur_pre - y_neur_post ) < 0.00001 * metre  )',
        'upeq': '(y_neur_pre > y_neur_post - 0.00001 * metre)',           # Greater or eq
        'dweq': '(y_neur_pre < y_neur_post + 0.00001 * metre)',           # Smaller or eq
    }

    # RELATIVE POSITION

    # Axis
    contype_dict_axis = {
        'ipsi'    : '( side_id_pre * side_id_post > 0 )', # Generic contralateral
        'contra'  : '( side_id_pre * side_id_post < 0 )', # Generic ipsilateral
        'same'    : '( side_id_pre == + side_id_post )',  # Same position
        'opposite': '( side_id_pre == - side_id_post )',  # Opposite (symmetric) position
    }

    # Limbs
    same_side_cond = '( abs(side_id_pre)%2 == abs(side_id_post)%2 )'
    diff_side_cond = '( abs(side_id_pre)%2 != abs(side_id_post)%2 )'

    same_dof_cond = '( abs(side_id_pre)//2 == abs(side_id_post)//2 )'
    diff_dof_cond = '( abs(side_id_pre)//2 != abs(side_id_post)//2 )'

    limb_conn_type = {
        'ipsi'        : f'{same_side_cond}',
        'contra'      : f'{diff_side_cond}',
        'ipsi_noneq'  : f'{same_side_cond} and {diff_dof_cond}',
        'contra_noneq': f'{diff_side_cond} and {diff_dof_cond}',
        'agonist'     : f'( side_id_pre == side_id_post ) and {same_dof_cond}',
        'antagonist'  : f'( side_id_pre != side_id_post ) and {same_dof_cond}',
    }

    limb_cond = 'limb_id_pre != 0 and limb_id_post != 0'
    limb_conn_cond = {
        'any_limb'      : f'( {limb_cond} )',
        'same_limb'     : f'( {limb_cond} and limb_id_pre == limb_id_post )',
        'different_limb': f'( {limb_cond} and limb_id_pre != limb_id_post )',
    }

    contype_dict_limbs = {
        f'{lb_cond_key}.{lb_type_key}' : f'( {lb_cond_val} and {lb_type_val} )'
        for (lb_cond_key, lb_cond_val) in limb_conn_cond.items()
        for (lb_type_key, lb_type_val) in limb_conn_type.items()
    }

    # Together
    contype_dict = contype_dict_axis | contype_dict_limbs

    # POOL IDENTITY
    axial_dict = {
        'ax'    : '( abs(side_id) == 1 )',
        'lb'    : '( abs(side_id) >= 2 )',
        'lbf'   : '( abs(side_id)%2 == 0)',
        'lbe'   : '( abs(side_id)%2 == 1 )',
        'lb_0'  : '( abs(side_id)//2 == 1 )',
        'lbf_0' : '( abs(side_id) == 2 )',
        'lbe_0' : '( abs(side_id) == 3 )',
    }

    # NEURONAL TYPE
    ner_types = {
        'ex': '(ner_id == 0)',  # excitatory
        'in': '(ner_id == 1)',  # inhibitory
        'mn': '(ner_id == 2)',  # motor neuron
        'rs': '(ner_id == 3)',  # reticulospinal
        'ps': '(ner_id == 4)',  # propriosensory neuron
        'es': '(ner_id == 5)',  # exterosensory neuron
        'mc': '(ner_id == 6)',  # muscle cell
    }

    ner_sub_types = {
        'ex': {
            'V2a': '( ner_sub_id == 0 )',
            'V0v': '( ner_sub_id == 1 )',
        },
        'in': {
            'V1' : '( ner_sub_id == 0 )',
            'V0d': '( ner_sub_id == 1 )',
            'dI6': '( ner_sub_id == 2 )',
        }
    }

    type_dict = (
        ner_types |
        {
            f'{ner_id}.{ner_sub_id}' : f'( {cond_id} and {cond_sub_id} )'
            for (ner_id, cond_id) in ner_types.items()
            for (ner_sub_id, cond_sub_id) in ner_sub_types.get(ner_id, {}).items()
        }
    )

    output = '( '
    for i_x, cond in enumerate(condlist):

        # Substitute conditions with corresponding expression (AND)
        xycondlist = []

        if cond[0] != '':
            xycondlist.append(exp_from_dict(cond[0], dir_dict))

        if cond[1] != '':
            xycondlist.append(exp_from_dict(cond[1], contype_dict))

        xycond_exp = ''
        for i_c, xy_cond in enumerate(xycondlist):
            if i_c == 0:
                xycond_exp += xy_cond
            else:
                xycond_exp += ' and ' + xy_cond

        if xycond_exp != '':
            xycond_exp = ' ( ' + xycond_exp + ' ) '

        # Substitute conditions with corresponding expression (AND)
        xcondlist = []
        if cond[2] != '':
            xcondlist.append(
                exp_from_dict(cond[2], axial_dict).replace('side_id', 'side_id_pre'))

        if cond[3] != '':
            xcondlist.append(
                exp_from_dict(
                    cond[3],
                    type_dict
                ).replace('ner_id', 'ner_id_pre').replace('ner_sub_id', 'ner_sub_id_pre')
            )

        xcond_exp = ''
        for i_c, x_cond in enumerate(xcondlist):
            if i_c == 0:
                xcond_exp += x_cond
            else:
                xcond_exp += ' and ' + x_cond

        if xcond_exp != '':
            xcond_exp = ' ( ' + xcond_exp + ' ) '

        # Substitute conditions with corresponding expression (AND)
        ycondlist = []

        if cond[4] != '':
            ycondlist.append(
                exp_from_dict(cond[4], axial_dict).replace('side_id', 'side_id_post'))

        if cond[5] != '':
            ycondlist.append(
                exp_from_dict(
                    cond[5],
                    type_dict
                ).replace('ner_id', 'ner_id_post').replace('ner_sub_id', 'ner_sub_id_post')
            )

        ycond_exp = ''
        for i_c, y_cond in enumerate(ycondlist):
            if i_c == 0:
                ycond_exp += y_cond
            else:
                ycond_exp += ' and ' + y_cond

        if ycond_exp != '':
            ycond_exp = ' ( ' + ycond_exp + ' ) '

        # Insert conditions in the output string (OR)
        cond = ''
        if xycond_exp != '':
            cond += '( ' + xycond_exp

        if xcond_exp != '' and cond == '':
            cond += '( ' + xcond_exp
        elif xcond_exp != '' and cond != '':
            cond += ' and ' + xcond_exp

        if ycond_exp != '' and cond == '':
            cond += '( ' + ycond_exp
        elif ycond_exp != '' and cond != '':
            cond += ' and ' + ycond_exp
        cond += ' )'

        if i_x == 0:
            output += cond
        else:
            output += ' or ' + cond
    output += ')'

    # Check if additional conditions were selected
    if extraconditions:
        output = extraconditions + ' and ' + output

    return '(i!=j) and ' + output

# BY PAIRS OF INDICES OF POSITIONS
def concond_pairs_limits(
    pre_limits     : dict,
    post_limits    : dict,
    seg_height     : float,
    extraconditions: str = ''
) -> str:
    '''
    Condition for multiple ranges for the indices of for the coordinates
    pre_limits and post_limits are dictionaries with two fields:\n
    - type specifies whethere the limits are for the indices (i) or for the coordinates (y)\n
    - limits list the pools' limits for the connections\n
    '''

    tol = 0.001
    tol_y = tol * float(seg_height)

    if pre_limits['type'] not in ['i', 'y']:
        raise ValueError('limit type should be in [i ,y]')
    elif pre_limits['type'] == 'i':
        pre_cond = "( i >= {min} and i <= {max} )"
    elif pre_limits['type'] == 'y':
        pre_cond = "( y_neur_pre >= ({min}-{tol_y}) * metre and y_neur_pre <= ({max}+{tol_y}) * metre )"

    if post_limits['type'] not in ['i', 'y']:
        raise ValueError('limit type should be in [i ,y]')
    elif post_limits['type'] == 'i':
        post_cond = "( j >= {min} and j <= {max} )"
    elif post_limits['type'] == 'y':
        post_cond = "( y_neur_post >= ({min}-{tol_y}) * metre and y_neur_post <= ({max}+{tol_y}) * metre )"

    condition = ' or '.join(
        [
            (
                '( '
                + pre_cond.format( min= pre_lim[0],  max= pre_lim[-1],  tol_y= tol_y)
                + ' and '
                + post_cond.format(min= post_lim[0], max= post_lim[-1], tol_y= tol_y)
                + ' )'
            )
            for pre_lim, post_lim in zip(pre_limits['limits'], post_limits['limits'])
        ]
    )

    if extraconditions != '':
        condition = extraconditions + ' and ' + condition

    return '(i!=j) and ' + condition

# BY RECIPROCAL POSITION

def concond_to_nth_neightbour(
    n_distance    : int,
    segment_height: float
) -> str:
    '''
    Condition to connect to n segments away (positive = descending)
    Note: equality considered with tolerance
    to avoid errors due to numerical approximations
    '''

    condition = (
        '( abs(y_neur_post - (y_neur_pre + {n} * {sh}*metre) ) < {tol} * {sh}*metre )'
    ).format(
        n   = n_distance,
        sh  = segment_height,
        tol = 0.01
    )

    return condition

# BY GIRDLE POSITION

def concond_n_distant_girdle(
    limb_pair_positions: list,
    microsegment_height: float,
    direction          : int,
    microsegments      : int = 1
) -> str:
    '''
    Condition to connect limbs at same or different girdles,
    in ascending (direction < 0) or descending (direction > 0) direction.
    Generalized for limbs constituted of multiple rows (ex: Bicanski Chapter 7)
    Note: equality considered with tolerance
    to avoid errors due to numerical approximations
    '''

    cond  = ''
    msh   = microsegment_height
    tol   = 0.01
    lb_pairs = len(limb_pair_positions)

    if direction == 0:
        lb_pair_strt = 0
        lb_pair_stop = lb_pairs
    if direction > 0:
        lb_pair_strt = 0
        lb_pair_stop = lb_pairs - direction
    elif direction < 0:
        lb_pair_strt = -direction
        lb_pair_stop = lb_pairs

    for i, limb_pos in enumerate(limb_pair_positions[lb_pair_strt: lb_pair_stop]):
        l1_start = msh * microsegments * limb_pos
        l1_stop = l1_start + msh * (microsegments - 1)

        l2_start = msh * microsegments * limb_pair_positions[i + lb_pair_strt + direction]
        l2_stop = l2_start + msh * (microsegments - 1)

        newcond = (
                f'''(
                         (
                             (y_neur_pre > ({l1_start} - {tol}* {msh}) * metre) and
                             (y_neur_pre < ({l1_stop}  + {tol}* {msh}) * metre)
                         ) and
                         (
                             (y_neur_post > ({l2_start} - {tol}* {msh}) * metre) and
                             (y_neur_post < ({l2_stop}  + {tol}* {msh}) * metre)
                         )
                     ) '''
            ).replace('\n', '').replace('    ', '')

        newcond = newcond if i == 0 else ' or ' + newcond
        cond += newcond

    return ' ( ' + cond + ' ) ' if bool(cond) else ''

def concond_equal_to_n_distant_girdle(
    direction        : int,
    exclude_recurrent: bool = False
) -> str:
    '''
    Condition to connect limbs at same or different girdles,
    in ascending (direction < 0) or descending (direction > 0) direction.
    '''

    cond_recurrent = 'and ( limb_id_pre != limb_id_post ) ' if exclude_recurrent else ''
    cond = (
        f'''(
                 ( limb_id_pre != 0 and limb_id_post != 0 ) and
                 ( (limb_id_post-1) // 2 - (limb_id_pre-1) // 2 == {direction})
                 {cond_recurrent}
            )'''
    ).replace('\n', '').replace('    ', '')

    return cond

def concond_less_than_n_distant_girdle(
    range            : int,
    exclude_recurrent: bool = False
) -> str:
    '''
    Condition to connect limbs distant less than range different girdles,
    '''

    cond_recurrent = 'and ( limb_id_pre != limb_id_post ) ' if exclude_recurrent else ''
    cond = (
        f'''(
                 ( limb_id_pre != 0 and limb_id_post != 0 ) and
                 ( abs( (limb_id_post-1) // 2 - (limb_id_pre-1) // 2 ) < {range} )
                 {cond_recurrent}
            )'''
    ).replace('\n', '').replace('    ', '')

    return cond

# \-------- [ FUNCTIONS TO DEFINE CONNECTIVITY EXPRESSIONS ] --------

# -------- [ FUNCTIONS TO CONNECT SYNAPSES ] --------
def connect(
    syn            : b2.Synapses,
    pools_list     : list,
    prob           : float,
    extraconditions: str = ''
) -> None:
    '''
    Connect synapses in the specified pools with a given probability
    '''

    if prob == 0:
        return

    cond = '( '
    for k, (i_ind, j_ind) in enumerate(pools_list):
        aux = f"( i >= {i_ind[0]} and i <= {i_ind[-1]} and j >= {j_ind[0]} and j<= {j_ind[-1]} )"
        if k == 0:
            cond += aux
        else:
            cond += ' or ' + aux
    cond += ' )'

    cond += ' and ' + extraconditions
    syn.connect(condition=cond, p=prob, skip_if_invalid=True)

    return

def connect_byidentity(
    syn            : b2.Synapses,
    condlist       : list,
    prob           : float,
    extraconditions: str = ''
) -> None:
    '''
    Connect according to a set of conditions regarding the identity of the neurons,
    considering the whole network.\n
    Fixed probability used.\n\n
    Ex: conditions = [ [ 'eq', 'ipsi', 'ax', 'ex', 'ax', 'ex' ],
    [ 'eq', 'ipsi', 'ax', 'ex', 'ax', 'in' ] ]\n
        connect_byidentity(self.S_E, conditions, 0.20)
    '''

    if prob == 0:
        return

    cond = concond_byidentity(condlist, extraconditions)
    syn.connect(condition=cond, p=prob, skip_if_invalid=True)
    return

def gaussian_connect(
    syn            : b2.Synapses,
    pools_list     : list,
    sigma          : float,
    amp            : float,
    extraconditions: str = ''
) -> None:
    '''
    Connect synapses in the specified pool pairs, in a given direction,
    according to a gaussian distribution.\n
    Ex: gaussian_connect(self.S_E, [ [el, EE_up_pools_l], [er, EE_up_pools_r] ],
     EE_sigma_up, EE_amp)
    '''

    if amp == 0 or float(sigma) == 0:
        return

    sigma = float(sigma)
    prob = (
        'clip( {amp} * exp( -( (y_neur_pre-y_neur_post)*(y_neur_pre-y_neur_post) '
        '/ (2*{sigma}*{sigma}*metre*metre) ) ),  0, 1 )'
    ).format(
        amp   = amp,
        sigma = sigma
    )

    for i_ind, j_pool in pools_list:
        # Starting indices (i_ind) associated to a set of ending pools containing indices (j_ind)
        syn.connect(condition=concond_ilimit_jpools(i_ind[0], i_ind[-1], j_pool, extraconditions),
                    p=prob, skip_if_invalid=True)

    return

def gaussian_connect_global(
    syn            : b2.Synapses,
    ipools         : list,
    jpools         : list,
    direction      : str,
    sigma          : float,
    amp            : float,
    extraconditions: str = ''
) -> None:
    '''
    Connect synapses in a given direction, with a gaussian distribution.\n
    Ex: gaussian_connect_global(self.S_E, ex_ind_l_axial, ex_ind_l_axial, 'u', EE_sigma_up, EE_amp)
    '''

    if amp == 0 or float(sigma) == 0:
        return

    sigma = float(sigma)
    prob = (
        'clip( {amp} * exp( -( (y_neur_pre-y_neur_post)*(y_neur_pre-y_neur_post) '
        '/ (2*{sigma}*{sigma}*metre*metre) ) ),  0, 1 )'
    ).format(
        amp   = amp,
        sigma = sigma
    )

    rangecond = f"( abs(y_neur_pre - y_neur_post) < 5.1* {sigma} * metre )"  # O(N)
    extraconditions = '( ' + rangecond + ' and ' + extraconditions + ' )'

    cond = concond_ipools_jpools_directional(
        ipools,
        jpools,
        direction,
        extraconditions
    )

    # Starting indices (i_ind) are associated to a set of ending pools containing indices (j_ind)
    syn.connect(condition=cond, p=prob, skip_if_invalid=True)

    return

def gaussian_asymmetric_connect_byidentity(
    syn            : b2.Synapses,
    condlist       : list,
    y_type         : str,
    amp            : float,
    sigma_up       : float,
    sigma_dw       : float,
    extraconditions: str = ''
) -> None:
    '''
    Connect according to a set of conditions regarding the identity of the neurons,
    considering the whole network.\n
    Gaussian distribution used.\n\n
    Ex: conditions_dw = [ [ 'dw', 'ipsi', 'ax', 'ex', 'ax', 'ex' ],
    [ 'dw', 'ipsi', 'ax', 'ex', 'ax', 'ex' ] ]\n
        gaussian_asymmetric_connect_byidentity(self.S_E, conditions_dw, sig_up, sig_dw, amp)
    '''

    assert y_type in ['y_neur', 'y_mech'], 'Invalid y_type'

    sigma_up = float(sigma_up)
    sigma_dw = float(sigma_dw)

    if amp == 0 or (sigma_up == 0 and sigma_dw == 0):
        return

    p_aux = partial(
        '''
            clip(
                {a} * exp(
                    -( ({y}_pre-{y}_post)**2 / (2 * {s}**2 * metre**2) )
                ),
                0,
                1
            )
        '''.format,
        a= amp,
        y= y_type,
    )

    p_up = f'+ {p_aux(s=sigma_up)} * int( {y_type}_pre >= {y_type}_post )' if sigma_up>0 else ''
    p_dw = f'+ {p_aux(s=sigma_dw)} * int( {y_type}_pre <  {y_type}_post )' if sigma_dw>0 else ''
    prob = (p_up + p_dw).replace('    ','').replace('\n','')

    rangecond = f"( abs({y_type}_pre - {y_type}_post) < 5.1* {max(sigma_up, sigma_dw)} * metre )"  # O(N)

    extraconditions = (
        f'( {rangecond} and {extraconditions} )'
        if extraconditions != ''
        else rangecond
    )
    cond = concond_byidentity(condlist, extraconditions)

    syn.connect(condition=cond, p=prob, skip_if_invalid=True)
    return

def satgaussian_connect_byidentity(
    syn            : b2.Synapses,
    condlist       : str,
    y_type         : str,
    pre_ilimits    : list,
    post_ylimits   : list,
    amp            : float,
    sigma_up       : float,
    sigma_dw       : float,
    segment_height : float,
    extraconditions: str ='',
    **kwargs
) -> None:
    '''
    Connect according to a set of conditions regarding the identity of the neurons,
    considering the whole network.\n
    Saturated gaussian distribution used.\n
    Limits are used to determine the rising, steady and falling sections of the
    saturated gaussian distribution, also a subset can be used by substituting with NaN\n
    Ex: conditions_dw = [ [ '', '', '', 'rs', 'ax', '' ], [ '', '', '', 'rs', 'lb', '' ] ]\n
        satgaussian_connect_byidentity(self.S_E, conditions_dw, LAipsi_sigma_dw, LAipsi_amp)
    '''

    if amp == 0:
        return

    assert y_type in ['y_neur', 'y_mech'], 'Invalid y_type'

    tanfac = kwargs.pop('tanfac', 5)

    yrise   = float( post_ylimits[0] )
    yrised  = float( post_ylimits[1] )
    yfall   = float( post_ylimits[2] )
    yfallen = float( post_ylimits[3] )

    sigma_up       = float(sigma_up)
    sigma_dw       = float(sigma_dw)
    segment_height = float(segment_height)

    # Conditions
    cond_ilimits = (
        f'( (i >= {pre_ilimits[0]}) and (i <= {pre_ilimits[-1]}) ) and '
        if pre_ilimits is not None else ''
    )

    cond_ylimits =  '''
        (
            ({ytype}_post >= ({ymin} - {tol} * {sh}) * metre)
            and
            ({ytype}_post <= ({ymax} + {tol} * {sh}) * metre)
        )
    '''.replace('\n', ' ').replace('    ', '').strip()

    exponential_prob = '''
        clip(
            {amp} * tanh(
                {tanfac} * exp( - ( {ytype}_post - {ymax}*metre )**2 / (2 * {sig}**2 * metre**2) )
            ),
            0.000,
            1
        )
    '''.replace('\n', ' ').replace('    ', '').strip()

    cond_all = concond_byidentity(condlist, extraconditions)

    cond_ylimits     = partial(
        cond_ylimits.format,
        ytype = y_type,
        tol   = 0.01,
        sh    = segment_height
    )

    exponential_prob = partial(
        exponential_prob.format,
        amp    = amp,
        ytype  = y_type,
        tanfac = tanfac
    )

    # Upper limit (upward oriented)
    if not isnan(yrise) and not isnan(yrised):
        condrise = f'( {cond_ilimits} {cond_ylimits(ymin= yrise, ymax= yrised)} ) and {cond_all}'
        probrise = exponential_prob(sig = sigma_up, ymax = yrised )

        syn.connect(condition=condrise, p=probrise, skip_if_invalid=True)

    # Central part
    if not isnan(yrised) and not isnan(yfall):
        condplateux = f'( {cond_ilimits} {cond_ylimits(ymin= yrised, ymax= yfall)} ) and {cond_all}'
        syn.connect(condition=condplateux, p=amp)

    # Lower limit (downward oriented)
    if not isnan(yfall) and not isnan(yfallen):
        condfall = f'( {cond_ilimits} {cond_ylimits(ymin= yfall, ymax= yfallen)} ) and {cond_all}'
        probfall = exponential_prob(sig= sigma_dw, ymax= yfall)

        syn.connect(condition=condfall, p=probfall, skip_if_invalid=True)

    return

def trapezoidal_connect_byidentity(
    syn            : b2.Synapses,
    condlist       : str,
    y_type         : str,
    pre_ilimits    : list,
    post_ylimits   : list,
    amp            : float,
    sigma_up       : float,
    sigma_dw       : float,
    segment_height : float,
    extraconditions: str ='',
    **kwargs
) -> None:
    '''
    Connect according to a set of conditions regarding the identity of the neurons,
    considering the whole network.\n
    Trapezoidal distribution used.\n
    Limits are used to determine the rising, steady and falling sections of the
    Trapezoidal distribution\n
    Ex: conditions_dw = [ [ '', '', '', 'rs', 'ax', '' ], [ '', '', '', 'rs', 'lb', '' ] ]\n
        satgaussian_connect_byidentity(self.S_E, conditions_dw, LAipsi_sigma_dw, LAipsi_amp)
    '''

    if amp == 0:
        return

    assert y_type in ['y_neur', 'y_mech'], 'Invalid y_type'


    y_str = float( post_ylimits[0] )
    y_end = float( post_ylimits[1] )

    sigma_up = float(sigma_up)
    sigma_dw = float(sigma_dw)
    delta_up = sigma_up / 2
    delta_dw = sigma_dw / 2

    tolerance      = 0.01
    segment_height = float(segment_height)
    y_err          = tolerance * segment_height

    # Conditions
    cond_ilimits = (
        f'( (i >= {pre_ilimits[0]}) and (i <= {pre_ilimits[-1]}) ) and '
        if pre_ilimits is not None else ''
    )

    cond_ylimits =  '''
        (
            ({ytype}_post >= {ymin} * metre)
            and
            ({ytype}_post <= {ymax} * metre)
        )
    '''.replace('\n', ' ').replace('    ', '').strip()

    linear_prob = '''
        {amp} * clip(
            ( {sig}*metre - abs( {ytype}_post - {ymax}*metre ) ) / ({sig}*metre),
            0.000,
            1.000,
        )
    '''.replace('\n', ' ').replace('    ', '').strip()

    cond_all = concond_byidentity(condlist, extraconditions)

    # Partial conditions
    cond_ylimits     = partial(
        cond_ylimits.format,
        ytype = y_type,
    )

    linear_prob = partial(
        linear_prob.format,
        amp    = amp,
        ytype  = y_type,
    )

    # Central part (closed interval)
    y0 = y_str + delta_up
    y1 = y_end - delta_dw
    condplateux = f'( {cond_ilimits} {cond_ylimits(ymin= y0, ymax= y1)} ) and {cond_all}'
    syn.connect(condition=condplateux, p=amp)

    # Ascending connections (open interval)
    if sigma_up:
        y0    = y_str - delta_up
        y1    = y_str + delta_up
        condrise = f'( {cond_ilimits} {cond_ylimits(ymin= y0 + y_err, ymax= y1 - y_err)} ) and {cond_all}'
        probrise = linear_prob(sig = sigma_up, ymax = y1 )
        syn.connect(condition=condrise, p=probrise, skip_if_invalid=True)

    # Descending connections (open interval)
    if sigma_dw:
        y0 = y_end - delta_dw
        y1 = y_end + delta_dw
        condfall = f'( {cond_ilimits} {cond_ylimits(ymin= y0 + y_err, ymax= y1 - y_err)} ) and {cond_all}'
        probfall = linear_prob(sig= sigma_dw, ymax= y0)
        syn.connect(condition=condfall, p=probfall, skip_if_invalid=True)

    return


# \-------- [ FUNCTIONS TO CONNECT SYNAPSES ] --------
