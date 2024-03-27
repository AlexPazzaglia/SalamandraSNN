'''
Module to store the auxiliary functions used to extract or
manipulate the neuronal indices of the networks
'''

import numpy as np
import brian2 as b2

# -------- [ FUNCTIONS TO RETURN NEURONS' INDICES ] --------

def indices_from_condition( pop: b2.NeuronGroup,
                            ner_type: str = None,
                            side: str = None,
                            net_type: str = None,
                            position: tuple[str, list] = None,
                            lb_ind: int = None,
                            lb_side: str = None,
                            lb_dof: int = None,
                            one_sided: str = None,
                            ex_only: bool = None,
                        ) -> np.ndarray:
    '''
    Indices satisfying given conditions for type and position

    - ner_type : str ('ex''in''mn''rs''ps''es''mc')
    - side : str ('l', 'r')
    - net_type : str ('ax', 'lb')
    - position : tuple ( orientation, [y0, y1] ) where orientation is in ['up', 'dw', 'bw']
    - lb_ind : int
    - lb_side : str ('f','e')
    - lb_dof : int
    - one_sided: str ('f','e')
    - ex_only: bool
    '''

    # Neural parameters
    ner_id = np.array(pop.ner_id)
    side_id = np.array(pop.side_id)
    limb_id = np.array(pop.limb_id)
    ypos = np.array(pop.y_neur)

    n_tot = len(pop)
    matching_inds_bool = np.ones(n_tot, dtype=bool)
    matching_inds = np.arange(n_tot, dtype=int)

    # Condition on neuron type
    if ner_type is not None:
        type_dict = {
            'ex': 0,  # excitatory
            'in': 1,  # inhibitory
            'mn': 2,  # motor neuron
            'rs': 3,  # reticulospinal
            'ps': 4,  # propriosensory neuron
            'es': 5,  # exterosensory neuron
            'mc': 6,  # muscle cell
        }

        if isinstance(ner_type, list):
            condition = np.zeros(n_tot, dtype=bool)
            for n_type in ner_type:
                target_type = type_dict[n_type]
                condition = np.logical_or( condition, (ner_id == target_type))
        else:
            target_type = type_dict[ner_type]
            condition = (ner_id == target_type)

        # Update matching indices
        matching_inds_bool = np.logical_and( matching_inds_bool, condition )

    # Condition on side
    if side is not None:
        side_dict = {
            'l': -1,
            'r': +1,
        }
        target_side = side_dict[side]
        condition = (side_id * target_side > 0)

        # Update matching indices
        matching_inds_bool = np.logical_and( matching_inds_bool, condition )

    # Condition on limb or axial network
    if net_type is not None:
        if   net_type == 'ax':
            condition = abs(side_id) == 1
        elif net_type == 'lb':
            condition = abs(side_id) >= 2
        else:
            raise ValueError('net_type is either ax or lb')

        # Update matching indices
        matching_inds_bool = np.logical_and( matching_inds_bool, condition )

    # Condition on limb index
    if lb_ind is not None:

        if isinstance(lb_ind, list):
            condition = np.zeros(n_tot, dtype=bool)
            for lb_i in lb_ind:
                condition = np.logical_or( condition, ( limb_id == lb_i ) )
        else:
            condition = ( limb_id == lb_ind )

        # Update matching indices
        matching_inds_bool = np.logical_and( matching_inds_bool, condition )

    # Condition on limb side
    if lb_side is not None:
        lb_side_dict = {
            'f': 0,
            'e': 1,
        }
        target_lb_side = lb_side_dict[lb_side]
        condition = ( abs(side_id) % 2 == target_lb_side )

        # Update matching indices
        matching_inds_bool = np.logical_and( matching_inds_bool, condition )

    # Condition on limb degree of freedom
    if lb_dof is not None:

        if isinstance(lb_dof, list):
            condition = np.zeros(n_tot, dtype=bool)
            for lb_d in lb_dof:
                condition = np.logical_or( condition, ( abs(side_id) == lb_d ) )
        else:
            condition = ( abs(side_id) == lb_dof )

        # Update matching indices
        matching_inds_bool = np.logical_and( matching_inds_bool, condition )

    # Condition on axial position
    if position is not None:
        orientation = position[0]
        ylims = position[1]

        if orientation == 'up':
            condition = ypos > ylims[0]
        elif orientation == 'dw':
            condition = ypos < ylims[0]
        elif orientation == 'bw':
            condition = np.logical_and( ypos > ylims[0], ypos < ylims[1] )
        if orientation == 'upeq':
            condition = ypos >= ylims[0]
        elif orientation == 'dweq':
            condition = ypos <= ylims[0]
        elif orientation == 'bweq':
            condition =np.logical_and( ypos >= ylims[0], ypos <= ylims[1] )
        else:
            raise ValueError('orientation must be in [up, dw, bw, upeq, dweq, bweq]')

        # Update matching indices
        matching_inds_bool = np.logical_and( matching_inds_bool, condition )

    # Condition one_sided (left + flex or right + extend)
    if one_sided is not None:
        if one_sided not in ['f', 'e']:
            raise ValueError('one_sided must be in [f, e]')

        condition = _indices_bool_onesided(
            pop,
            axis_side = 'l' if one_sided=='f' else 'r',
            lb_side = one_sided,
        )
        # Update matching indices
        matching_inds_bool = np.logical_and( matching_inds_bool, condition )

    # Condition ex_only (discard inhibitory)
    if ex_only is not None:
        condition = _indices_bool_exonly(pop, ex_only)
        # Update matching indices
        matching_inds_bool = np.logical_and( matching_inds_bool, condition )

    return np.sort( matching_inds[matching_inds_bool] )

def _indices_bool_onesided(pop: b2.NeuronGroup,
                    axis_side: str,
                    lb_side: str,
                    ner_type: str = None ) -> np.ndarray:
    '''
    Returns indices of neurons from one side (l,r) of the axial networks
    and from one dof (f,e) of the limb network
    '''

    inds_axis_side = indices_from_condition(
        pop,
        ner_type = ner_type,
        side = axis_side,
        net_type = 'ax',
    )
    inds_limbs_side = indices_from_condition(
        pop,
        ner_type = ner_type,
        lb_side = lb_side,
        net_type = 'lb',
    )
    target_inds = np.sort( np.concatenate( (inds_axis_side, inds_limbs_side) ) )

    # Boolean array
    n_tot = len(pop)
    matching_inds = np.isin(np.arange(n_tot, dtype= int), target_inds)

    return matching_inds

def _indices_bool_exonly( pop: b2.NeuronGroup,
                    ex_only: bool ) -> np.ndarray:
    '''
    Returns indices of neurons from one side (l,r) of the axial networks
    and from one dof (f,e) of the limb network
    '''

    all_ner_types = ['ex','in','mn','rs','ps','es','mc']
    target_types = all_ner_types if not ex_only else all_ner_types.remove('in')
    target_inds = indices_from_condition(pop, ner_type= target_types)

    # Boolean array
    n_tot = len(pop)
    matching_inds = np.isin(np.arange(n_tot, dtype= int), target_inds)

    return matching_inds

# \------- [ FUNCTIONS TO RETURN NEURONS' INDICES ] --------

# -------- [ FUNCTIONS TO RETURN POOLS' INDICES ] --------
def cpg_seg_pools_indices(  segments: int,
                        n_e_semi: int,
                        n_i_semi: int
                    ) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    '''
    Indeces of axial neuron pool (ex_l, ex_r, in_l, in_r).\n
    Each output is a list containing the lists of indices for the corresponding pool and segments.\n
    Ex: ex_l[i][j] represents the j-th neuron of the i-th segment in the left excitatory pool
    '''

    n_ex = 2*n_e_semi
    n_seg = n_ex + 2*n_i_semi

    ex_l = [
        [x + y * n_seg for x in range(n_e_semi)]
        for y in range(segments)
    ]
    ex_r = [
        [x + y * n_seg + n_e_semi for x in range(n_e_semi)]
        for y in range(segments)
    ]
    in_l = [
        [x + y * n_seg + n_ex for x in range(n_i_semi)]
        for y in range(segments)
    ]
    in_r = [
        [x + y * n_seg + n_ex + n_i_semi for x in range(n_i_semi)]
        for y in range(segments)
    ]

    return ex_l, ex_r, in_l, in_r

def cpg_seg_sides_indices(  segments: int,
                        n_e_semi: int,
                        n_i_semi: int) -> tuple[list[list[int]], list[list[int]]]:
    '''
    Indeces of left and right populations (left, right).\n
    Each output is a list containing the lists of indices for the corresponding side and segment.\n
    '''

    n_ex = 2*n_e_semi
    n_seg = n_ex + 2*n_i_semi

    l_ind = [
        [ seg * n_seg + neur        for neur in range(n_e_semi) ] +
        [ seg * n_seg + neur + n_ex for neur in range(n_i_semi) ]
        for seg in range(segments)
    ]

    r_ind = [
        [ seg * n_seg + neur + n_e_semi        for neur in range(n_e_semi) ] +
        [ seg * n_seg + neur + n_ex + n_i_semi for neur in range(n_i_semi) ]
        for seg in range(segments)
    ]

    return l_ind, r_ind

def cpg_types_indices(  segments: int,
                        n_e_semi: int,
                        n_i_semi: int,
                        one_sided: bool = False) -> tuple[list[list[int]], list[list[int]]]:
    '''
    Indeces of excitatory and inhibitory neurons.\n
    Each output is a list containing the lists of indices for the
    corresponding neuron type and segment.\n
    '''

    side_factor = 2 if not one_sided else 1

    n_ex = side_factor * n_e_semi
    n_in = side_factor * n_i_semi
    n_seg = n_ex + n_in

    ex_ind = [x + y * n_seg for y in range(segments) for x in range(n_ex)]
    in_ind = [x + y * n_seg + n_ex for y in range(segments) for x in range(n_in)]

    return ex_ind, in_ind

def cpg_segments_indices(   segments: int,
                            n_e_semi: int,
                            n_i_semi: int,
                            one_sided: bool = False) -> list[list[int]]:
    '''
    Indeces of neurons for every segment.\n
    Each output is a list containing the lists of indices for the corresponding segment.\n
    '''
    side_factor = 2 if not one_sided else 1
    n_seg = side_factor * (n_e_semi + n_i_semi)

    return [[x + y * n_seg for x in range(n_seg)] for y in range(segments)]

def cpg_middles_indices(segments: int,
                        n_e_semi: int,
                        n_i_semi: int,
                        ex_only: bool = False) -> tuple[list[list[int]], int]:
    '''
    Indeces of "middle" neurons in the populations (ex_l, ex_r, in_l, in_r)
    Output is a list containing the lists of indices representing
    the middle individual in every neuronal pool.\n
    - If ex_Only = True --> Return only the indices of the excitatory pools and indmax = 2
    - If ex_Only = False --> Return indices of the excitatory and inhibitory pools and indmax = 4
    '''

    n_ex = 2*n_e_semi
    n_seg = n_ex + 2*n_i_semi

    if ex_only:
        m_ind = [[x * n_ex + n_e_semi//2, x * n_ex + n_e_semi + n_e_semi//2]
                 for x in range(segments)]
        indmax = 2
    else:
        m_ind = [[x * n_seg + n_e_semi//2, x * n_seg + n_e_semi + n_e_semi//2,
                  x * n_seg + n_ex + n_i_semi//2, x * n_seg + n_ex + n_i_semi + n_i_semi//2]
                 for x in range(segments)]
        indmax = 4

    return m_ind, indmax

def cpg_lines_indices(  segments: int,
                        n_e_semi: int,
                        n_i_semi: int,
                        ex_only: bool = False) -> tuple[list[list[int]], int]:
    '''
    Indeces of the lines in the grids. Used to separate segments and pools.
    - If ex_Only = True --> Return only the indices of the excitatory pools and indmax = 2
    - If ex_Only = False --> Return  indices of the excitatory and inhibitory pools and indmax = 4
    '''

    n_ex = 2*n_e_semi
    n_seg = n_ex + 2*n_i_semi

    if ex_only:
        l_ind = [[-0.5 + x * n_ex + n_e_semi, -0.5 + x * n_ex + n_ex]
                 for x in range(segments)]
        indmax = 2
    else:
        l_ind = [[-0.5 + x * n_seg + n_e_semi,
                  -0.5 + x * n_seg + n_e_semi + n_e_semi,
                  -0.5 + x * n_seg + n_ex + n_i_semi,
                  -0.5 + x * n_seg + n_ex + n_i_semi + n_i_semi] for x in range(segments)]
        indmax = 4

    return l_ind, indmax

def cpg_limbs_indices(  segments: int,
                        n_e_semi: int,
                        n_i_semi: int,
                        limbs: int,
                        hemisegments_per_limb: int) -> list[list[int]]:
    '''
    Indeces of neurons for every limb.\n
    Each output is a list containing the lists of indices for the corresponding limb.\n
    '''
    n_seg = 2 * (n_e_semi + n_i_semi)
    n_cpg = n_seg * segments
    n_lmb = n_seg * hemisegments_per_limb // 2
    aux_inds = np.arange(n_cpg, n_cpg + n_lmb, dtype=int)

    return [ aux_inds + (lmb_ind * n_lmb) for lmb_ind in range(limbs) ]

def net_middles_indices(segments: int,
                        n_e_semi: int,
                        n_i_semi: int,
                        additional_pools_n: list[int] = None,
                        one_sided: bool = False,
                        ex_only: bool = False,
                        densegrid: bool = True) -> list[list[int]]:
    '''
    Function to compute the indices of the middle elements for each pool of the network
    '''

    if len(additional_pools_n) == 0:
        additional_pools_n = []
    additional_pools_n = np.array(additional_pools_n)

    aux = {
        'e_025' : n_e_semi//2,
        'e_075' : n_e_semi + n_e_semi//2,
        'e_050' : n_e_semi,
        'e_100' : n_e_semi * 2,
        'i_025' : n_i_semi//2,
        'i_075' : n_i_semi + n_i_semi//2,
        'i_050' : n_i_semi,
        'i_100' : n_i_semi * 2,
    }

    # Middle indices to consider in each segment
    if densegrid:
        if not one_sided and not ex_only:
            segmax = aux['e_100'] + aux['i_100']
            segment_inds = [ aux['e_025'],
                             aux['e_075'],
                             aux['e_100'] + aux['i_025'],
                             aux['e_100'] + aux['i_075'] ]

        elif not one_sided and ex_only:
            segmax = aux['e_100']
            segment_inds = [ aux['e_025'],
                             aux['e_075'] ]

        elif one_sided and not ex_only:
            segmax = aux['e_50'] + aux['i_50']
            segment_inds = [ aux['e_025'],
                             aux['e_050'] + aux['i_025'] ]

        elif one_sided and ex_only:
            segmax = aux['e_050']
            segment_inds = [ aux['e_025'] ]

    else:
        if not one_sided and not ex_only:
            segmax = aux['e_100'] + aux['i_100']
            segment_inds = [ aux['e_050'],
                             aux['e_100'] + aux['i_050'] ]

        elif not one_sided and ex_only:
            segmax = aux['e_100']
            segment_inds = [ aux['e_050'] ]

        elif one_sided and not ex_only:
            segmax = aux['e_050'] + aux['i_050']
            segment_inds = [ aux['e_025'],
                             aux['e_050'] + aux['i_025'] ]

        elif one_sided and ex_only:
            segmax = aux['e_050']
            segment_inds = [ aux['e_025'] ]

    # Create the indices for all the segments in the cpg network
    n_cpg = segmax * segments
    m_ind = [ segmax * seg + np.array(segment_inds) for seg in range(segments) ]

    # Additional pools
    for i, _ in enumerate(additional_pools_n):
        sum_i = np.sum(additional_pools_n[:i]) + additional_pools_n[i]//2
        m_ind.append(n_cpg + sum_i)

    return m_ind

def net_lines_indices(  segments: int,
                        n_e_semi: int,
                        n_i_semi: int,
                        one_sided: bool = False,
                        n_additional_pools: list[int] = None,
                        ex_only: bool = False,
                        densegrid: bool = True) -> list[list[int]]:
    '''
    Function to compute the indices of the lines to be placed to separate the different pools
    Lines are placed in correspondence of the first and last element of each pool
    '''

    if len(n_additional_pools) == 0:
        n_additional_pools = []
    n_additional_pools = np.array(n_additional_pools)

    # Indices to consider in each segment
    aux = {
        'e_050' : n_e_semi,
        'e_100' : n_e_semi * 2,
        'i_050' : n_i_semi,
        'i_100' : n_i_semi * 2,
    }

    if densegrid:
        if not one_sided and not ex_only:
            segment_inds = [ aux['e_050'],
                             aux['e_100'],
                             aux['e_100'] + aux['i_050'],
                             aux['e_100'] + aux['i_100'] ]

        elif not one_sided and ex_only:
            segment_inds = [ aux['e_050'],
                             aux['e_100'] ]

        elif one_sided and not ex_only:
            segment_inds = [ aux['e_050'],
                             aux['e_050'] + aux['i_050'] ]

        elif one_sided and ex_only:
            segment_inds = [ aux['e_050'] ]

    else:
        if not one_sided and not ex_only:
            segment_inds = [ aux['e_100'],
                             aux['e_100'] + aux['i_100'] ]

        elif not one_sided and ex_only:
            segment_inds = [ aux['e_100'] ]

        elif one_sided and not ex_only:
            segment_inds = [ aux['e_050'],
                             aux['e_050'] + aux['i_050'] ]

        elif one_sided and ex_only:
            segment_inds = [ aux['e_050'] ]

    segmax = segment_inds[-1]

    # Create the indices for all the segments in the cpg network
    n_cpg = segmax * segments
    l_ind = [ segmax * seg + np.array(segment_inds) for seg in range(segments) ]

    # Additional pools
    for i, _ in enumerate(n_additional_pools):
        sum_i = np.sum( n_additional_pools[:i+1] )
        l_ind.append( n_cpg + sum_i )

    return l_ind

def oriented_neighboring_pools( pools_indices: list[list[int]],
                                starting_seg: int,
                                seg_range: int,
                                seg_limits: list[int]) -> list[list[int]]:
    '''
    Indeces of a given pool type (ex, in, left, right) adjacent
    to a starting segment in a desired direction
    '''

    # Ascending or descending connections
    if seg_range >= 0:
        seg_range += 1
        step = 1
    else:
        seg_range -= 1
        step = -1

    # Collect the desired pools
    out_pools = []
    for ind in range(starting_seg, starting_seg + seg_range, step):
        if ind < seg_limits[0] or ind >= seg_limits[1]:
            break
        out_pools.append(pools_indices[ind])

    return out_pools

# \-------- [ FUNCTIONS TO RETUNR POOLS' INDICES ] --------

# -------- [ FUNCTIONS TO MANIPULATE INDICES ] --------
def get_mapped_pools_indices(indices: int,
                n_tot: int,
                n_e_semi: int,
                n_i_semi: int,
                segments_ax: int,
                pos_segments_lb: list[int],
                one_sided: bool = False,
                times: np.ndarray = None,
                insert_limbs: bool = False,
                ex_only: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:

    ''' Return the indices (and optionally times) of the spikes for axis, limbs and others'''

    indices = np.array(indices, dtype= int)
    times = np.array(times)

    segments_lb = len(pos_segments_lb)
    n_ex_seg = 2 * (n_e_semi )
    n_seg = 2 * (n_e_semi + n_i_semi)
    n_ax = n_seg * segments_ax
    n_cpg = n_seg * (segments_ax + segments_lb)

    pools = {
        'ex_l' : np.arange(n_e_semi, dtype= int),
        'ex_r' : n_e_semi + np.arange(n_e_semi, dtype= int),
        'in_l' : n_ex_seg + np.arange(n_i_semi, dtype= int),
        'in_r' : n_ex_seg + n_i_semi + np.arange(n_i_semi, dtype= int),
    }

    if ex_only:
        pools.pop('in_l', None)
        pools.pop('in_r', None)
    if one_sided:
        pools.pop('ex_r', None)
        pools.pop('in_r', None)

    ranges =  np.concatenate( [
                                    pools.get('ex_l', []),
                                    pools.get('ex_r', []),
                                    pools.get('in_l', []),
                                    pools.get('in_r', [])
                                ] )

    seg_inds = np.concatenate( [seg * n_seg + ranges for seg in range(segments_ax + segments_lb)] )
    seg_inds = np.concatenate( [seg_inds, np.arange(n_cpg, n_tot, dtype= int)] )

    # Select desired indices
    times = times[ np.isin(indices, seg_inds) ]
    indices = indices[ np.isin(indices, seg_inds) ]

    if len(times):
        axis_out = np.array(
            [ times[ (indices < n_ax) ],
              indices[ (indices < n_ax)] ]
        ).T

        limb_out = np.array(
            [ times[ (indices >= n_ax) & (indices < n_cpg)],
              indices[ (indices >= n_ax) & (indices < n_cpg) ] ]
        ).T

        other_out = np.array(
            [ times[ (indices >= n_cpg) ],
              indices[ (indices >= n_cpg) ] ]
        ).T

    else:
        axis_out = np.array( indices[ (indices < n_ax) ] )
        limb_out = np.array( indices[ (indices >= n_ax) & (indices < n_cpg) ] )
        other_out = np.array( indices[ (indices >= n_cpg) ] )

    # Mask to insert the limbs inside the network
    if insert_limbs:
        limb_mask = _insert_limb_mask( n_tot, n_e_semi, n_i_semi, segments_ax, pos_segments_lb )
        # Apply mask to the indices
        if len(times):
            axis_out[:,1] = limb_mask[ np.array( axis_out[:,1], dtype= int ) ]
            limb_out[:,1] = limb_mask[ np.array( limb_out[:,1], dtype= int ) ]
            other_out[:,1] = limb_mask[ np.array( other_out[:,1], dtype= int ) ]
        else:
            axis_out = limb_mask[ np.array(axis_out, dtype= int) ]
            limb_out = limb_mask[ np.array(limb_out, dtype= int) ]
            other_out = limb_mask[ np.array(other_out, dtype= int) ]

    # Create mask to account for one_sided and ex_only
    seg_mask = np.zeros((4), dtype= int)
    if pools.get('ex_l', None) is None:
        seg_mask[0] = n_e_semi

    if pools.get('ex_r', None) is None:
        seg_mask[1] = n_e_semi

    if pools.get('in_l', None) is None:
        seg_mask[2] = n_i_semi

    if pools.get('in_r', None) is None:
        seg_mask[3] = n_i_semi

    seg_mask_sum = np.sum(seg_mask)

    mask = np.arange(n_tot, dtype= int)
    mask[n_cpg:] -= (segments_ax + segments_lb) * seg_mask_sum

    for seg in range(segments_ax + segments_lb):
        ex_strt = seg * n_seg
        ex_stop = seg * n_seg + n_ex_seg
        in_strt = seg * n_seg + n_ex_seg
        in_stop = seg * n_seg + n_seg

        if seg_mask[0]:
            mask[ ex_strt : ex_strt + n_e_semi ] = -1
        else:
            mask[ ex_strt : ex_strt + n_e_semi ] -= ( seg * seg_mask_sum + np.sum(seg_mask[:0]) )

        if seg_mask[1]:
            mask[ ex_strt + n_e_semi : ex_stop ] = -1
        else:
            mask[ ex_strt + n_e_semi : ex_stop ] -= ( seg * seg_mask_sum + np.sum(seg_mask[:1]) )

        if seg_mask[2]:
            mask[ in_strt : in_strt + n_i_semi ] = -1
        else:
            mask[ in_strt : in_strt + n_i_semi ] -= ( seg * seg_mask_sum + np.sum(seg_mask[:2]) )

        if seg_mask[3]:
            mask[ in_strt + n_i_semi : in_stop ] = -1
        else:
            mask[ in_strt + n_i_semi : in_stop ] -= ( seg * seg_mask_sum + np.sum(seg_mask[:3]) )


    # Index of the last element in the cpg network
    n_cpg_remap = n_cpg - (segments_ax + segments_lb) * seg_mask_sum

    # Apply mask to the indices
    if len(times):
        axis_out[:,1] = mask[ np.array( axis_out[:,1], dtype= int) ]
        limb_out[:,1] = mask[ np.array( limb_out[:,1], dtype= int) ]
        other_out[:,1] = mask[ np.array( other_out[:,1], dtype= int) ]
    else:
        axis_out = mask[ np.array( axis_out, dtype= int) ]
        limb_out = mask[ np.array( limb_out, dtype= int) ]
        other_out = mask[ np.array( other_out, dtype= int) ]


    return axis_out, limb_out, other_out, n_cpg_remap

def _insert_limb_mask(   n_tot: int,
                        n_e_semi: int,
                        n_i_semi: int,
                        segments: int,
                        limbpos: list[int]) -> np.ndarray:
    '''
    Returns the map to insert the indices of the limbs within the axial population.
    '''

    n_seg = 2 * (n_e_semi + n_i_semi)
    n_ax = n_seg * segments

    mask = np.arange(n_tot, dtype= int)
    segrange = np.arange(n_seg, dtype= int)

    for k, lbpos in enumerate(limbpos):
        mask[ n_ax + k*n_seg : n_ax + k*n_seg + n_seg ] = lbpos * n_seg \
                                                         + k*n_seg + segrange
        mask[ lbpos * n_seg : n_ax ] += n_seg

    return np.array(mask, dtype= int)

def get_mapped_pools_limits(n_tot: int,
                            target_indices: list[int],
                            pools_limits: tuple[list[int]] ) -> list[list[int]]:
    '''
    Maps original pools' limits to the new domain specified by target_indices
    '''

    pruned_inds = np.ones(n_tot)
    pruned_inds[target_indices] = 0

    inds_mask = np.array(
        [ np.sum(pruned_inds[:i]) for i in range(n_tot) ],
        dtype= int
    )

    # Mapped pools indices
    pools_limits_mapped = np.array(
        [
            np.array(lims, dtype= int) - inds_mask[lims]
            for lims in pools_limits
        ]
    )
    return pools_limits_mapped

# -------- [ FUNCTIONS TO MANIPULATE INDICES ] --------
