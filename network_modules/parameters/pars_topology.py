''' New way of defining the topology of the network '''
import yaml
import json
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from brian2 import get_dimensions, meter

from network_modules.parameters.network_module import SnnNetworkModule
from network_modules.parameters.network_neuron import SnnNeuron

from network_modules.parameters.pars_utils import SnnPars
from network_modules.parameters.pars_limb_connectivity import SnnParsLimbConnectivity

class SnnParsTopology(SnnPars):
    ''' Class to define the core organization of the network '''

    def __init__(
        self,
        parsname : str,
        new_pars : dict = None,
        pars_path: str = None,
        **kwargs
    ):
        if pars_path is None:
            pars_path = 'network_parameters/parameters_topology'

        super().__init__(
            pars_path = pars_path,
            parsname  = parsname,
            new_pars  = new_pars,
            pars_type = 'parameters_topology',
            **kwargs
        )

        # Parameters modifying the connectivity
        self.connectivity_axial_newpars : dict[str, list[dict]]= self.params_to_update.pop('connectivity_axial_newpars', {})
        self.connectivity_limbs_newpars : dict[str, list[dict]]= self.params_to_update.pop('connectivity_limbs_newpars', {})

        # Inter-limb connectivity parameters
        self.__pars_limb_connectivity_filename : str = self.pars.pop('pars_limb_connectivity_filename')
        self.__limb_connectivity_scheme        : str = self.pars.pop('limb_connectivity_scheme')

        self.pars_limb_conn = SnnParsLimbConnectivity(
            parsname            = self.pars_limb_connectivity_filename,
            connectivity_scheme = self.limb_connectivity_scheme
        )

        # Axis and limb connectivities
        self.__connectivity_axial_filename : str = self.pars.pop('connectivity_axial_filename', {})
        self.__connectivity_limbs_filename : str = self.pars.pop('connectivity_limbs_filename', {})

        self.__trunk_tail_discontinuity_flag : bool = self.pars.pop('trunk_tail_discontinuity_flag')

        # Ascending feedback
        self.__ascending_feedback_list : tuple[str] = tuple(self.pars.pop('ascending_feedback'))
        self.__ascending_feedback_flag        : int = self.pars.pop('ascending_feedback_flag')
        self.__ascending_feedback_str         : str = self.ascending_feedback_list[self.ascending_feedback_flag]

        self.__ascfb    = bool(self.ascending_feedback_flag)
        self.__ascfb_lb = bool(self.ascending_feedback_flag)

        # Parameters
        self.__length_axial            : float       = float(self.pars.pop('length_axial')) * meter
        self.__segments_axial          : int         = self.pars.pop('segments_axial')
        self.__segments_per_limb       : int         = self.pars.pop('segments_per_limb')
        self.__limbs                   : int         = self.pars.pop('limbs')
        self.__limbs_f_positions       : tuple[float] = tuple( self.pars.pop('limbs_positions') )
        self.__limbs_pairs_f_positions : tuple[float] = tuple( self.limbs_f_positions[::2] )
        self.__limbs_y_positions       : tuple[float] = tuple( [ x * self.length_axial for x in self.limbs_f_positions] )
        self.__limbs_pairs_y_positions : tuple[float] = tuple( self.limbs_y_positions[::2] )
        self.__limbs_i_positions       : tuple[int]   = tuple( [ round( x * self.segments_axial ) for x in self.limbs_f_positions] )
        self.__limbs_pairs_i_positions : tuple[int]   = tuple( self.limbs_i_positions[::2] )

        self.__length_segment = self.length_axial / self.segments_axial if self.segments_axial else self.length_axial
        self.__segments_limbs = self.segments_per_limb * self.limbs

        self.__define_modules(self.pars.pop('topology'))
        self.__define_neurons_lists()
        self.__load_connectivity_schemes()
        self.__define_derived_parameters()
        self.consistency_checks()

    # BUILDING
    def __define_network_subpart(
        self,
        m_pars : dict, # Parameters of the parent part
        d_pars : dict, # Parameters of the subpart
    ):
        ''' Sub-part of a parent part of the network '''

        # Parent
        p_name         : str       = m_pars.get('name')
        p_neuron_group : str       = m_pars.get('neuron_group')
        p_include      : bool      = m_pars.get('include', True)
        p_depth        : int       = m_pars.get('depth', 0)
        p_pools        : int       = m_pars.get('pools', None)
        p_sides_list   : list[str] = m_pars.get('sides_list', None)
        p_n_pool       : int       = m_pars.get('n_pool', None)
        p_copies       : int       = m_pars.get('copies', 1)

        p_rows                   : int               = m_pars.get('pool_rows', 1)
        p_pools_names_aux        : list[str]         = m_pars.get('pools_names_aux', None)
        p_pools_positions_limits : list[list[float]] = m_pars.get('pools_positions_limits', None)
        p_plotting               : dict[str, str]    = m_pars.get('plotting', {})

        # Daughter
        d_name                   : str               = d_pars.get('name')
        d_n_pool                 : int               = d_pars.get(               'n_pool',  None)
        d_fraction               : float             = d_pars.get(             'fraction',  None)
        d_sub_parts_pars         : list[dict]        = d_pars.get('sub_parts_description',    [])
        d_include                : bool              = d_pars.get(              'include',  True) if p_include else False
        d_neuron_group           : str               = d_pars.get(          'neuron_group',           p_neuron_group)
        d_pools                  : int               = d_pars.get(                 'pools',                  p_pools)
        d_sides_list             : list[str]         = d_pars.get(            'sides_list',             p_sides_list)
        d_pools_names_aux        : list[str]         = d_pars.get(       'pools_names_aux',        p_pools_names_aux)
        d_rows                   : int               = d_pars.get(             'pool_rows',                   p_rows)
        d_pools_positions_limits : list[list[float]] = d_pars.get('pools_positions_limits', p_pools_positions_limits)
        d_copies                 : int               = d_pars.get(                'copies',                 p_copies)
        d_plotting               : dict[str, str]    = d_pars.get(              'plotting',               p_plotting)

        # BUILD DAUGHTER
        daughter = {
            'name'        : f'{p_name}.{d_name}',
            'include'     : d_include,
            'neuron_group': d_neuron_group,
            'type'        : d_name,
            'depth'       : p_depth + 1,
            'copies'      : d_copies,
            'sides_list'  : d_sides_list,
            'sides'       : len(d_sides_list),
            'pools'       : d_pools,
            'pools_copy'  : d_pools // d_copies if d_copies else d_pools,
            'pool_rows'   : d_rows,
            'plotting'    : d_plotting,

            'n'                     : 0,
            'n_side'                : 0,
            'n_pool'                : 0,
            'n_pool_side'           : 0,
            'pools_names'           : [],
            'pools_names_aux'       : [],
            'pools_positions_limits': [],
            'sub_parts_list'        : [],
        }

        # DAUGHTER NOT INCLUDED
        if not d_include:
            logging.warning(f"NOT INCLUDED: {daughter['name']}")
            return daughter

        # DAUGHTER INCLUDED
        d_pools_names = (
            [ f"{daughter['name']}.pool_{ind}" for ind in range(d_pools)]
            if d_pools_names_aux is None
            else
            [ f"{daughter['name']}.{pool_name}" for pool_name in d_pools_names_aux]
        )

        if d_pools_positions_limits in [None, []]:
            d_length_pool = self.length_axial / d_pools if d_pools else self.length_axial
            d_pools_positions_limits = [
                [d_length_pool * ind, d_length_pool*(ind+1)]
                for ind in range(d_pools)
            ]
        elif get_dimensions(d_pools_positions_limits[0][0]).is_dimensionless:
            d_pools_positions_limits = [
                [ limit[0] * self.length_axial, limit[1] * self.length_axial]
                for limit in d_pools_positions_limits
            ]

        daughter['pools_names']            = d_pools_names
        daughter['pools_names_aux']        = d_pools_names_aux
        daughter['pools_positions_limits'] = d_pools_positions_limits

        # Define numerosity
        p_numerosity = p_n_pool not in [None, 0] and d_fraction is not None
        d_numerosity = d_n_pool not in [None, 0]

        # Both true    = Non unique definition
        # p_numerosity = Uniquely determined as a fraction
        # d_numerosity = Uniquely determined as a number
        # Both false   = Can still be determined by summing the sub-parts

        if d_numerosity:
            daughter['n_pool'] = d_n_pool
            daughter['n']      = d_n_pool * d_pools

        if p_numerosity:
            daughter['n_pool'] = round( d_fraction * p_n_pool )
            daughter['n']      = round( d_fraction * p_n_pool * d_pools )

        # Define sub-parts
        daughter['sub_parts_list'] = [
            self.__define_network_subpart(
                daughter,
                sub_part_pars,
            )
            for sub_part_pars in d_sub_parts_pars
        ]

        # Add reference to all sub-parts
        for sub_part in daughter['sub_parts_list']:
            daughter[sub_part['type']] = sub_part

        # Add pool numerosity information
        if daughter.get('n') in [None, 0]:
            daughter['n'] = sum( pop['n'] for pop in daughter['sub_parts_list'] )
            daughter['n_pool'] = round( daughter['n'] / d_pools ) if d_pools else 0

        if daughter['n'] == 0:
            daughter.update(
                {
                    'include'               : 0,
                    'copies'                : 0,
                    'sides_list'            : 0,
                    'sides'                 : 0,
                    'pools'                 : 0,
                    'pools_copy'            : 0,
                    'pool_rows'             : 0,
                    'n'                     : 0,
                    'n_side'                : 0,
                    'n_pool'                : 0,
                    'n_pool_side'           : 0,
                    'pools_names'           : [],
                    'pools_names_aux'       : [],
                    'pools_positions_limits': [],
                    'sub_parts_list'        : [],
                }
            )

        else:
            daughter['n_pool_side'] = round( daughter['n_pool'] / daughter['sides'] )
            daughter['n_side']      = round( daughter['n'] / daughter['sides'] )
            daughter['n_copy']      = round( daughter['n'] / daughter['copies'] )
            daughter['n_side_copy'] = round( daughter['n_side'] / daughter['copies'] )

        # CHECKS
        if (
            (daughter['pools_names'] is not None) and
            len(daughter['pools_names']) != daughter['pools']
        ):
            raise ValueError('d_pools_names should have length d_pools')

        if (
            (daughter['pools_positions_limits'] is not None) and
            len(daughter['pools_positions_limits']) != daughter['pools']
        ):
            raise ValueError('d_pools_positions_limits should have length d_pools')

        if (p_n_pool is None or p_n_pool == 0) != (d_fraction is None):
            raise ValueError('p_n_pool and d_fraction should be provided together')

        if p_numerosity and d_numerosity:
            raise ValueError(f'Cannot uniquely determine the numerosity of {d_name}')

        if daughter['n'] and daughter['n_pool'] % daughter['sides'] != 0:
            raise ValueError('n_pool must be a multiple of sides')

        return daughter

    def __prune_branches(self, submodules: list[dict], module: dict = None):
        ''' Remove branches with no neurons '''

        empty_modules =[]
        for submod_ind, submod in enumerate(submodules):
            if not submod['include'] or submod['n'] == 0:
                empty_modules.append(submod_ind)
                continue
            self.__prune_branches(submod['sub_parts_list'], submod)

        for index in sorted(empty_modules, reverse=True):
            if module is not None:
                del module[submodules[index]['type']]
            del submodules[index]

    def __prune_parameters(self, modules: list[dict] ):
        ''' Remove useless parameters'''

        for module in modules:
            module.pop('pools_names_aux', None)

            self.__prune_parameters(module['sub_parts_list'])

    def __get_leaf_modules(self):
        ''' Get list of all leaf modules (i.e. the ones with no sub modules) '''

        # Auxiliary function
        def append_leaves(sub_module: SnnNetworkModule, neuron_group: int, mod_list: list = None):
            ''' Recursively get all leaf modules '''
            if mod_list is None:
                mod_list = []
            elif sub_module.neuron_group != neuron_group:
                return

            if sub_module.sub_parts_list == []:
                mod_list.append(sub_module)
                return

            for sub_sub_module in sub_module.sub_parts_list:
                append_leaves(sub_sub_module, neuron_group, mod_list)

            return mod_list

        # Collect modules
        leaf_modules = [
            append_leaves(self.network_modules, group_id)
            for group_id in self.neuron_groups_ids
        ]

        return leaf_modules

    def __define_modules(self, topology: list[dict]):
        ''' Defines all the modules of the network, their parameters and their sub-parts '''

        ax_p_limits = [ [self.length_segment * ind, self.length_segment * (ind + 1)] for ind in range(self.segments_axial)]
        lb_p_limits = [ [pos, pos + self.length_segment] for pos in self.limbs_y_positions for _seg in range(self.segments_per_limb)]

        sub_modules_names      = (             'axial',                  'limbs' )
        sub_modules_segments   = ( self.segments_axial,   self.segments_per_limb )
        sub_modules_copies     = (                   1,               self.limbs )
        sub_modules_sides      = (          ['l', 'r'],               ['f', 'e'] )
        sub_modules_limits     = (         ax_p_limits,              lb_p_limits )
        sub_modules_pool_names = (                None, ['LF', 'RF', 'LH', 'RH'] )

        network_modules_list = []
        for module_description in topology:
            # CPG, RS, PS etc.

            module = {
                'name'                  : module_description['name'],
                'type'                  : module_description['name'],
                'neuron_group'          : module_description['neuron_group'],

                'include'               : module_description.pop('include', True),
                'pool_rows'             : module_description.pop('pool_rows', 1),
                'pools_positions_limits': module_description.pop('pools_positions_limits', None),
                'n_pool'                : module_description.pop('n_pool', None),
                'copies'                : module_description.pop('copies', 1),
                'plotting'              : module_description.pop('plotting', {}),

                'depth'                 : 0,
                'sub_parts_list'        : [],
            }

            for mod_type, segments, copies, sides_list, limits, pool_names in zip(
                sub_modules_names,
                sub_modules_segments,
                sub_modules_copies,
                sub_modules_sides,
                sub_modules_limits,
                sub_modules_pool_names,
            ):
                # AXIAL, LIMBS

                type_pars : dict = module_description.pop(mod_type, {})
                sub_module_description = {
                    'name'                  : mod_type,
                    'pools_names_aux'       : pool_names,

                    'include'               : type_pars.get('include', module['include']),
                    'copies'                : type_pars.get('copies',  copies),
                    'pools'                 : type_pars.get('pools', segments) * type_pars.get('copies',  copies),
                    'sides_list'            : type_pars.get('sides_list', sides_list),
                    'pool_rows'             : type_pars.get('pool_rows', 1),
                    'pools_positions_limits': type_pars.get('pools_positions_limits', limits),
                    'plotting'              : type_pars.get('plotting', {}),

                    'n_pool'                : module_description.get('n_pool'),
                    'fraction'              : module_description.get('fraction'),
                    'sub_parts_description' : module_description.get('sub_parts_description'),

                }

                sub_module = self.__define_network_subpart(module, sub_module_description)
                module['sub_parts_list'].append(sub_module)
                module[mod_type] = sub_module

            module['n']          = sum( submod['n'] for submod in module['sub_parts_list'] )
            module['pools']      = sum( submod['pools'] for submod in module['sub_parts_list'] )
            module['pools_copy'] = module['pools'] // module['copies']
            module['n_copy']     = module['n'] // module ['copies']

            n_pool_list = list(
                set(
                    [ submod['n_pool'] for submod in module['sub_parts_list'] if submod['include']]
                )
            )
            module['n_pool'] = n_pool_list[0] if n_pool_list != [] else 0

            assert len(n_pool_list) in [0,1], 'Pool size should be uniform among sub parts '

            module['pools_names'] = [
                pname for submod in module['sub_parts_list']
                for pname in submod['pools_names']
            ]
            module['pools_positions_limits'] = [
                pname for submod in module['sub_parts_list']
                for pname in submod['pools_positions_limits']
            ]

            network_modules_list.append( module )

        # Global parameters
        self.neuron_groups_ids : list[int] = list(
            set( submod['neuron_group'] for submod in network_modules_list )
        )
        self.neurons_lists : list[list[SnnNeuron]] = [ [] for _ in self.neuron_groups_ids]

        self.n_tot = [
            sum(
                submod['n']
                for submod in network_modules_list
                if submod['neuron_group'] == group_id
            )
            for group_id in self.neuron_groups_ids
        ]

        self.__prune_branches(network_modules_list)
        self.__prune_parameters(network_modules_list)
        self.__define_indices(network_modules_list)

        self.network_modules = SnnNetworkModule({'sub_parts_list' : network_modules_list}, root= True)
        self.network_leaf_modules = self.__get_leaf_modules()
        return

    # CONNECTIVITY SCHEMES
    def __update_connectivity_schemes(self):
        ''' Update the fields of the connectivity schemes based on input dictionary '''

        connectivity_and_newpars = [
            [self.__connectivity_axial, self.connectivity_axial_newpars],
            [self.__connectivity_limbs, self.connectivity_limbs_newpars],
        ]

        for net_connectivity, connectivity_newpars in connectivity_and_newpars:
            for conn_type, conn_params_list in connectivity_newpars.items():
                # New connection type
                if net_connectivity.get(conn_type) is None:
                    logging.info(f'Adding {conn_type} connections based on input parameters')
                    logging.info(f'{json.dumps(conn_params_list, indent=4)}')
                    net_connectivity[conn_type] = conn_params_list
                    continue

                existing_conn_params_names = [ cp['name'] for cp in net_connectivity[conn_type] ]

                for conn_params in conn_params_list:
                    # New connection within a connection type
                    if conn_params['name'] not in existing_conn_params_names:
                        logging.info(f"Adding {conn_params['name']} in {conn_type} connections based on input parameters")
                        logging.info(f'{json.dumps(conn_params, indent=4)}')
                        net_connectivity[conn_type].append(conn_params)
                        existing_conn_params_names.append(conn_params['name'])
                        continue

                    # Update connection within a connection type
                    logging.info(f"Updating {conn_params['name']} in {conn_type} connections based on input parameters")
                    logging.info(f'{json.dumps(conn_params, indent=4)}')
                    conn_params_index = existing_conn_params_names.index(conn_params['name'])
                    net_connectivity[conn_type][conn_params_index] = conn_params

    def __load_connectivity_schemes(self):
        ''' Loads the yaml files describing the connectivity '''

        self.__connectivity_axial : dict[str, list[dict]] = {}
        self.__connectivity_limbs : dict[str, list[dict]] = {}

        folder_path = 'network_implementations/connectivity_schemes'
        if self.network_modules['cpg']['axial'].include:
            filename = self.connectivity_axial_filename
            with open(f'{folder_path}/{filename}.yaml', encoding='UTF-8') as infile:
                self.__connectivity_axial: dict[str, list[dict]] = yaml.safe_load(infile)

        if self.network_modules['cpg']['limbs'].include:
            filename = self.connectivity_limbs_filename
            with open(f'{folder_path}/{filename}.yaml', encoding='UTF-8') as infile:
                self.__connectivity_limbs: dict[str, list[dict]] = yaml.safe_load(infile)

        self.__update_connectivity_schemes()

    # DERIVED PARAMETERS
    def __define_derived_parameters(self):
        '''
        Auxiliary parameters that are computed from the existing ones,
        which are defined at initialization
        '''

        self.__segments = self.segments_axial + self.segments_limbs

        self.__limb_pairs = self.limbs // 2
        self.__limbs_segments_y_positions = tuple(
            [
                lbpos
                for lbpos in self.limbs_y_positions
                for _ in range(self.segments_per_limb)
            ]
        )

        self.__limbs_segments_i_positions = tuple(
            [
                lbpos
                for lbpos in self.limbs_i_positions
                for _ in range(self.segments_per_limb)
            ]
        )

        self.__rows_segment       = self.network_modules['cpg'].pool_rows

        axis_length_rows          = self.rows_segment * max(self.segments_axial, self.limb_pairs )
        self.__height_segment_row = self.length_axial / axis_length_rows
        self.__height_segment     = self.height_segment_row * self.rows_segment

        # Included pools
        net_mod = self.network_modules

        self.__include_cpg_axial = net_mod['cpg']['axial'].include
        self.__include_cpg_limbs = net_mod['cpg']['limbs'].include
        self.__include_cpg       = net_mod['cpg'].include

        self.__include_reticulospinal_axial = net_mod['rs']['axial'].include
        self.__include_reticulospinal_limbs = net_mod['rs']['limbs'].include
        self.__include_reticulospinal       = net_mod['rs'].include

        self.__include_motor_neurons_axial = net_mod['mn']['axial'].include
        self.__include_motor_neurons_limbs = net_mod['mn']['limbs'].include
        self.__include_motor_neurons       = net_mod['mn'].include

        self.__include_muscle_cells_axial = net_mod['mc']['axial'].include
        self.__include_muscle_cells_limbs = net_mod['mc']['limbs'].include
        self.__include_muscle_cells       = net_mod['mc'].include

        self.__include_proprioception_axial = net_mod['ps']['axial'].include
        self.__include_proprioception_limbs = net_mod['ps']['limbs'].include
        self.__include_proprioception       = net_mod['ps'].include

        self.__include_exteroception_axial = net_mod['es']['axial'].include
        self.__include_exteroception_limbs = net_mod['es']['limbs'].include
        self.__include_exteroception       = net_mod['es'].include

    # INDEXING
    def __define_indices_group(self, modules: list[dict], ind0: int, group_id = 0):
        ''' Define limits and indices of the modules for the specified neuron group '''

        #TODO: Add indices for the copies

        for module in modules:
            if module['neuron_group'] != group_id:
                continue

            # Define limits and indices
            module['indices_limits'] = [ind0, ind0 + module['n'] - 1 ]
            module['indices'] = np.arange(ind0, ind0 + module['n'])

            # TREE LEAF
            if module['sub_parts_list'] == []:

                n_copy      = module['n_copy']
                n_side_copy = module['n_side_copy']
                n_pool_side = module['n_pool_side']

                # 4 - dim
                module['indices_pools_sides_copies'] = np.array(
                    [
                        [
                            [
                                np.arange(
                                    ind0 + n_copy*copy + n_side_copy*side + n_pool_side* seg,
                                    ind0 + n_copy*copy + n_side_copy*side + n_pool_side* (seg + 1)
                                )
                                for seg in range(module['pools_copy'])
                            ]
                            for side in range(module['sides'])
                        ]
                        for copy in range(module['copies'])
                    ]
                )

                # 3 - dim
                module['indices_sides_copies'] = np.concatenate(
                    [
                        module['indices_pools_sides_copies'][:, :, pool]
                        for pool in range(module['pools_copy'])
                    ],
                    axis=2
                )

                module['indices_pools_copies'] = np.concatenate(
                    [
                        module['indices_pools_sides_copies'][:, side]
                        for side in range(module['sides'])
                    ],
                    axis=2
                )

                module['indices_pools_sides'] = np.concatenate(
                    [
                        module['indices_pools_sides_copies'][copy]
                        for copy in range(module['copies'])
                    ],
                    axis=1
                )

                # 2 - dim
                module['indices_copies'] = np.concatenate(
                    [
                        module['indices_sides_copies'][:, side]
                        for side in range(module['sides'])
                    ],
                    axis=1
                )

                module['indices_sides'] = np.concatenate(
                    [
                        module['indices_sides_copies'][copy]
                        for copy in range(module['copies'])
                    ],
                    axis=1
                )

                module['indices_pools'] = np.concatenate(
                    [
                        module['indices_pools_copies'][copy]
                        for copy in range(module['copies'])
                    ],
                    axis=0
                )

                # Increment starting index
                ind0 += module['n']
                continue

            # SUB BRANCHES
            ind0 = self.__define_indices_group(module['sub_parts_list'], ind0, group_id)

            if module['depth'] != 0:
                # TREE BRANCH
                # Define indices from the inheriting branches (e.g. excitatory, inhibitory)

                # 4 - dim
                module['indices_pools_sides_copies'] = np.concatenate(
                    [ sub_part['indices_pools_sides_copies'] for sub_part in module['sub_parts_list'] ],
                    axis = 3
                )

                # 3 - dim
                module['indices_sides_copies'] = np.concatenate(
                    [ sub_part['indices_sides_copies'] for sub_part in module['sub_parts_list'] ],
                    axis = 2
                )
                module['indices_pools_copies'] = np.concatenate(
                    [ sub_part['indices_pools_copies'] for sub_part in module['sub_parts_list'] ],
                    axis = 2
                )
                module['indices_pools_sides'] = np.concatenate(
                    [ sub_part['indices_pools_sides'] for sub_part in module['sub_parts_list'] ],
                    axis = 2
                )

                # 2 - dim
                module['indices_copies'] = np.concatenate(
                    [ sub_part['indices_copies'] for sub_part in module['sub_parts_list'] ],
                    axis = 1
                )
                module['indices_sides'] = np.concatenate(
                    [ sub_part['indices_sides'] for sub_part in module['sub_parts_list'] ],
                    axis = 1
                )
                module['indices_pools'] = np.concatenate(
                    [ sub_part['indices_pools'] for sub_part in module['sub_parts_list'] ],
                    axis = 1
                )

            else:
                # TREE ROOT
                # Concatenate indices from the main branches (e.g. axial, limbs)

                module['indices_sides'] = np.concatenate(
                    [ submod['indices_sides'] for submod in module['sub_parts_list'] ],
                    axis= 1
                )
                module['indices_pools'] = np.concatenate(
                    [ submod['indices_pools'] for submod in module['sub_parts_list'] ],
                    axis= 0
                )
                module['indices_pools_sides'] = np.concatenate(
                    [ submod['indices_pools_sides'] for submod in module['sub_parts_list'] ],
                    axis= 1
                )

        return ind0

    def __define_indices(self, network_modules_list: list[dict]):
        ''' Define limits and indices of the modules for every neuron group '''

        # Compute indices for the sub-modules
        for group_id in self.neuron_groups_ids:
            self.__define_indices_group(network_modules_list, ind0= 0, group_id= group_id)

    # NEURONS
    def __define_neurons_list_group(self, modules: list[SnnNetworkModule], group_id: int):
        ''' Define properties of all neurons of a neuron group '''
        for module in modules:
            if module.neuron_group != group_id:
                continue

            if module.sub_parts_list == []:
                copies, sides, pools_copy, n_side_pool = module.indices_pools_sides_copies.shape

                module_neurons = [
                    SnnNeuron(
                        module   = module,
                        copy_num = copy,
                        side_num = side,
                        pool_num = pool,
                        neur_num = ind,
                    )
                    for copy in range(copies)
                    for side in range(sides)
                    for pool in range(pools_copy)
                    for ind  in range(n_side_pool)
                ]

                self.neurons_lists[group_id].extend(module_neurons)
            else:
                self.__define_neurons_list_group(module.sub_parts_list, group_id)

    def __define_neurons_lists(self):
        ''' Define properties of all neurons of all neuron groups'''

        self.neurons_side_id    = [ np.array([]) for _ in self.neuron_groups_ids]
        self.neurons_ner_id     = [ np.array([]) for _ in self.neuron_groups_ids]
        self.neurons_ner_sub_id = [ np.array([]) for _ in self.neuron_groups_ids]
        self.neurons_limb_id    = [ np.array([]) for _ in self.neuron_groups_ids]
        self.neurons_pool_id    = [ np.array([]) for _ in self.neuron_groups_ids]

        # TO BE DEFINED (network_parameters.py)
        self.neurons_y_neur     = [ np.array([]) for _ in self.neuron_groups_ids]
        self.neurons_y_mech     = [ np.array([]) for _ in self.neuron_groups_ids]

        for neuron_group in self.neuron_groups_ids:
            self.__define_neurons_list_group(self.network_modules.sub_parts_list, group_id= neuron_group)
            self.neurons_side_id   [neuron_group] = np.array([   ner.side_id for ner in self.neurons_lists[neuron_group]])
            self.neurons_ner_id    [neuron_group] = np.array([    ner.ner_id for ner in self.neurons_lists[neuron_group]])
            self.neurons_ner_sub_id[neuron_group] = np.array([ner.ner_sub_id for ner in self.neurons_lists[neuron_group]])
            self.neurons_limb_id   [neuron_group] = np.array([   ner.limb_id for ner in self.neurons_lists[neuron_group]])
            self.neurons_pool_id   [neuron_group] = np.array([   ner.pool_id for ner in self.neurons_lists[neuron_group]])

    # SAVING
    def save_yaml_files(self, destination_path):
        ''' Saves the yaml files to the destination path'''
        super().save_yaml_files(destination_path)

        source_path = 'network_implementations/connectivity_schemes'

        if self.network_modules['cpg']['axial'].include:
            filename = self.connectivity_axial_filename

            file_src = f'{source_path}/{filename}.yaml'
            file_dst = f'{destination_path}/{filename}.yaml'

            logging.info('Copying %s file to %s', file_src,  file_dst)
            shutil.copyfile(file_src, file_dst)

        if self.network_modules['cpg']['limbs'].include:
            filename = self.connectivity_limbs_filename

            file_src = f'{source_path}/{filename}.yaml'
            file_dst = f'{destination_path}/{filename}.yaml'

            logging.info('Copying %s file to %s', file_src,  file_dst)
            shutil.copyfile(file_src, file_dst)

    # PLOTTING
    def __plot_structure_group(self, modules: list[SnnNetworkModule] = None, axis: plt.Axes= None, group_id= 0):
        ''' Plots the tree structure of the network for a specific neuron group'''

        for module in modules:

            if module.neuron_group != group_id:
                continue

            randcol = np.random.rand(3)
            col = randcol # if not hasattr(module, 'plotting') else module.plotting.get('color', randcol)
            rect = Rectangle(
                xy        = (module.indices_limits[0],  module.depth),
                width     = module.n_tot,
                height    = 1,
                facecolor = col,
            )

            axis.add_patch(rect)
            axis.text(
                s        = module.type,
                x        = rect.get_x() + rect.get_width()/2.0,
                y        = rect.get_y() + rect.get_height()/2.0,
                rotation = 90.0,
                fontsize = 20,
                color    = 'black',
                weight   = 'bold',
                ha       = 'center',
                va       = 'center',
            )

            axis.plot( module.indices_limits, [module.depth, module.depth], color= col )
            self.__plot_structure_group(module.sub_parts_list, axis= axis, group_id= group_id)

        return axis

    def __plot_structure(self):
        ''' Plots the tree structure of the network for all neuron groups'''

        for group_id in self.neuron_groups_ids:
            plt.figure(f'Neuron Group {group_id}')
            axis = plt.axes()
            axis.set_title(f'Neuron Group {group_id}')
            self.__plot_structure_group(self.network_modules.sub_parts_list, axis= axis, group_id= group_id)

    def __plot_neurons(self):
        ''' Plot properties of the neurons '''

        figures = (
            (   'index [#]', [ ner.index    for ner in self.neurons_lists[0]] ),
            (  'ner_id [#]', [ ner.ner_id   for ner in self.neurons_lists[0]] ),
            ( 'side_id [#]', [ ner.side_id  for ner in self.neurons_lists[0]] ),
            (    'pool [#]', [ ner.pool_id  for ner in self.neurons_lists[0]] ),
            ( 'limb_id [#]', [ ner.limb_id  for ner in self.neurons_lists[0]] ),
        )
        n_figures = len(figures)

        plt.figure('Neurons Identifiers')
        for ind, (title, values) in enumerate(figures):
            plt.subplot(n_figures, 1, ind + 1)
            plt.title(title)
            plt.plot(values)
            plt.grid()

    def plotting(self):
        ''' Plotting'''
        self.__plot_structure()
        self.__plot_neurons()

    # TESTING
    def __check_consistency_numerosity(self, modules: dict[str, SnnNetworkModule] = None):
        ''' Checks that the sum of the sub-parts corresponds to the total '''

        if modules is None:
            modules = self.network_modules.sub_parts

        for module in modules.values():
            if module.sub_parts == {}:
                continue
            if module.n_tot != sum( submod.n_tot for submod in module.sub_parts.values() ):
                raise ValueError(
                    f"Non-matching population in {module.name} and its sub-parts"
                )
            self.__check_consistency_numerosity(module.sub_parts)
        return True

    def __check_consistency_neurons(self):
        ''' Checks that the neurons are defined in ascending index order'''
        for neurons in self.neurons_lists:
            ner_indices = [ner.index for ner in neurons]
            assert ner_indices == sorted(ner_indices), 'Neurons are not sorted'

    def __check_consistency_inclusions(self):
        ''' Consistency of included populations '''

        if not self.include_cpg_axial:
            assert not self.include_reticulospinal_axial, 'Inconsistent inclusion'
            assert not self.include_motor_neurons_axial,  'Inconsistent inclusion'
            assert not self.include_muscle_cells_axial,   'Inconsistent inclusion'
            assert not self.include_proprioception_axial, 'Inconsistent inclusion'
            assert not self.include_exteroception_axial,  'Inconsistent inclusion'

        if not self.include_cpg_limbs:
            assert not self.include_reticulospinal_limbs, 'Inconsistent inclusion'
            assert not self.include_motor_neurons_limbs,  'Inconsistent inclusion'
            assert not self.include_muscle_cells_limbs,   'Inconsistent inclusion'
            assert not self.include_proprioception_limbs, 'Inconsistent inclusion'
            assert not self.include_exteroception_limbs,  'Inconsistent inclusion'

        if not self.include_motor_neurons_axial:
            assert not self.include_muscle_cells_axial, 'Inconsistent inclusion'

        if not self.include_motor_neurons_limbs:
            assert not self.include_muscle_cells_limbs, 'Inconsistent inclusion'

        if not self.include_muscle_cells_axial:
            assert not self.include_proprioception_axial, 'Inconsistent inclusion'
            assert not self.include_exteroception_axial,  'Inconsistent inclusion'

        if not self.include_muscle_cells_limbs:
            assert not self.include_proprioception_limbs, 'Inconsistent inclusion'
            assert not self.include_exteroception_limbs,  'Inconsistent inclusion'

    def consistency_checks(self):
        ''' Tests to verify the consistency '''
        super().consistency_checks()
        self.__check_consistency_numerosity()
        self.__check_consistency_neurons()
        self.__check_consistency_inclusions()

    # PROPERTIES

    # Files
    pars_limb_connectivity_filename : str = SnnPars.read_only_attr('pars_limb_connectivity_filename')
    limb_connectivity_scheme        : str = SnnPars.read_only_attr('limb_connectivity_scheme')

    connectivity_axial_filename    : str = SnnPars.read_only_attr('connectivity_axial_filename')
    connectivity_limbs_filename    : str = SnnPars.read_only_attr('connectivity_limbs_filename')

    connectivity_axial : dict[str, list[dict]] = SnnPars.read_only_attr('connectivity_axial')
    connectivity_limbs : dict[str, list[dict]] = SnnPars.read_only_attr('connectivity_limbs')

    # Spatial arrangement
    length_axial            : float       = SnnPars.read_only_attr('length_axial')
    segments_axial          : int         = SnnPars.read_only_attr('segments_axial')
    length_segment          : float       = SnnPars.read_only_attr('length_segment')
    limbs                   : int         = SnnPars.read_only_attr('limbs')
    segments_limbs          : int         = SnnPars.read_only_attr('segments_limbs')
    segments_per_limb       : int         = SnnPars.read_only_attr('segments_per_limb')
    segments                : int         = SnnPars.read_only_attr('segments')
    limbs_f_positions       : list[float] = SnnPars.read_only_attr('limbs_f_positions')
    limbs_pairs_f_positions : list[float] = SnnPars.read_only_attr('limbs_pairs_f_positions')
    limbs_y_positions       : list[float] = SnnPars.read_only_attr('limbs_y_positions')
    limbs_pairs_y_positions : list[float] = SnnPars.read_only_attr('limbs_pairs_y_positions')
    limbs_i_positions       : list[int]   = SnnPars.read_only_attr('limbs_i_positions')
    limbs_pairs_i_positions : list[int]   = SnnPars.read_only_attr('limbs_pairs_i_positions')

    # Connectivity
    ascending_feedback_list : tuple[str] = SnnPars.read_only_attr('ascending_feedback_list')
    ascending_feedback_flag : int        = SnnPars.read_only_attr('ascending_feedback_flag')
    ascending_feedback_str  : str        = SnnPars.read_only_attr('ascending_feedback_str')
    ascfb                   : bool       = SnnPars.read_only_attr('ascfb')
    ascfb_lb                : bool       = SnnPars.read_only_attr('ascfb_lb')

    trunk_tail_discontinuity_flag : str  = SnnPars.read_only_attr('trunk_tail_discontinuity_flag')

    # Derived parameters
    limb_pairs                   : int          = SnnPars.read_only_attr('limb_pairs')
    limbs_segments_y_positions   : tuple[float] = SnnPars.read_only_attr('limbs_segments_y_positions')
    limbs_segments_i_positions   : tuple[int]   = SnnPars.read_only_attr('limbs_segments_i_positions')
    rows_segment                 : int          = SnnPars.read_only_attr('rows_segment')
    height_segment_row           : float        = SnnPars.read_only_attr('height_segment_row')
    height_segment               : float        = SnnPars.read_only_attr('height_segment')
    include_cpg_axial            : bool         = SnnPars.read_only_attr('include_cpg_axial')
    include_cpg_limbs            : bool         = SnnPars.read_only_attr('include_cpg_limbs')
    include_cpg                  : bool         = SnnPars.read_only_attr('include_cpg')
    include_reticulospinal_axial : bool         = SnnPars.read_only_attr('include_reticulospinal_axial')
    include_reticulospinal_limbs : bool         = SnnPars.read_only_attr('include_reticulospinal_limbs')
    include_reticulospinal       : bool         = SnnPars.read_only_attr('include_reticulospinal')
    include_motor_neurons_axial  : bool         = SnnPars.read_only_attr('include_motor_neurons_axial')
    include_motor_neurons_limbs  : bool         = SnnPars.read_only_attr('include_motor_neurons_limbs')
    include_motor_neurons        : bool         = SnnPars.read_only_attr('include_motor_neurons')
    include_muscle_cells_axial   : bool         = SnnPars.read_only_attr('include_muscle_cells_axial')
    include_muscle_cells_limbs   : bool         = SnnPars.read_only_attr('include_muscle_cells_limbs')
    include_muscle_cells         : bool         = SnnPars.read_only_attr('include_muscle_cells')
    include_proprioception_axial : bool         = SnnPars.read_only_attr('include_proprioception_axial')
    include_proprioception_limbs : bool         = SnnPars.read_only_attr('include_proprioception_limbs')
    include_proprioception       : bool         = SnnPars.read_only_attr('include_proprioception')
    include_exteroception_axial  : bool         = SnnPars.read_only_attr('include_exteroception_axial')
    include_exteroception_limbs  : bool         = SnnPars.read_only_attr('include_exteroception_limbs')
    include_exteroception        : bool         = SnnPars.read_only_attr('include_exteroception')

# TEST
def main(plot = True):
    ''' Test case '''
    logging.info('TEST: Topology Parameters')
    par = SnnParsTopology(parsname= 'pars_topology_test')
    if plot:
        par.plotting()
        plt.show()

    return par

if __name__ == '__main__':
    main()
