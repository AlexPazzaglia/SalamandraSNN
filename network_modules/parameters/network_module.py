from functools import reduce
import operator

class SnnNetworkModule():
    '''Descriptor of the properties of a module of the network '''

    def __getitem__(self, key: str):

        item = self.sub_parts.get(key, None)

        if item is not None:
            return item

        default_modules = ['cpg', 'rs', 'mn', 'mc', 'ps', 'es']
        default_parts   = ['axial', 'limbs']
        default_types   = ['ex', 'in', 'mn', 'mc', 'ps', 'es']

        if key in default_modules + default_parts + default_types:
            return DEFAULT_MODULE

        raise ValueError(f'{key} is not a sub-part of {self.name} nor a standard sub-part name')

    def __init__(self, mod_description: dict, root = False):

        if root:
            self._add_sub_parts(mod_description['sub_parts_list'])
            return

        # Identifier
        self.name         : str  = mod_description['name']
        self.include      : bool = mod_description['include']
        self.neuron_group : int  = mod_description['neuron_group']
        self.type         : str  = mod_description['type']
        self.depth        : int  = mod_description['depth']

        # Internal organization
        self.pools     : int = mod_description['pools']
        self.pool_rows : int = mod_description['pool_rows']

        # Numerosity
        self.n_tot       : int = mod_description['n']
        self.n_pool      : int = mod_description['n_pool']

        # Pools
        self.pools_names            : list[str]         = mod_description['pools_names']
        self.pools_positions_limits : list[list[float]] = mod_description['pools_positions_limits']

        # Plotting
        self.plotting   : dict[str, str] = mod_description['plotting']

        # Indexing
        list1d = list[int]
        list2d = list[list[int]]
        list3d = list[list[list[int]]]
        list4d = list[list[list[list[int]]]]

        self.indices                    : list1d = mod_description['indices']
        self.indices_limits             : list1d = mod_description['indices_limits']

        if self.depth is None or self.depth > 0:
            # Indexing
            self.indices_copies             : list2d = mod_description['indices_copies']
            self.indices_sides              : list2d = mod_description['indices_sides']
            self.indices_pools              : list2d = mod_description['indices_pools']
            self.indices_sides_copies       : list3d = mod_description['indices_sides_copies']
            self.indices_pools_copies       : list3d = mod_description['indices_pools_copies']
            self.indices_pools_sides        : list3d = mod_description['indices_pools_sides']
            self.indices_pools_sides_copies : list4d = mod_description['indices_pools_sides_copies']

            # Internal organization
            self.copies      : int       = mod_description['copies']
            self.pools_copy  : int       = mod_description['pools_copy']
            self.sides_list  : list[str] = mod_description['sides_list']
            self.sides       : int       = mod_description['sides']
            self.n_copy      : int       = mod_description['n_copy']
            self.n_side      : int       = mod_description['n_side']
            self.n_copy_side : int       = mod_description['n_side_copy']
            self.n_pool_side : int       = mod_description['n_pool_side']

        self._add_sub_parts(mod_description['sub_parts_list'])

    def _add_sub_parts(self, sub_parts_list):
        ''' Add sub parts of the network '''
            # Sub populations

        self.sub_parts_names        : list[str]                   = []
        self.sub_parts_list         : list[SnnNetworkModule]      = []
        self.sub_parts              : dict[str, SnnNetworkModule] = {}

        for sub_part in sub_parts_list:
            sub_part_object = SnnNetworkModule(sub_part)
            self.sub_parts_names.append(sub_part['type'])
            self.sub_parts_list.append(sub_part_object)
            self.sub_parts[sub_part['type']] = sub_part_object
            setattr(self, sub_part['type'], sub_part_object)

    def get_sub_module_from_full_name(self, sub_module_name: str):
        ''' Returns sub module from its complete name '''
        target_sub_module = reduce(
            operator.getitem,
            sub_module_name.split('.'),
            self,
        )
        return target_sub_module

# Empty module
class EmptyIndices(list):
    def __getitem__(self, key):
        return []

DEFAULT_MODULE =  SnnNetworkModule(
    mod_description = {
        'name'                      : 'default',
        'type'                      : 'default',
        'include'                   : False,
        'neuron_group'              : None,
        'depth'                     : None,
        'pools'                     : 0,
        'pool_rows'                 : 0,
        'n'                         : 0,
        'n_pool'                    : 0,
        'pools_names'               : [],
        'pools_positions_limits'    : [],
        'plotting'                  : {},
        'indices'                   : EmptyIndices(),
        'indices_limits'            : EmptyIndices(),
        'indices_copies'            : EmptyIndices(),
        'indices_sides'             : EmptyIndices(),
        'indices_pools'             : EmptyIndices(),
        'indices_sides_copies'      : EmptyIndices(),
        'indices_pools_copies'      : EmptyIndices(),
        'indices_pools_sides'       : EmptyIndices(),
        'indices_pools_sides_copies': EmptyIndices(),
        'copies'                    : 0,
        'pools_copy'                : 0,
        'sides_list'                : [],
        'sides'                     : 0,
        'n_copy'                    : 0,
        'n_side'                    : 0,
        'n_side_copy'               : 0,
        'n_pool_side'               : 0,
        'sub_parts_list'            : [],
    }
)
