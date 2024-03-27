import numpy as np

from network_modules.parameters.network_module import SnnNetworkModule

class SnnNeuron():
    '''
    Descriptor of the properties of a neuron of the network

    SIDE_ID
    'ax' : 1,             # Axial
    'lbl': 2 + 2 * K,     # Left  hemisegment, Kth limb segment
    'lbr': 2 + 2 * K + 1, # Right hemisegment, Kth limb segment

    NER_ID
    'ex': 0,  # CPG excitatory
    'in': 1,  # CPG inhibitory
    'mn': 2,  # motor neuron
    'rs': 3,  # reticulospinal
    'ps': 4,  # propriosensory neuron
    'es': 5,  # exterosensory neuron
    'mc': 6,  # muscle cell
    '''

    neuron_identifiers = {
        'ex': 0,  # cpg excitatory
        'in': 1,  # cpg inhibitory
        'mn': 2,  # motor neuron
        'rs': 3,  # reticulospinal
        'ps': 4,  # propriosensory neuron
        'es': 5,  # exterosensory neuron
        'mc': 6,  # muscle cell
    }

    neuron_sub_identifiers = {
        'any': 0,
        # EX
        'V2a': 0,
        'V0v': 1,
        # IN
        'V1' : 0,
        'V0d': 1,
        'dI6': 2,
    }


    def __init__(
        self,
        module  : SnnNetworkModule,
        copy_num: int,
        side_num: int,
        pool_num: int,
        neur_num: int,
    ):

        self.module       = module
        self.pool_id      = copy_num * module.pools_copy + pool_num
        self.neuron_group = module.neuron_group
        self.copy         = copy_num
        self.index        = module.indices_pools_sides_copies[copy_num, side_num, pool_num, neur_num]
        self.neuron_name  = module.name
        self.neuron_type  = module.type
        self.side_str     = module.sides_list[side_num]

        self.index_pool_side_copy = [copy_num, side_num, pool_num, neur_num]

        # TO BE ASSIGNED (network_parameters.py)
        self.position_neur = np.NaN
        self.position_mech = np.NaN

        # Ex: cpg.limbs.ex.V2a
        name_parts_list = self.neuron_name.split('.')
        sub_network_str = name_parts_list[1]  # Axial, Limbs
        ner_type_str    = name_parts_list[2]  # Ex, In
        ner_subtype_str = name_parts_list[3] if len(name_parts_list) > 3 else 'any' # V2a, V0v

        # Ner_id
        self.ner_id     = SnnNeuron.neuron_identifiers[ner_type_str]
        self.ner_sub_id = SnnNeuron.neuron_sub_identifiers[ner_subtype_str]

        # Side_id and Limb_id
        axial_neuron = 'axial' == sub_network_str
        limbs_neuron = 'limbs' == sub_network_str

        if not axial_neuron and not limbs_neuron:
            raise ValueError(f'NER_ID not specified for {self.neuron_name}')

        if axial_neuron:
            self.limb_id = 0
            self.side_id = -1 if self.side_str == 'l' else + 1

        if limbs_neuron:
            self.limb_id = copy_num + 1

            multip_side = -1 * ( 1 - copy_num % 2 ) + 1 * ( copy_num % 2 )
            offset_dof  = +2 if self.side_str == 'f' else +3

            self.side_id = multip_side * ( 2 * ( pool_num // 2 )  + offset_dof )
