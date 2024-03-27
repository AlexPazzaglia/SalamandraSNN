''' Parameters of the mechanical model '''
import logging
import numpy as np
from network_modules.parameters.pars_utils import SnnPars

class SnnParsMechanics(SnnPars):
    ''' Defines the parameters of the mechanical model '''

    def __init__(
        self,
        parsname : str,
        new_pars : dict = None,
        pars_path: str = None,
        **kwargs
    ):
        if pars_path is None:
            pars_path = 'network_parameters/parameters_mechanics'

        super().__init__(
            pars_path = pars_path,
            parsname  = parsname,
            new_pars  = new_pars,
            pars_type = 'parameters_mechanics',
            **kwargs
        )

        # Parameters
        self.__mech_timestep     : float = self.pars.pop('mech_timestep')
        self.__mech_limbs        : int   = self.pars.pop('mech_limbs')
        self.__mech_pca_joints   : int   = self.pars.pop('mech_pca_joints')
        self.__mech_axial_joints : int   = self.pars.pop('mech_axial_joints')
        self.__mech_limbs_joints : int   = self.pars.pop('mech_limbs_joints')

        self.__mech_joints_silenced : tuple[int] = tuple( self.pars.pop('mech_joints_silenced') )

        self.__mech_lb_joints_lead : tuple[int] = tuple( self.pars.pop('mech_lb_joints_lead') )
        self.__mech_lb_joints_foll : tuple[int] = tuple( self.pars.pop('mech_lb_joints_foll') )
        self.__mech_lb_joints_free : tuple[int] = tuple( self.pars.pop('mech_lb_joints_free') )
        self.__mech_lb_joints_fixd : tuple[int] = tuple( self.pars.pop('mech_lb_joints_fixd') )

        self.__mech_lb_joints_swap : tuple[int] = tuple( self.pars.pop('mech_lb_joints_swap') )

        self.__mech_lb_pairs_gains_mean_fe_lead : np.ndarray = np.array(self.pars.pop('mech_lb_pairs_gains_mean_fe_lead'))
        self.__mech_lb_pairs_gains_mean_fe_foll : np.ndarray = np.array(self.pars.pop('mech_lb_pairs_gains_mean_fe_foll'))
        self.__mech_lb_pairs_gains_mean_fe_free : np.ndarray = np.array(self.pars.pop('mech_lb_pairs_gains_mean_fe_free'))
        self.__mech_lb_pairs_gains_asym_fe_lead : np.ndarray = np.array(self.pars.pop('mech_lb_pairs_gains_asym_fe_lead'))
        self.__mech_lb_pairs_gains_asym_fe_foll : np.ndarray = np.array(self.pars.pop('mech_lb_pairs_gains_asym_fe_foll'))
        self.__mech_lb_pairs_gains_asym_fe_free : np.ndarray = np.array(self.pars.pop('mech_lb_pairs_gains_asym_fe_free'))

        self.__mech_axial_length          : float        =  self.pars.pop('mech_axial_length')
        self.__mech_axial_joints_position : tuple[float] =  tuple( self.pars.pop('mech_axial_joints_position') )
        self.__mech_limbs_joints_position : tuple[float] =  tuple( self.pars.pop('mech_limbs_joints_position') )

        self.__lb_offset_swim : tuple[tuple[float]] =  tuple( self.pars.pop('lb_offset_swim') )
        self.__lb_offset_walk : tuple[tuple[float]] =  tuple( self.pars.pop('lb_offset_walk') )

        self.__mech_activation_delays : np.ndarray = np.array(self.pars.pop('mech_activation_delays'))

        # Derived parameters
        self.derived_parameters()

        # Consistency checks
        self.consistency_checks()

    def derived_parameters(self):
        ''' Derived parameters '''

        self.__mech_limbs_pairs = self.mech_limbs // 2
        self.__mech_n_lb_joints = self.mech_limbs_joints // self.mech_limbs if self.mech_limbs else 0

        self.__mech_n_lb_joints_lead = len(self.mech_lb_joints_lead)
        self.__mech_n_lb_joints_free = len(self.mech_lb_joints_free)
        self.__mech_n_lb_joints_foll = len(self.mech_lb_joints_foll)
        self.__mech_n_lb_joints_fixd = len(self.mech_lb_joints_fixd)

        self.__mech_n_lb_joints_act = self.mech_n_lb_joints_lead + self.mech_n_lb_joints_free
        self.__mech_n_lb_joints_mov = self.mech_n_lb_joints_act  + self.mech_n_lb_joints_foll

        # Geometry
        self.__mech_axial_links_length      = np.zeros(self.mech_axial_joints+1)
        self.__mech_axial_links_length[0]   = self.mech_axial_joints_position[0]
        self.__mech_axial_links_length[1:-1] = np.diff(self.mech_axial_joints_position)
        self.__mech_axial_links_length[-1]  = self.mech_axial_length - self.mech_axial_joints_position[-1]

        self.__mech_limbs_pairs_positions = self.mech_limbs_joints_position[::2]

        self.__mech_limbs_joints_indices = np.array(
            [
                np.argmin(
                    np.abs(
                        self.mech_limbs_joints_position[lb] -
                        np.array(self.mech_axial_joints_position)
                    )
                )
                for lb in range(self.mech_limbs)
            ]
        )
        self.__mech_limbs_pairs_indices = self.mech_limbs_joints_indices[::2]

        # DOFS
        n_act = self.mech_n_lb_joints_act

        # silenced dofs
        self.__mech_pools_silenced = np.array(
            [
                round(2 * dof + side)
                for dof in self.mech_joints_silenced
                for side in range(2)
            ],
            dtype = int,
        )

        # LEADER DOF
        self.__mech_lb_pools_lead = np.array(
            [
                (
                    2 * dof + side
                    if dof not in self.mech_lb_joints_swap
                    else
                    2 * dof + (1 - side)
                )
                for dof in self.mech_lb_joints_lead
                for side in range(2)
            ]
        )
        self.__mc_lb_ind_lead = np.concatenate(
            [
                np.arange(self.mech_n_lb_joints_lead),
                np.arange(self.mech_n_lb_joints_lead) + n_act,
            ]
        )
        # FREELY MOVING DOFS
        self.__mech_lb_pools_free = np.array(
            [
                (
                    2 * dof + side
                    if dof not in self.mech_lb_joints_swap
                    else
                    2 * dof + (1 - side)
                )
                for dof in self.mech_lb_joints_free
                for side in range(2)
            ]
        )
        self.__mc_lb_ind_free = np.concatenate(
            [
                np.arange(self.mech_n_lb_joints_lead, n_act),
                np.arange(self.mech_n_lb_joints_lead, n_act) + n_act,
            ]
        )
        # FOLLOWING DOFS
        self.__mech_lb_pools_foll = np.array(
            [
                (
                    2 * dof + side
                    if dof not in self.mech_lb_joints_swap
                    else
                    2 * dof + (1 - side)
                )
                for dof in self.mech_lb_joints_foll
                for side in range(2)
            ]
        )
        self.__mc_lb_ind_foll = np.concatenate(
            [
                np.arange(self.mech_n_lb_joints_lead),
                np.arange(self.mech_n_lb_joints_lead) + n_act,
            ]
        )

        # MOTOR OFFSETS
        self.__mech_motor_offset_swimming = np.concatenate(
            [
                np.zeros( self.mech_axial_joints ),
                np.array(
                    [
                        self.lb_offset_swim[lb_dof][lb_pair]
                        for lb_pair in range(self.mech_limbs_pairs)
                        for lb_side in range(2)
                        for lb_dof  in range(self.mech_n_lb_joints)
                    ]
                ),
            ]
        )

        self.__mech_motor_offset_walking  = np.concatenate(
            [
                np.zeros( self.mech_axial_joints ),
                np.array(
                    [
                        self.lb_offset_walk[lb_dof][lb_pair]
                        for lb_pair in range(self.mech_limbs_pairs)
                        for lb_side in range(2)
                        for lb_dof  in range(self.mech_n_lb_joints)
                    ]
                ),
            ]
        )

        # GAINS
        self.get_limbs_gains()

    def get_limbs_gains(self):
        ''' Compute gains for flexors and extensors of all the DOFs of all the limbs'''

        # AUXILIARY FUNCTION
        def _get_dof_type_gains(
            gains_mean: np.ndarray,
            gains_asym: np.ndarray,
            n_joints  : int,
            dof_name  : str,
        ) -> tuple[np.ndarray, np.ndarray]:
            ''' Get gains for a given dof type '''

            if not n_joints:
                return np.array([]), np.array([]), np.array([])

            limb_pairs = self.mech_limbs_pairs
            assert gains_mean.shape == (limb_pairs, n_joints), \
                f'gains_{dof_name} should have shape ({limb_pairs}, {n_joints})'
            assert gains_asym.shape == (limb_pairs, n_joints), \
                f'gains_{dof_name} should have shape ({limb_pairs}, {n_joints})'

            flx_gains = np.array(
                [
                    [
                        gains_mean[lb_pair][dof] * (1 - gains_asym[lb_pair][dof]/2)
                        for dof in range(n_joints)
                    ]
                    for lb_pair in range(limb_pairs)
                    for lb_side in range(2)
                ]
            )
            ext_gains = np.array(
                [
                    [
                        gains_mean[lb_pair][dof] * (1 + gains_asym[lb_pair][dof]/2)
                        for dof in range(n_joints)
                    ]
                    for lb_pair in range(limb_pairs)
                    for lb_side in range(2)
                ]
            )

            limbs_gains = np.array(
                [
                    [
                        gain
                        for dof in range(n_joints)
                        for gain in [flx_gains[lb_ind][dof], ext_gains[lb_ind][dof]]
                    ]
                    for lb_ind in range(limb_pairs * 2)
                ]
            )
            return flx_gains, ext_gains, limbs_gains

        # LEADER
        (
            self.__mc_limb_flx_gains_lead,
            self.__mc_limb_ext_gains_lead,
            self.__mc_limbs_gains_lead,
        ) = _get_dof_type_gains(
            gains_mean = self.mech_lb_pairs_gains_mean_fe_lead,
            gains_asym = self.mech_lb_pairs_gains_asym_fe_lead,
            n_joints   = self.mech_n_lb_joints_lead,
            dof_name   = 'lead',
        )
        # FOLLOWER
        (
            self.__mc_limb_flx_gains_foll,
            self.__mc_limb_ext_gains_foll,
            self.__mc_limbs_gains_foll,
        ) = _get_dof_type_gains(
            gains_mean = self.mech_lb_pairs_gains_mean_fe_foll,
            gains_asym = self.mech_lb_pairs_gains_asym_fe_foll,
            n_joints   = self.mech_n_lb_joints_foll,
            dof_name   = 'foll',
        )
        # FREE
        (
            self.__mc_limb_flx_gains_free,
            self.__mc_limb_ext_gains_free,
            self.__mc_limbs_gains_free,
        ) = _get_dof_type_gains(
            gains_mean = self.mech_lb_pairs_gains_mean_fe_free,
            gains_asym = self.mech_lb_pairs_gains_asym_fe_free,
            n_joints   = self.mech_n_lb_joints_free,
            dof_name   = 'free',
        )
        return

    ## PROPERTIES
    mech_timestep              : float               = SnnPars.read_only_attr('mech_timestep')
    mech_limbs                 : int                 = SnnPars.read_only_attr('mech_limbs')
    mech_pca_joints            : int                 = SnnPars.read_only_attr('mech_pca_joints')
    mech_axial_joints          : int                 = SnnPars.read_only_attr('mech_axial_joints')
    mech_limbs_joints          : int                 = SnnPars.read_only_attr('mech_limbs_joints')
    mech_joints_silenced       : tuple[int]          = SnnPars.read_only_attr('mech_joints_silenced')
    mech_lb_joints_lead        : tuple[int]          = SnnPars.read_only_attr('mech_lb_joints_lead')
    mech_lb_joints_foll        : tuple[int]          = SnnPars.read_only_attr('mech_lb_joints_foll')
    mech_lb_joints_free        : tuple[int]          = SnnPars.read_only_attr('mech_lb_joints_free')
    mech_lb_joints_fixd        : tuple[int]          = SnnPars.read_only_attr('mech_lb_joints_fixd')
    mech_lb_joints_swap        : tuple[int]          = SnnPars.read_only_attr('mech_lb_joints_swap')
    mech_axial_length          : float               = SnnPars.read_only_attr('mech_axial_length')
    mech_axial_joints_position : tuple[float]        = SnnPars.read_only_attr('mech_axial_joints_position')
    mech_axial_links_length    : np.ndarray          = SnnPars.read_only_attr('mech_axial_links_length')
    mech_limbs_joints_position : tuple[float]        = SnnPars.read_only_attr('mech_limbs_joints_position')
    mech_limbs_joints_indices  : np.ndarray          = SnnPars.read_only_attr('mech_limbs_joints_indices')
    mech_limbs_pairs_positions : tuple[float]        = SnnPars.read_only_attr('mech_limbs_pairs_positions')
    mech_limbs_pairs_indices   : np.ndarray          = SnnPars.read_only_attr('mech_limbs_pairs_indices')
    mech_limbs_pairs           : int                 = SnnPars.read_only_attr('mech_limbs_pairs')
    mech_n_lb_joints           : int                 = SnnPars.read_only_attr('mech_n_lb_joints')
    mech_n_lb_joints_lead      : int                 = SnnPars.read_only_attr('mech_n_lb_joints_lead')
    mech_n_lb_joints_free      : int                 = SnnPars.read_only_attr('mech_n_lb_joints_free')
    mech_n_lb_joints_foll      : int                 = SnnPars.read_only_attr('mech_n_lb_joints_foll')
    mech_n_lb_joints_fixd      : int                 = SnnPars.read_only_attr('mech_n_lb_joints_fixd')
    mech_n_lb_joints_act       : int                 = SnnPars.read_only_attr('mech_n_lb_joints_act')
    mech_n_lb_joints_mov       : int                 = SnnPars.read_only_attr('mech_n_lb_joints_mov')
    mech_pools_silenced        : tuple[int]          = SnnPars.read_only_attr('mech_pools_silenced')
    mech_lb_pools_lead         : tuple[int]          = SnnPars.read_only_attr('mech_lb_pools_lead')
    mech_lb_pools_free         : tuple[int]          = SnnPars.read_only_attr('mech_lb_pools_free')
    mech_lb_pools_foll         : tuple[int]          = SnnPars.read_only_attr('mech_lb_pools_foll')
    mc_lb_ind_lead             : np.ndarray          = SnnPars.read_only_attr('mc_lb_ind_lead')
    mc_lb_ind_free             : np.ndarray          = SnnPars.read_only_attr('mc_lb_ind_free')
    mc_lb_ind_foll             : np.ndarray          = SnnPars.read_only_attr('mc_lb_ind_foll')
    lb_offset_swim             : tuple[tuple[float]] = SnnPars.read_only_attr('lb_offset_swim')
    lb_offset_walk             : tuple[tuple[float]] = SnnPars.read_only_attr('lb_offset_walk')
    mech_motor_offset_swimming : np.ndarray          = SnnPars.read_only_attr('mech_motor_offset_swimming')
    mech_motor_offset_walking  : np.ndarray          = SnnPars.read_only_attr('mech_motor_offset_walking')

    def __update_delays(self, attr, value):
        ''' Check dimensions of the limb gains'''
        assert len(value) == self.mech_n_lb_joints_foll, \
            f'Number of delays must match number of follower joints'

    mech_activation_delays     : np.ndarray = SnnPars.read_write_attr('mech_activation_delays', fun= __update_delays)

    def __update_gains(self, attr, value):
        ''' Update values of the limb gains'''
        self.get_limbs_gains()

    # Mean and Asymmetry of the limb gains
    mech_lb_pairs_gains_mean_fe_lead : np.ndarray = SnnPars.read_write_attr('mech_lb_pairs_gains_mean_fe_lead', fun= __update_gains)
    mech_lb_pairs_gains_mean_fe_foll : np.ndarray = SnnPars.read_write_attr('mech_lb_pairs_gains_mean_fe_foll', fun= __update_gains)
    mech_lb_pairs_gains_mean_fe_free : np.ndarray = SnnPars.read_write_attr('mech_lb_pairs_gains_mean_fe_free', fun= __update_gains)
    mech_lb_pairs_gains_asym_fe_lead : np.ndarray = SnnPars.read_write_attr('mech_lb_pairs_gains_asym_fe_lead', fun= __update_gains)
    mech_lb_pairs_gains_asym_fe_foll : np.ndarray = SnnPars.read_write_attr('mech_lb_pairs_gains_asym_fe_foll', fun= __update_gains)
    mech_lb_pairs_gains_asym_fe_free : np.ndarray = SnnPars.read_write_attr('mech_lb_pairs_gains_asym_fe_free', fun= __update_gains)

    # Flexor and Extensor gains for each limb
    mc_limb_flx_gains_lead : np.ndarray = SnnPars.read_only_attr('mc_limb_flx_gains_lead')
    mc_limb_ext_gains_lead : np.ndarray = SnnPars.read_only_attr('mc_limb_ext_gains_lead')
    mc_limb_flx_gains_foll : np.ndarray = SnnPars.read_only_attr('mc_limb_flx_gains_foll')
    mc_limb_ext_gains_foll : np.ndarray = SnnPars.read_only_attr('mc_limb_ext_gains_foll')
    mc_limb_flx_gains_free : np.ndarray = SnnPars.read_only_attr('mc_limb_flx_gains_free')
    mc_limb_ext_gains_free : np.ndarray = SnnPars.read_only_attr('mc_limb_ext_gains_free')

    # Combined gains for each limb
    mc_limbs_gains_lead : np.ndarray = SnnPars.read_only_attr('mc_limbs_gains_lead')
    mc_limbs_gains_foll : np.ndarray = SnnPars.read_only_attr('mc_limbs_gains_foll')
    mc_limbs_gains_free : np.ndarray = SnnPars.read_only_attr('mc_limbs_gains_free')

# TEST
def main():
    ''' Return test parameters '''
    logging.info('TEST: Drive Parameters')
    pars = SnnParsMechanics(
        parsname= 'pars_mechanics_test',
    )
    return pars

if __name__ == '__main__':
    main()
