import copy
import logging
import numpy as np

from typing import Union
from brian2 import second

from network_modules.parameters.pars_utils import SnnPars

class SnnParsSimulation(SnnPars):
    ''' Parameters for the simulation '''

    def __init__(
        self,
        parsname    : str,
        results_path: str,
        new_pars    : dict = None,
        pars_path   : str = None,
        **kwargs
    ):
        if pars_path is None:
            pars_path = 'network_parameters/parameters_simulation'

        super().__init__(
            pars_path = pars_path,
            parsname  = parsname,
            new_pars  = new_pars,
            pars_type = 'parameters_simulation',
            **kwargs
        )

        self.__sim_pars__ : dict = copy.deepcopy(self.pars)

        # Simulation data path, tags
        self.__results_path              : str = results_path
        self.__netname                   : str = self.pars.pop('netname')
        self.__animal_model              : str = self.pars.pop('animal_model')
        self.__data_file_tag             : str = self.pars.pop('simulation_data_file_tag')
        self.__tag_folder                : str = self.params_to_update.pop('tag_folder', '0')
        self.__tag_process               : str = self.params_to_update.pop('tag_process', '0')
        self.__tag_sub_process           : str = self.params_to_update.pop('tag_sub_process', None)
        self.__tag_run                   : str = self.params_to_update.pop('tag_run', '0')
        self.__connectivity_indices_file : str = self.params_to_update.pop('connectivity_indices_file', None)

        self.__load_connectivity_indices : bool = self.pars.pop('load_connectivity_indices')

        self.__define_results_files_names_process()
        self.__define_results_files_names_sub_process()
        self.__define_results_files_names_run()

        # Include units
        self.pars['timestep']          = float(self.pars['timestep'])          * second
        self.pars['duration']          = float(self.pars['duration'])          * second
        self.pars['callback_dt']       = float(self.pars['callback_dt'])       * second
        self.pars['online_act_window'] = float(self.pars['online_act_window']) * second

        # SIMULATION
        self.__int_method :   str = self.pars.pop('int_method')
        self.__timestep   : float = self.pars.pop('timestep')
        self.__set_seed   :  bool = self.pars.pop('set_seed')
        self.__seed_value :   int = self.pars.pop('seed_value')
        self.__duration   : float = self.pars.pop('duration')

        # Functionalities
        self.__silencing         : bool = self.pars.pop('silencing')
        self.__noise_term        : bool = self.pars.pop('noise_term')
        self.__synaptic_weighting: bool = self.pars.pop('synaptic_weighting')

        self.__include_callback : bool  = self.pars.pop('include_callback')
        self.__callback_dt      : float = self.pars.pop('callback_dt')
        self.__callback_steps   : int   = round( float( self.duration // self.callback_dt) ) + 1

        self.__save_mech_logs   : bool = self.pars.pop('save_mech_logs')

        self.__include_online_act : bool  = self.pars.pop('include_online_act')
        self.__online_act_window  : float = self.pars.pop('online_act_window')
        self.__online_threshold1  : float = 0.5
        self.__online_threshold2  : float = 1.0

        # Parameter files
        self.__pars_topology_filename : str = self.pars.pop('pars_topology_filename')
        self.__pars_neurons_filename  : str = self.pars.pop('pars_neurons_filename')
        self.__pars_synapses_filename : str = self.pars.pop('pars_synapses_filename')
        self.__pars_drive_filename    : str = self.pars.pop('pars_drive_filename')
        self.__pars_mech_filename     : str = self.pars.pop('pars_mech_filename')
        self.__pars_monitor_filename  : str = self.pars.pop('pars_monitor_filename')

        # Metrics
        self.__compute_metrics   : bool = self.pars.pop('compute_metrics')
        self.__metrics_filtering : bool = self.pars.pop('metrics_filtering')
        self.__metrics_trunk_only: bool = self.pars.pop('metrics_trunk_only')

        # Diagnostics
        self.__verboserun        : bool = self.pars.pop('verboserun')
        self.__brian_profiling   : bool = self.pars.pop('brian_profiling')

        # PROPERTIES
        # Gait pattern
        self.__gaits    : tuple[str] = tuple(self.pars.pop('gaits'))
        self.__gaitflag :        int = self.pars.pop('gaitflag')
        self.__gait     :        str = self.gaits[self.gaitflag]

        # Network excitation
        self.__stim_a_mul: Union[float, np.ndarray] = self.pars.pop('stim_a_mul')
        self.__stim_a_off: Union[float, np.ndarray] = self.pars.pop('stim_a_off')
        self.__stim_l_mul: Union[float, np.ndarray] = self.pars.pop('stim_l_mul')
        self.__stim_l_off: Union[float, np.ndarray] = self.pars.pop('stim_l_off')

        # Turning
        self.__stim_lr_asym:  float = self.pars.pop('stim_lr_asym')
        self.__stim_lr_off :  float = self.pars.pop('stim_lr_off')
        self.__stim_l_multiplier = 1 + self.stim_lr_asym/2
        self.__stim_r_multiplier = 1 - self.stim_lr_asym/2

        # Duty cycle
        self.__stim_fe_asym:  float = self.pars.pop('stim_fe_asym')
        self.__stim_fe_off :  float = self.pars.pop('stim_fe_off')
        self.__stim_f_multiplier = 1 + self.stim_fe_asym/2
        self.__stim_e_multiplier = 1 - self.stim_fe_asym/2

        self.__stim_f_mn_off : float = self.pars.pop('stim_f_mn_off')
        self.__stim_e_mn_off : float = self.pars.pop('stim_e_mn_off')

        # Trailing oscillator hypothesis
        self.__gains_drives_axis  : np.ndarray = np.array( self.pars.pop('gains_drives_axis') )
        self.__gains_drives_limbs : np.ndarray = np.array( self.pars.pop('gains_drives_limbs') )

        # Gains
        self.__mc_gain_axial: Union[float, np.ndarray] = self.pars.pop('mc_gain_axial')
        self.__mc_gain_limbs: Union[float, np.ndarray] = self.pars.pop('mc_gain_limbs')
        self.__ps_gain_axial: Union[float, np.ndarray] = self.pars.pop('ps_gain_axial')
        self.__ps_gain_limbs: Union[float, np.ndarray] = self.pars.pop('ps_gain_limbs')
        self.__es_gain_axial: Union[float, np.ndarray] = self.pars.pop('es_gain_axial')
        self.__es_gain_limbs: Union[float, np.ndarray] = self.pars.pop('es_gain_limbs')

        self.__mo_cocontraction_gain: Union[float, np.ndarray] = self.pars.pop('mo_cocontraction_gain')
        self.__mo_cocontraction_off : Union[float, np.ndarray] = self.pars.pop('mo_cocontraction_off')

        # Parameters for inter-limb coordination
        self.syn_inds_shuffled              : dict[str, np.ndarray] = {}
        self.interlimb_syn_inds_ex_shuffled : dict[str, list[np.ndarray]] = {}
        self.interlimb_syn_inds_in_shuffled : dict[str, list[np.ndarray]] = {}

        self.consistency_checks()

    # FILE NAMES
    def __define_results_files_names_process(self):
        ''' Defines names of the destination files and folders of the process '''

        folder_name_aux = f'{self.netname}_{self.data_file_tag}_{self.tag_folder}'
        saving_folder_process_aux = f'{folder_name_aux}/process_{self.tag_process}'

        self.logging_data_folder_process = f'''{self.results_path}/logs/{folder_name_aux}'''
        self.results_data_folder_process = f'''{self.results_path}/data/{saving_folder_process_aux}'''
        self.figures_data_folder_process = f'''{self.results_path}/images/{saving_folder_process_aux}'''

        # Connectivity matrices
        if self.connectivity_indices_file is None:
            self.__connectivity_indices_file : str = f'{self.results_data_folder_process}/snn_connectivity_indices.dill'

    def __define_results_files_names_run(self):
        ''' Defines names of the destination files and folders of the run '''

        saving_folder_run_aux = (
            f'{self.netname}_{self.data_file_tag}_{self.tag_folder}/'
            f'process_{self.tag_process}/'
            f'run_{self.tag_run}'
        )
        self.results_data_folder_run = f'''{self.results_path}/data/{saving_folder_run_aux}'''
        self.figures_data_folder_run = f'''{self.results_path}/images/{saving_folder_run_aux}'''

    def __define_results_files_names_sub_process(self):
        '''
        Changes file names to account for an additional sub-folder
        for the specified sub process index
        '''

        if self.tag_sub_process is None:
            self.results_data_folder_sub_process = self.results_data_folder_process
            self.figures_data_folder_sub_process = self.figures_data_folder_process
            return

        saving_folder_sub_process_aux = (
            f'{self.netname}_{self.data_file_tag}_{self.tag_folder}/'
            f'process_{self.tag_process}/'
            f'sub_process_{self.tag_sub_process}'
        )
        self.results_data_folder_sub_process = f'''{self.results_path}/data/{saving_folder_sub_process_aux}'''
        self.figures_data_folder_sub_process = f'''{self.results_path}/images/{saving_folder_sub_process_aux}'''

        saving_folder_run_aux = (f'{saving_folder_sub_process_aux}/run_{self.tag_run}')

        self.results_data_folder_run = f'''{self.results_path}/data/{saving_folder_run_aux}'''
        self.figures_data_folder_run = f'''{self.results_path}/images/{saving_folder_run_aux}'''

    # GET PARAMETERS
    def get_simulation_parameters(self) -> dict:
        ''' Return the minimum set of parameters to run the simulation '''
        return self.__sim_pars__

    ## PROPERTIES
    def __update_sim_pars(self, attr, value):
        ''' Update __sim_pars__'''
        self.__sim_pars__[attr] = value

    # Simulation
    duration         : float = SnnPars.read_write_attr('duration')

    int_method       : str   = SnnPars.read_only_attr('int_method')
    timestep         : float = SnnPars.read_only_attr('timestep')
    set_seed         : bool  = SnnPars.read_only_attr('set_seed')
    seed_value       : int   = SnnPars.read_only_attr('seed_value')

    # Functionalities
    include_silencing  : bool  = SnnPars.read_only_attr('silencing')
    include_noise_term : bool  = SnnPars.read_only_attr('noise_term')
    include_syn_weight : bool  = SnnPars.read_only_attr('synaptic_weighting')
    include_callback   : bool  = SnnPars.read_only_attr('include_callback')
    callback_dt        : float = SnnPars.read_only_attr('callback_dt')
    callback_steps     : int   = SnnPars.read_only_attr('callback_steps')

    save_mech_logs     : bool  = SnnPars.read_only_attr('save_mech_logs')

    include_online_act : bool  = SnnPars.read_write_attr('include_online_act')
    online_act_window  : float = SnnPars.read_write_attr('online_act_window')
    online_threshold1  : float = SnnPars.read_write_attr('online_threshold1')
    online_threshold2  : float = SnnPars.read_write_attr('online_threshold2')

    # Parameter files
    load_connectivity_indices : bool = SnnPars.read_write_attr('load_connectivity_indices')

    pars_topology_filename   : str  = SnnPars.read_only_attr('pars_topology_filename')
    pars_neurons_filename    : str  = SnnPars.read_only_attr('pars_neurons_filename')
    pars_synapses_filename   : str  = SnnPars.read_only_attr('pars_synapses_filename')
    pars_drive_filename      : str  = SnnPars.read_only_attr('pars_drive_filename')
    pars_mech_filename       : str  = SnnPars.read_only_attr('pars_mech_filename')
    pars_monitor_filename    : str  = SnnPars.read_only_attr('pars_monitor_filename')

    # Metrics
    compute_metrics   : bool = SnnPars.read_write_attr('compute_metrics')
    metrics_filtering : bool = SnnPars.read_write_attr('metrics_filtering')
    metrics_trunk_only: bool = SnnPars.read_write_attr('metrics_trunk_only')

    # Diagnostics
    verboserun        : bool = SnnPars.read_write_attr('verboserun')
    brian_profiling   : bool = SnnPars.read_write_attr('brian_profiling')

    # NAMING
    def __update_files(self, attr, value):
        ''' Additional update of file names '''
        self.__update_sim_pars(attr, value)
        self.__define_results_files_names_run()

    def __update_sub_process(self, attr, value):
        ''' Update file names for the sub_process tag '''
        self.__update_sim_pars(attr, value)
        self.__define_results_files_names_sub_process()

    results_path              : str = SnnPars.read_only_attr('results_path')
    netname                   : str = SnnPars.read_only_attr('netname')
    animal_model              : str = SnnPars.read_only_attr('animal_model')
    data_file_tag             : str = SnnPars.read_only_attr('data_file_tag')
    tag_folder                : str = SnnPars.read_only_attr('tag_folder')
    tag_process               : str = SnnPars.read_only_attr('tag_process')
    connectivity_indices_file : str = SnnPars.read_only_attr('connectivity_indices_file')

    tag_run                  : str = SnnPars.read_write_attr('tag_run', fun= __update_files)
    tag_sub_process          : str = SnnPars.read_write_attr('tag_sub_process', fun= __update_sub_process)

    # GAITS
    def __gait_update(self, attr, value):
        ''' Additional update of gait/gaitflag '''
        if attr == 'gait':
            self.__gaitflag = self.gaits.index(value)
        if attr == 'gaitflag':
            self.__gait = self.gaits[value]
        self.__update_sim_pars('gaitflag', self.gaitflag)
        if self.gaitflag == 0:
            self.include_online_act = False

    gaits    : tuple[str] = SnnPars.read_only_attr('gaits')
    gaitflag : int        = SnnPars.read_write_attr('gaitflag', fun= __gait_update)
    gait     : str        = SnnPars.read_write_attr('gait',     fun= __gait_update)

    # CURRENTS
    def __multiplier_update(self, attr, value):
        ''' Additional update of the multiplier '''
        self.__update_sim_pars(attr, value)
        if attr == 'stim_lr_asym':
            self.__stim_l_multiplier = 1 + value/2
            self.__stim_r_multiplier = 1 - value/2
        if attr == 'stim_fe_asym':
            self.__stim_f_multiplier = 1 + value/2
            self.__stim_e_multiplier = 1 - value/2

    # Drive
    stim_a_mul : Union[float, np.ndarray] = SnnPars.read_write_attr('stim_a_mul')
    stim_a_off : Union[float, np.ndarray] = SnnPars.read_write_attr('stim_a_off')
    stim_l_mul : Union[float, np.ndarray] = SnnPars.read_write_attr('stim_l_mul')
    stim_l_off : Union[float, np.ndarray] = SnnPars.read_write_attr('stim_l_off')

    stim_l_multiplier   : float = SnnPars.read_only_attr('stim_l_multiplier')
    stim_r_multiplier   : float = SnnPars.read_only_attr('stim_r_multiplier')
    stim_f_multiplier   : float = SnnPars.read_only_attr('stim_f_multiplier')
    stim_e_multiplier   : float = SnnPars.read_only_attr('stim_e_multiplier')

    # Turning
    stim_lr_asym  : float = SnnPars.read_write_attr('stim_lr_asym', fun= __multiplier_update)
    stim_lr_off   : float = SnnPars.read_write_attr('stim_lr_off')

    # Duty cycle
    stim_fe_asym  : float = SnnPars.read_write_attr('stim_fe_asym', fun= __multiplier_update)
    stim_fe_off   : float = SnnPars.read_write_attr('stim_fe_off')
    stim_f_mn_off : float = SnnPars.read_write_attr('stim_f_mn_off')
    stim_e_mn_off : float = SnnPars.read_write_attr('stim_e_mn_off')

    # CURRENT GAINS
    gains_drives_axis    : np.ndarray = SnnPars.read_write_attr('gains_drives_axis',    fun= __update_sim_pars)
    gains_drives_limbs   : np.ndarray = SnnPars.read_write_attr('gains_drives_limbs',   fun= __update_sim_pars)
    mc_gain_axial : Union[float, np.ndarray] = SnnPars.read_write_attr('mc_gain_axial', fun= __update_sim_pars)
    mc_gain_limbs : Union[float, np.ndarray] = SnnPars.read_write_attr('mc_gain_limbs', fun= __update_sim_pars)
    ps_gain_axial : Union[float, np.ndarray] = SnnPars.read_write_attr('ps_gain_axial', fun= __update_sim_pars)
    ps_gain_limbs : Union[float, np.ndarray] = SnnPars.read_write_attr('ps_gain_limbs', fun= __update_sim_pars)
    es_gain_axial : Union[float, np.ndarray] = SnnPars.read_write_attr('es_gain_axial', fun= __update_sim_pars)
    es_gain_limbs : Union[float, np.ndarray] = SnnPars.read_write_attr('es_gain_limbs', fun= __update_sim_pars)

    mo_cocontraction_gain: Union[float, np.ndarray] = SnnPars.read_write_attr('mo_cocontraction_gain', fun= __update_sim_pars)
    mo_cocontraction_off : Union[float, np.ndarray] = SnnPars.read_write_attr('mo_cocontraction_off',  fun= __update_sim_pars)

# TEST
def main():
    ''' Return test parameters '''
    logging.info('TEST: Simulation Parameters')
    pars = SnnParsSimulation(
        parsname= 'pars_simulation_test'
    )
    return pars

if __name__ == '__main__':
    main()