#!/usr/bin/env python3
"""Run salamander simulation"""

import sys
import copy
import logging
import numpy as np

from typing import Union
from dm_control.rl.control import PhysicsError

from network_modules.simulation.network_simulation import SnnSimulation
from network_experiments.snn_logging import pretty_string

from farms_core                          import pylog
from farms_core.model.options            import AnimatOptions, ArenaOptions
from farms_core.simulation.options       import SimulationOptions, Simulator
from farms_mujoco.simulation.simulation  import Simulation as MuJoCoSimulation
from farms_mujoco.sensors.camera         import CameraCallback
from farms_sim.simulation                import (
    setup_from_clargs,
    run_simulation,
    simulation_setup,
    postprocessing_from_clargs,
)

from farms_amphibious.callbacks     import setup_callbacks
from farms_amphibious.model.options import (
    SpikingOptions,
    AmphibiousArenaOptions
)

from farms_amphibious.data.data     import (
    SpikingData,
    AmphibiousKinematicsData,
    get_amphibious_data,
)

from farms_amphibious.control.network    import NetworkSNN, NetworkHYBRID
from farms_amphibious.control.kinematics import KinematicsController
from farms_amphibious.control.amphibious import (
    SpikingController,
    get_spiking_controller,
)

class MechSimulation():

    def __init__(
        self,
        snn_network          : SnnSimulation,
        mech_sim_options_dict: dict = None,
        run_params           : dict = None,
        **kwargs
    ):

        pylog.set_level('critical')

        self.snn_network  = snn_network
        self.animal_model = snn_network.params.simulation.animal_model
        self.control_type = snn_network.control_type

        # File paths
        if self.control_type in  ['position_control', 'hybrid_position_control']:
            control_tag = 'position_control'
        else:
            control_tag = 'spiking'

        if snn_network.params.simulation.gait == 'swim':
            gait_tag = 'swimming'
        else:
            gait_tag = 'walking'

        self.mech_path = f'farms_experiments/experiments/{self.animal_model}_{control_tag}_{gait_tag}/'

        self.results_data_folder = f'{snn_network.params.simulation.results_data_folder_run}/farms'
        self.figures_data_folder = f'{snn_network.params.simulation.figures_data_folder_run}/farms'

        if run_params is None:
            run_params = {}

        self.mech_sim_options_dict = {} if mech_sim_options_dict is None else mech_sim_options_dict
        self.mech_sim_options_dict.update(
            run_params.pop('mech_sim_options', {})
        )
        self.save_all_metrics_data = self.mech_sim_options_dict.get('save_all_metrics_data', False)

        # Simulation objects
        self.clargs = None
        self.animat_options  : AnimatOptions     = None
        self.mech_sim_options: SimulationOptions = None
        self.arena_options   : ArenaOptions      = None
        self.simulator       : Simulator         = None

        self.animat_network    : NetworkSNN                                   = None
        self.animat_data       : Union[SpikingData, AmphibiousKinematicsData] = None
        self.animat_controller : SpikingController                            = None
        self.sim               : MuJoCoSimulation                             = None


        # Configuration and outuput files
        # 'farms_experiments/experiments/salamander_spiking_swimming/'
        # 'farms_experiments/experiments/salamander_spiking_walking/'

        sys.argv += [
            '--simulator',          'MUJOCO',
            '--simulation_config',  f'{self.mech_path}simulation.yaml',
            '--animat_config',      f'{self.mech_path}animat.yaml',
            '--arena_config',       f'{self.mech_path}arena.yaml',
            '--profile',            f'{self.mech_path}output/simulation.profile',
        ]

        if self.snn_network.params.simulation.save_mech_logs:
            sys.argv += ['--log_path', self.results_data_folder]

        # Setup
        pylog.info('Loading options from clargs')
        (
            self.clargs,
            self.animat_options,
            self.mech_sim_options,
            self.arena_options,
            self.simulator,
        ) = setup_from_clargs(
            animat_options_loader = SpikingOptions,
            arena_options_loader  = AmphibiousArenaOptions,
        )

        # Randomize initial angles
        # self.randomize_initial_positions()

        # Update parameters
        self._update_mech_sim_parameters()
        self._update_arena_parameters()
        self._update_spawn_parameters()
        self._update_drag_coefficients()
        self._update_muscle_parameters()
        self._update_position_control_parameters()

        # Spiking data
        self.animat_data: Union[SpikingData, AmphibiousKinematicsData] = (
                get_amphibious_data(
                animat_options     = self.animat_options,
                simulation_options = self.mech_sim_options,
            )
        )

        # Spiking network (OK)
        # NOTE: Parameters to use a dummy function
        #  use_motor_output_signal        (bool)
        #  motor_output_signal_parameters (dict)
        #  motor_output_signal_func       (function)

        if self.control_type in ['closed_loop', 'signal_driven']:
            self.animat_network = NetworkSNN(
                data        = self.animat_data,
                snn_network = self.snn_network,
                **kwargs
            )
            controller_args = {'animat_network': self.animat_network}
        elif self.control_type in ['hybrid_position_control']:
            self.animat_network = NetworkHYBRID(
                data        = self.animat_data,
                snn_network = self.snn_network,
                **kwargs
            )
            controller_args = {'animat_network': self.animat_network}
        else:
            controller_args = {}

        # Spiking controller (OK)
        self.animat_controller: Union[SpikingController, KinematicsController] = (
            get_spiking_controller(
                animat_data    = self.animat_data,
                animat_options = self.animat_options,
                sim_options    = self.mech_sim_options,
                **controller_args,
            )
        )

        # Additional engine-specific options
        assert self.simulator == Simulator.MUJOCO, 'Only mujoco simulations are supported'

        self.options = {}
        self.camera = None
        if self.mech_sim_options.video:
            self.camera = CameraCallback(
                camera_id    = 0,
                timestep     = self.mech_sim_options.timestep,
                n_iterations = self.mech_sim_options.n_iterations,
                fps          = self.mech_sim_options.video_fps,
                speed        = self.mech_sim_options.video_speed,
                width        = self.mech_sim_options.video_resolution[0],
                height       = self.mech_sim_options.video_resolution[1],
            )
        self.options['callbacks'] = setup_callbacks(
            animat_options = self.animat_options,
            arena_options  = self.arena_options,
            camera         = self.camera,
        )

    def _update_mech_sim_parameters(self):
        ''' Update mech_sim_parameters '''

        # Ensure duration consistency with neural simulation
        duration = float( self.snn_network.params.simulation.duration )
        timestep = self.mech_sim_options_dict.get('timestep', self.mech_sim_options['timestep'] )

        self.mech_sim_options_dict['timestep']     = timestep
        self.mech_sim_options_dict['n_iterations'] = round( duration / timestep )

        # Update mech_sim_parameters from dictionary
        for key,value in self.mech_sim_options_dict.items():
            if hasattr(self.mech_sim_options, key):
                logging.info(f'Updating mech_sim_options.{key} to {value}')
                setattr(self.mech_sim_options, key, value)

    def _update_arena_parameters(self):
        ''' Update arena parameters '''

        if not 'arena_parameters_options' in self.mech_sim_options_dict:
            return

        arena_pars_options : dict = (
            self.mech_sim_options_dict.pop('arena_parameters_options')
        )

        # Generic arena parameters
        for key in list( arena_pars_options.keys() ):
            if key not in self.arena_options:
                continue

            logging.info(f'Updating arena parameter {key} to {arena_pars_options[key]}')
            self.arena_options[key] = arena_pars_options.pop(key)

        # Water parameters
        if 'water_maps' in arena_pars_options:
            assert 'water_velocity' in arena_pars_options, 'Missing water velocity'

        for key in ['velocity', 'maps', 'viscosity', 'density', 'height', 'sdf', 'maps_parameters']:
            water_key = f'water_{key}'

            if water_key not in arena_pars_options:
                continue

            logging.info(
                f'Updating arena parameter water.{key} to '
                f'{pretty_string( arena_pars_options[water_key] )}'
            )

            # OTHER WATER PARAMETERS
            if water_key != 'water_maps_parameters':
                self.arena_options['water'][key] = arena_pars_options.pop(water_key)
                continue

            # WATER_MAPS_PARAMETERS
            maps_pars = arena_pars_options.pop(water_key)

            self.arena_options['water']['maps'] = [
                maps_pars['path_vx'],
                maps_pars['path_vy'],
            ]

            self.arena_options['water']['velocity'] = [
                maps_pars['v_min_x'],
                maps_pars['v_min_y'],
                maps_pars['v_min_z'],
                maps_pars['v_max_x'],
                maps_pars['v_max_y'],
                maps_pars['v_max_z'],
                maps_pars['pos_min_x'],
                maps_pars['pos_min_y'],
                maps_pars['pos_max_x'],
                maps_pars['pos_max_y'],
            ]

        assert not arena_pars_options, f'Invalid arena parameters {arena_pars_options}'

    def _update_spawn_parameters(self):
        ''' Update spawn parameters '''
        if not 'spawn_parameters_options' in self.mech_sim_options_dict:
            return

        spawn_pars_options : dict[str, tuple[float]] = (
            self.mech_sim_options_dict.pop('spawn_parameters_options')
        )

        for key in spawn_pars_options:
            if key not in self.animat_options['spawn']:
                raise ValueError(f'Invalid spawn parameter {key}')

            logging.info(f'Updating spawn parameter {key} to {spawn_pars_options[key]}')
            self.animat_options['spawn'][key] = spawn_pars_options[key]

    def _update_drag_coefficients(self):
        ''' Update drag coefficients '''

        if not 'drag_coefficients_options' in self.mech_sim_options_dict:
            return

        drag_coeff_options : list[tuple[list[str], float]] = (
            self.mech_sim_options_dict.pop('drag_coefficients_options')
        )

        drag_coeff_links_names = [ drag_coeff_opt[0] for drag_coeff_opt in drag_coeff_options ]
        drag_coeff_values      = [ drag_coeff_opt[1] for drag_coeff_opt in drag_coeff_options ]

        for links_names, drag_coeff in zip(drag_coeff_links_names, drag_coeff_values):
            updaated = np.zeros(len(links_names), dtype= bool)

            # Update drag coefficients of the selected links
            for muscle_options in self.animat_options['morphology']['links']:
                if muscle_options['name'] not in links_names:
                    continue

                updaated[links_names.index(muscle_options['name'])] = True
                logging.info(f'Updating drag coefficients of {muscle_options["name"]} to {drag_coeff}')
                muscle_options['drag_coefficients'][0] = drag_coeff

            if not updaated.all():
                raise ValueError(f'Invalid link names {[links_names[i] for i in np.where(~updaated)[0]]}')

    def _update_muscle_parameters(self):
        ''' Update muscle parameters '''
        if not 'muscle_parameters_options' in self.mech_sim_options_dict:
            return

        muscle_pars_options : tuple[tuple[tuple[str], dict[str, float]]] = (
            self.mech_sim_options_dict.pop('muscle_parameters_options')
        )

        muscle_pars_joints_names = [ muscle_pars_opt[0] for muscle_pars_opt in muscle_pars_options ]
        muscle_pars_values_dict  = [ muscle_pars_opt[1] for muscle_pars_opt in muscle_pars_options ]

        for joints_names, values_dict in zip(
            muscle_pars_joints_names,
            muscle_pars_values_dict,
        ):
            updaated = np.zeros(len(joints_names), dtype= bool)

            # Update muscle parameters of the selected joints
            for muscle_options in self.animat_options['control']['muscles']:
                if muscle_options['joint_name'] not in joints_names:
                    continue

                updaated[joints_names.index(muscle_options['joint_name'])] = True
                logging.info(f'Updating muscle parameters of {muscle_options["joint_name"]} to {values_dict}')
                for var_name, var_value in values_dict.items():
                    muscle_options[var_name] = var_value

            if not updaated.all():
                raise ValueError(f'Invalid joint names {[joints_names[i] for i in np.where(~updaated)[0]]}')

    def _update_position_control_parameters(self):
        ''' Update position control parameters '''
        if not 'position_control_parameters_options' in self.mech_sim_options_dict:
            return

        position_control_pars_options : dict = (
            self.mech_sim_options_dict.pop('position_control_parameters_options')
        )

        # Update position control gains
        if 'position_control_gains' in position_control_pars_options:

            position_gains : list[ tuple[ list[str], list[float] ]] = position_control_pars_options.pop('position_control_gains')

            gains_joints_names  = [ pos_pars_opt[0] for pos_pars_opt in position_gains ]
            gains_joints_values = [ pos_pars_opt[1] for pos_pars_opt in position_gains ]

            for joints_names, gains_values in zip(
                gains_joints_names,
                gains_joints_values,
            ):
                updaated = np.zeros(len(joints_names), dtype= bool)

                # Update muscle parameters of the selected joints
                for joint_options in self.animat_options['control']['motors']:
                    if joint_options['joint_name'] not in joints_names:
                        continue

                    updaated[joints_names.index(joint_options['joint_name'])] = True
                    logging.info(f'Updating position control gains of {joint_options["joint_name"]} to {gains_values}')
                    joint_options['gains'] = gains_values

                if not updaated.all():
                    raise ValueError(f'Invalid joint names {[joints_names[i] for i in np.where(~updaated)[0]]}')

        # Update position control parameters
        for key, value in position_control_pars_options.items():
            if key not in self.animat_options['control'] or 'kinematics' not in key:
                raise ValueError(f'Invalid position control parameter {key}')

            logging.info(f'Updating position control parameter {key} to {value}')
            self.animat_options['control'][key] = value

        return

    def setup_empty_simulation(self):
        ''' Setup simulation '''
        self.sim = simulation_setup(
            animat_data        = self.animat_data,
            animat_options     = self.animat_options,
            animat_controller  = self.animat_controller,
            simulation_options = self.mech_sim_options,
            arena_options      = self.arena_options,
            simulator          = self.simulator,
            **self.options,
        )

    def simulation_run(self, main_thread= False):
        """Main"""

        try:
            # Simulation
            pylog.info('Creating simulation environment')
            self.sim: MuJoCoSimulation = run_simulation(
                animat_data        = self.animat_data,
                animat_options     = self.animat_options,
                animat_controller  = self.animat_controller,
                simulation_options = self.mech_sim_options,
                arena_options      = self.arena_options,
                simulator          = self.simulator,
                **self.options,
            )
        except PhysicsError as physics_error:

            # Directly raise the error if it is the main thread
            if main_thread:
                raise physics_error

            # Otherwise, put the error in the queue
            self.snn_network.q_in.put(physics_error)
            return False

        return True

    def simulation_post_processing(
        self,
        curr_sim: MuJoCoSimulation = None
    ):
        ''' Post-processing '''

        if curr_sim is None:
            curr_sim = self.sim

        # Post-processing
        pylog.info('Running post-processing')
        postprocessing_from_clargs(
            sim                = curr_sim,
            clargs             = self.clargs,
            simulator          = self.simulator,
            animat_data_loader = SpikingData,
        )
