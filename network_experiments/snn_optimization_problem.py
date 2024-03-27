'''
Functions for Neuromechanical Model Optimization

This module contains functions and classes for performing optimization on neuromechanical models using the Pymoo library.
It defines an optimization problem using multi-objective optimization techniques and parallel simulation processes.

Classes:
    - OptimizationProblem
'''

import os
import copy
import time
import shutil
import logging

from typing import Callable, Union
from multiprocessing import Process, Pipe

import dill
import numpy as np

from pymoo.core.problem import Problem

from network_modules.experiment.network_experiment import SnnExperiment
from network_modules.performance.network_performance import SNN_METRICS
from network_modules.performance.mechanics_performance import MECH_METRICS

from network_experiments import (
    snn_simulation_data,
    snn_simulation_setup,
    snn_optimization_utils,
    snn_simulation,
)

# Default parameters for the optimization algorithm
DEFAULT_PARAMS = {
    'pop_size'      : 50,
    'n_gen'         : 300,
    'np_random_seed': 100,
}

class OptimizationPropblem(Problem):
    '''
    Pymoo optimization problem for neuromechanical simulations.

    This class defines an optimization problem using the Pymoo library for multi-objective optimization.

    Parameters:
    - control_type (str): Type of control for the simulation.
    - n_sub_processes (int): Number of sub-processes for parallel simulation.
    - net (SnnExperiment): The SnnExperiment object containing simulation parameters.
    - vars_optimization (list[list[str, float, float]]): List of variable optimization parameters.
    - obj_optimization (list[list[str, str]]): List of objective optimization parameters.
    - constr_optimization (dict[str, tuple[float]]): Dictionary of constraint optimization parameters (default: None).
    - mech_sim_options (dict): Options for mechanics simulation (default: None).
    - pop_size (int): Population size for optimization (default: 50).
    - n_gen (int): Number of generations for optimization (default: 300).
    - gen_index_start (int): Starting generation index (default: 0).
    '''

    def __init__(
        self,
        control_type            : str,
        n_sub_processes         : int,
        net                     : SnnExperiment,
        vars_optimization       : list[ list[str, float, float]],
        obj_optimization        : list[list[str, str]],
        constr_optimization     : dict[str, tuple[float]] = None,
        mech_sim_options        : dict = None,
        pop_size                : int = DEFAULT_PARAMS['pop_size'],
        n_gen                   : int = DEFAULT_PARAMS['n_gen'],
        gen_index_start         : int = 0,
        motor_output_signal_func: Callable = None,
    ):
        '''
        Optimization problem

        - vars optimization contains the name, lower limit and upper limit of the parameters \n
        vars_optimization = [
            ('mc_gain_axial', 0, 2),
            ('mc_gain_limbs', 0, 2),
        ]

        - obj_optimization contains a list of the names of the objective functions \n
        obj_optimization = [
            [    'speed', 'trg', 0.1 ],
            [  'ptcc_ax', 'max'],
            [      'cot', 'min'],
        ]

        - constr_optimization contains the name, lower bound and upper bount of the constraints \n
        constr_optimization = {
            'speed'   : [0.5, None],
            'ptcc_lb' : [0.9, 1.8],
        }
        '''

        if mech_sim_options is None:
            mech_sim_options = snn_simulation_setup.get_mech_sim_options()

        if control_type == 'closed_loop':
            self.mech_sim_options         = mech_sim_options
            self.metrics_names_list       = SNN_METRICS + MECH_METRICS
            self.motor_output_signal_func = None
        elif control_type == 'open_loop':
            self.mech_sim_options         = None
            self.metrics_names_list       = SNN_METRICS
            self.motor_output_signal_func = None
        elif control_type == 'signal_driven':
            self.mech_sim_options         = mech_sim_options
            self.metrics_names_list       = MECH_METRICS
            self.motor_output_signal_func = motor_output_signal_func
        elif control_type == 'position_control':
            self.mech_sim_options         = mech_sim_options
            self.metrics_names_list       = MECH_METRICS
            self.motor_output_signal_func = None
        elif control_type == 'hybrid_position_control':
            self.mech_sim_options         = mech_sim_options
            self.metrics_names_list       = SNN_METRICS + MECH_METRICS
            self.motor_output_signal_func = None

        # Internal parameters
        self.exclude_from_serialization = ['snn_sim']

        self.control_type     = control_type
        self.snn_sim          = net
        self.n_sub_processes  = n_sub_processes
        self.pop_size         = pop_size
        self.n_gen            = n_gen
        self.params_runs      = None
        self.gen_index_start  = gen_index_start
        self.gen_index        = gen_index_start

        # Optimization parameters
        self._define_variables(vars_optimization)
        self._define_objectives(obj_optimization)
        self._define_constraints(constr_optimization)

        logging.info(
            'RUNNING OPTIMIZATION:'
            ' %i population size, %i generations, %i sub processes,'
            ' %i input variables, %i objective functions, %i constraints',
            self.pop_size,
            self.n_gen,
            self.n_sub_processes,
            self.n_vars,
            self.n_obj,
            self.n_ieq_constr,
        )

        # Saving
        self.data_path = net.params.simulation.results_data_folder_process
        self.data      = {}
        os.makedirs(f'{self.data_path}/optimization_results', exist_ok=True)

        self._save_internal_parameters()

        # Divide indices of inputs across sub_processes
        self.n_sub_processes_inputs = np.array(
            [
                self.pop_size // self.n_sub_processes
                for sub_process in range(self.n_sub_processes)
            ]
        )
        for i in range(self.pop_size % self.n_sub_processes):
            self.n_sub_processes_inputs[i] += 1

        # Mother class
        super().__init__(
            n_var        = self.n_vars,         # Number of variables
            n_obj        = self.n_obj,          # Number of objectives
            n_ieq_constr = self.n_ieq_constr,   # Number of constraints
            xl           = self.pars_l_list,    # Minumum values of variables
            xu           = self.pars_u_list,    # Maximum values of variables
        )

    # Define optimization parameters
    def _define_vector_variables(self):
        '''
        Define parameters accounting for multiple equal or independent scalar values.

        This method defines vector parameters for the optimization problem.
        '''

        mech_pars = self.snn_sim.params.mechanics
        rs_module = self.snn_sim.params.topology.network_modules['rs']

        # Each element is independent
        self.vector_parameters_independent : dict[str, np.ndarray] = {
            # Drives
            'gains_drives_axis'                : np.zeros(rs_module['axial'].pools),
            'gains_drives_limbs'               : np.zeros(rs_module['limbs'].pools),
            'stim_a_mul_vect'                  : np.zeros(rs_module['axial'].pools),
            'stim_l_mul_vect'                  : np.zeros(rs_module['limbs'].pools),
            'stim_a_off_vect'                  : np.zeros(rs_module['axial'].pools),
            'stim_l_off_vect'                  : np.zeros(rs_module['limbs'].pools),
            # Gains
            'mc_gain_axial_vect'               : np.zeros(mech_pars.mech_axial_joints),
            'mc_gain_limbs_vect'               : np.zeros(mech_pars.mech_limbs_joints),
            'ps_gain_axial_vect'               : np.zeros(mech_pars.mech_axial_joints),
            'ps_gain_limbs_vect'               : np.zeros(mech_pars.mech_limbs_joints),
            'es_gain_axial_vect'               : np.zeros(mech_pars.mech_axial_joints),
            'es_gain_limbs_vect'               : np.zeros(mech_pars.mech_limbs_joints),
            # Limbs
            'mech_lb_pairs_gains_mean_fe_lead' : mech_pars.mech_lb_pairs_gains_mean_fe_lead,
            'mech_lb_pairs_gains_mean_fe_foll' : mech_pars.mech_lb_pairs_gains_mean_fe_foll,
            'mech_lb_pairs_gains_mean_fe_free' : mech_pars.mech_lb_pairs_gains_mean_fe_free,
            'mech_lb_pairs_gains_asym_fe_lead' : mech_pars.mech_lb_pairs_gains_asym_fe_lead,
            'mech_lb_pairs_gains_asym_fe_foll' : mech_pars.mech_lb_pairs_gains_asym_fe_foll,
            'mech_lb_pairs_gains_asym_fe_free' : mech_pars.mech_lb_pairs_gains_asym_fe_free,
        }
        # All values are equal
        self.vector_parameters_equal : dict[str, np.ndarray] = {
            'gains_drives_limbs' : self.snn_sim.params.simulation.gains_drives_limbs,
        }
        return

    def _define_variables(self, vars_optimization: list[ list[str, float, float]]):
        ''' Default definition of optimization variables '''

        self._define_vector_variables()

        self.vars_optimization  = vars_optimization
        self.pars_names_list    = [pars[0] for pars in vars_optimization]

        self.pars_names_extended_list = []
        self.pars_l_list              = []
        self.pars_u_list              = []
        for par_ind, par_name in enumerate(self.pars_names_list):

            # Independent vector values
            if par_name in self.vector_parameters_independent:
                n_vec = self.vector_parameters_independent[par_name].size
                self.pars_names_extended_list += [f'{par_name}_{i}' for i in range(n_vec)]
                self.pars_l_list += [ vars_optimization[par_ind][1] for _ in range(n_vec)]
                self.pars_u_list += [ vars_optimization[par_ind][2] for _ in range(n_vec)]
                continue

            # Scalar values and equal vector values
            self.pars_names_extended_list.append(par_name)
            self.pars_l_list.append(vars_optimization[par_ind][1])
            self.pars_u_list.append(vars_optimization[par_ind][2])

        self.n_vars = len(self.pars_names_extended_list)
        return

    def _define_objectives(self, obj_optimization: list[list[str, str]]):
        ''' Default definition of optimization objectives '''

        assert all([obj[1] in ['max','min','trg']  for obj in obj_optimization])

        self.obj_optimization : list[dict] = []

        for obj_opt in obj_optimization:

            objective = {}
            objective['name'] = obj_opt[0]
            objective['type'] = obj_opt[1]

            if objective['type'] == 'min':
                objective['sign']   = +1
                objective['target'] = None

            if objective['type'] == 'max':
                objective['sign']   = -1
                objective['target'] = None

            if objective['type'] == 'trg':
                objective['sign']   = None
                objective['target'] = obj_opt[2]

            self.obj_optimization.append(objective)

        self.n_obj = len( self.obj_optimization )
        return

    def _define_constraints(self, constr_optimization: dict[str, tuple[float]]):
        ''' Default definition of optimization constraints '''

        if constr_optimization is None:
            constr_optimization = {}

        self.constr_optimization       : dict[str, tuple[float]] = constr_optimization
        self.constr_optimization_names : list[str]               = sorted(constr_optimization)

        self.n_ieq_constr = 1 + len(
            [
                limit
                for metric_name in self.constr_optimization_names
                for limit in self.constr_optimization[metric_name]
                if limit is not None
            ]
        )
        return

    # Assign parameters
    def _get_single_run_parameters(
        self,
        input_vector: np.ndarray
    ) -> dict:
        '''
        Get the run_params dictionary for a single run.

        Parameters:
        - input_vector (np.ndarray): Input vector for the optimization.

        Returns:
        - dict: Dictionary of simulation parameters for a single run.
        '''
        param_ind      = 0
        params_run = {}

        for par in self.pars_names_list:

            # Independent vector parameters
            if par in self.vector_parameters_independent:
                dims_vec        = self.vector_parameters_independent[par].shape
                n_vec           = self.vector_parameters_independent[par].size
                par             = par.replace('_vect','')
                params_run[par] = input_vector[ param_ind : param_ind + n_vec ].reshape(dims_vec)
                param_ind      += n_vec
                continue

            # Equal vector parameters
            if par in self.vector_parameters_equal:
                dims_vec        = self.vector_parameters_equal[par].shape
                par             = par.replace('_vect','')
                params_run[par] = input_vector[param_ind] * np.ones(dims_vec)
                param_ind      += 1
                continue

            # Scalar parameters
            params_run[par] = input_vector[param_ind]
            param_ind      += 1

        assert param_ind == len(input_vector), 'Wrong number of parameters'

        return params_run

    def _get_multi_run_parameters(
        self,
        input_vector_list: list[np.ndarray]
    ) -> list[dict]:
        '''
        Get a list of run_params dictionaries for multiple runs.

        Parameters:
        - input_vector_list (list[np.ndarray]): List of input vectors for the optimization.

        Returns:
        - list[dict]: List of dictionaries containing simulation parameters for multiple runs.
        '''

        self.params_runs = []
        for input_vector in input_vector_list:
            params_run = self._get_single_run_parameters(input_vector)
            self.params_runs.append(params_run)

        return self.params_runs

    def _distribute_run_parameters(self):
        '''
        Distribute the list of run parameters across sub-processes.

        This method divides the parameters among sub-processes for parallel simulation.
        '''
        self.sub_processes_runs_params = [
            [
                self.params_runs[ np.sum(self.n_sub_processes_inputs[:sub_process])  + i]
                for i in range(self.n_sub_processes_inputs[sub_process])
            ]
            for sub_process in range(self.n_sub_processes)
        ]

    # Function evaluation
    def _evaluate_objectives(self, metrics):
        ''' Default objective function evaluation '''

        # Objective functions to be minimized (e.g. out["F"] = [+cot, abs(speed_fwd - 0.2)] )
        self.objectives = np.array(
            [
                [
                    metrics[ obj['name'] ][individual] * obj['sign']
                    if obj['target'] is None
                    else
                    np.linalg.norm( metrics[ obj['name'] ][individual] - obj['target'] )

                    for obj in self.obj_optimization
                ]
                for individual in range(self.pop_size)
            ]
        )
        return self.objectives

    def _evaluate_constraints(self, metrics):
        ''' Default constraint function evaluation '''

        # Constraints g(x) in the form g(x) <= 0 (e.g. out["G"] = [ (0.9 - ptcc_ax)] )
        self.constraints = np.array(
            [
                [
                    ( metrics[constr_name][individual] - bound_val ) * (2*bound_ind - 1)
                    for constr_name in self.constr_optimization_names
                    for bound_ind, bound_val in enumerate(self.constr_optimization[constr_name])
                    if bound_val is not None
                ] +
                [
                    -0.5 + np.count_nonzero( np.isnan( self.objectives[individual] ) )
                ]
                for individual in range(self.pop_size)
            ]
        )

        # Substitute NaNs with a large value
        self.constraints[ np.isnan(self.constraints) ] = 1e6

        return self.constraints

    def _evaluate(self, x, out, *args, **kwargs):
        '''
        Function called at each problem evaluation.

        Parameters:
        - x (np.ndarray): Input vector for the optimization.
        - out (dict): Output dictionary containing evaluation results.
        '''

        logging.info('Launching generation %i', self.gen_index)
        assert len(x) == self.pop_size

        # Run parameters
        self._get_multi_run_parameters(x)

        # Distribute across sub processes
        self._distribute_run_parameters()

        # Create communication channels
        sub_processes_connections = [
            Pipe() for _sub_process in range(self.n_sub_processes)
        ]

        # Create sub processes
        sub_processes = [
            Process(
                target= snn_optimization_utils.network_sub_process_runner,
                kwargs= {
                    'control_type'            : self.control_type,
                    'snn_sim'                 : self.snn_sim,
                    'sub_process_ind'         : sub_process_ind,
                    'connection'              : sub_processes_connections[sub_process_ind][1],
                    'params_runs'             : self.sub_processes_runs_params[sub_process_ind],
                    'mech_sim_options'        : self.mech_sim_options,
                    'verbose_logging'         : self.gen_index - self.gen_index_start == 0,
                    'motor_output_signal_func': self.motor_output_signal_func,
                }
            )
            for sub_process_ind in range(self.n_sub_processes)
        ]

        # Start sub processes
        for sub_process in sub_processes:
            sub_process.start()
            time.sleep(0.1)

        # Collect output
        sub_processes_metrics_list : list[dict[str, np.ndarray]] = []
        for sub_process_connection in sub_processes_connections:
            sub_processes_metrics_list.append( sub_process_connection[0].recv() )

        # Terminate sub processes
        for sub_process in sub_processes:
            sub_process.join()

        # Append all sub processes results
        metrics : dict[str, np.ndarray] = {}
        for sub_process_metrics in sub_processes_metrics_list:
            for metric_key in sub_process_metrics:

                if metrics.get(metric_key) is None:
                    metrics[metric_key] = sub_process_metrics[metric_key]
                    continue

                metrics[metric_key] = np.concatenate(
                    [
                        metrics[metric_key],
                        sub_process_metrics[metric_key],
                    ],
                    axis= 0,
                )

        # Collect all computed metrics in the output
        for key in self.metrics_names_list:
            out[key] = [ metrics[key][individual] for individual in range(self.pop_size)]

        # Objective functions to be minimized (e.g. out["F"] = [+cot, abs(speed_fwd - 0.2)] )
        out["F"] = self._evaluate_objectives(metrics)

        # Constraints g(x) in the form g(x) <= 0 (e.g. out["G"] = [ (0.9 - ptcc_ax)] )
        out["G"] = self._evaluate_constraints(metrics)

        # Save generation
        self._save_generation_data(input_vec= x, out_object= out)
        self.gen_index += 1
        return

    def _evaluate_single(
        self,
        individual_input: np.ndarray,
        plot_figures    : bool,
        save_prompt     : bool = True,
        new_run_params  : dict = None,
        **kwargs,
    ) -> tuple[ bool, dict[str, Union[float, np.ndarray[float]]] ]:
        '''
        Run a network model with multiple parameter combinations.

        Args:
            params_runs (list[dict]): List of dictionaries containing run parameters.
            save_prompt (bool): Whether to prompt results saving.
            plot_figures (bool): Whether to plot figures.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: Flag for successful simulation.
            dict[str, NDArrayFloat]: Dictionary containing computed metrics for each run.
        '''

        if new_run_params is None:
            new_run_params = {}

        # Ausiliary function to update params_run
        def __update_params_run(pars_run, new_pars_run: dict) -> dict:
            ''' Update params_run with new_params_run '''
            for key, value in new_pars_run.items():
                if isinstance(value, dict):
                    if key not in pars_run:
                        pars_run[key] = {}
                    pars_run[key] = __update_params_run(pars_run[key], value)
                    continue
                pars_run[key] = value
            return pars_run

        # Update params_run
        params_run = self._get_single_run_parameters(input_vector=individual_input)
        params_run = __update_params_run(params_run, new_run_params)

        return snn_simulation.simulate_single_net_single_run(
            control_type             = self.control_type,
            snn_sim                  = self.snn_sim,
            params_run               = params_run,
            plot_figures             = plot_figures,
            save_prompt              = save_prompt,
            mech_sim_options         = self.mech_sim_options,
            motor_output_signal_func = self.motor_output_signal_func,
            **kwargs,
        )

    # Parameter saving
    def _save_internal_parameters(self):
        '''
        Save internal parameters of the optimization.

        This method saves internal parameters of the optimization to a file.
        '''

        folder = self.snn_sim.params.simulation.results_data_folder_process
        os.makedirs(folder, exist_ok=True)

        filename = f'{folder}/parameters_optimization_process.dill'
        logging.info('Saving parameters_optimization_process data to %s', filename)
        with open(filename, 'wb') as outfile:
            dill.dump(self.pars_names_extended_list,  outfile)
            dill.dump(self.metrics_names_list,        outfile)

    # Result saving
    def _save_generation_data(
        self,
        input_vec : np.ndarray,
        out_object: dict[str, np.ndarray],
    ):
        '''
        Save generation data to a file.

        Parameters:
        - input_vec (np.ndarray): Input vector for the optimization.
        - out_object (dict): Output dictionary containing evaluation results.
        '''

        # Collect
        self.data = {}
        self.data["generation"]  = self.gen_index
        self.data["inputs"]      = input_vec
        self.data["outputs"]     = out_object.get("F")
        self.data["constraints"] = out_object.get("G")
        for metric in self.metrics_names_list:
            self.data[metric] = out_object.get(metric)

        # Save
        filename = f'{self.data_path}/optimization_results/generation_{self.gen_index}.dill'
        logging.info(f'Saving generation {self.gen_index} data to {filename}')
        with open(filename, 'wb') as outfile:
            dill.dump(self.data, outfile)
        return

    # Clean saved files
    def clean_saved_files(self):
        '''
        Clean saved files and sub-process folders.

        This method deletes saved files and sub-process folders after optimization.
        '''

        # Neural simulation
        snn_simulation_data.delete_state_files(self.snn_sim)

        # Sub processes
        for sub_process in range(self.n_sub_processes):
            folder_path = f'{self.data_path}/sub_process_{sub_process}'
            logging.info('Deleting %s', folder_path)
            shutil.rmtree(folder_path)

    # Deep Copy override
    def __deepcopy__(self, memo):
        '''
        Deep copy override to avoid copying the neural simulation.

        Parameters:
        - memo: Memoization argument.

        Returns:
        - OptimizationProblem: Deep copy of the object.
        '''
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for attr, value in self.__dict__.items():
            if attr in ['snn_sim']:
                continue
            setattr(result, attr, copy.deepcopy(value, memo))
        return result
