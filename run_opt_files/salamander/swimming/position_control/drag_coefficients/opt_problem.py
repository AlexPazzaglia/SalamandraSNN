''' Run optimization in closed loop with farms '''

import os
import sys
import inspect

import numpy as np
from typing import Callable

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_modules.experiment.network_experiment import SnnExperiment

from network_experiments import snn_optimization_problem

N_JOINTS_AXIS = 8

class SalamanderOptimizationProblem(snn_optimization_problem.OptimizationPropblem):

    def __init__(
        self,
        n_sub_processes         : int,
        net                     : SnnExperiment,
        vars_optimization       : list[ list[str, float, float]],
        obj_optimization        : list[dict],
        constr_optimization     : dict[str, tuple[float]] = None,
        pop_size                : int = snn_optimization_problem.DEFAULT_PARAMS['pop_size'],
        n_gen                   : int = snn_optimization_problem.DEFAULT_PARAMS['n_gen'],
        gen_index_start         : int = 0,
        motor_output_signal_func: Callable = None,
        **kwargs
    ):

        super().__init__(
            control_type             = 'position_control',
            n_sub_processes          = n_sub_processes,
            net                      = net,
            vars_optimization        = vars_optimization,
            obj_optimization         = obj_optimization,
            constr_optimization      = constr_optimization,
            pop_size                 = pop_size,
            n_gen                    = n_gen,
            gen_index_start          = gen_index_start,
            motor_output_signal_func = motor_output_signal_func,
        )

    def _get_single_run_parameters(self, input_vector: np.ndarray) -> dict:
        ''' Get the parameters for a single run '''

        drag_x_head, drag_yz_head, drag_x_body, drag_yz_body = input_vector

        params_run = {
            'mech_sim_options' : {
                'drag_coefficients_options' : [
                        [
                            ['link_body_0'],
                            [ drag_x_head, drag_yz_head, drag_yz_head]
                        ],
                        [
                            [f'link_body_{i}' for i in range(1, N_JOINTS_AXIS+1)],
                            [ drag_x_body, drag_yz_body, drag_yz_body]
                        ]
                    ],
            }
        }

        return params_run
