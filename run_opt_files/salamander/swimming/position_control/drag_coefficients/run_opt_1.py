''' Run optimization in closed loop with farms '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_modules.experiment.network_experiment import SnnExperiment

from network_experiments import (
    snn_signals_neural,
    snn_utils,
    snn_simulation_setup,
    snn_simulation,
    snn_optimization,
    snn_optimization_parser
)

from run_opt_files.swimming.drag_coefficients.salamander_opt_problem import SalamanderOptimizationProblem

DURATION = 10
TIMESTEP = 0.001

def main():
    ''' Run optimization in closed loop with farms '''

    args = snn_optimization_parser.parse_arguments()

    results_path = args['results_path']
    model_name   = '4limb_1dof_unweighted'

    # Process parameters
    n_processes     = args['n_processes']
    n_sub_processes = args['n_sub_processes']

    params_processes_shared = {
        'animal_model'            : 'salamandra_v4',
        'simulation_data_file_tag': 'test_signal',
        'gaitflag'                : 0,
        'mech_sim_options'        : snn_simulation_setup.get_mech_sim_options(video= False),
        'duration'                : DURATION,
        'timestep'                : TIMESTEP,
    }

    # SWIMMING
    pars_optimization = (
        [
            [ f'drag_head_x',  -1.0, 0.0 ],
            [ f'drag_head_xy', -1.0, 0.0 ],
            [ f'drag_body_x',  -1.0, 0.0 ],
            [ f'drag_body_xy', -1.0, 0.0 ],
        ]

    )

    obj_optimization = [
        [ 'mech_speed_fwd_bl',     'trg',  0.71],
        [ 'mech_tail_beat_amp_bl', 'trg',  0.16],
    ]

    constr_optimization = {
        'mech_traj_curv'      : (-0.05,  +0.05),    # Min curvature radius at 20m
    }

    snn_optimization.run_optimization_position_control(
        modname          = snn_simulation.MODELS_FARMS[model_name][0],
        parsname         = snn_simulation.MODELS_FARMS[model_name][1],
        params_processes = snn_simulation_setup.get_params_processes(
                        params_processes_shared = params_processes_shared,
                        n_processes_copies             = n_processes,
                    ),
        tag_folder          = snn_utils.prepend_date_to_tag('OPTIMIZATION'),
        results_path        = results_path,
        pars_optimization   = pars_optimization,
        obj_optimization    = obj_optimization,
        constr_optimization = constr_optimization,
        n_sub_processes     = n_sub_processes,
        problem_class       = SalamanderOptimizationProblem,

        pop_size = args['pop_size'],
        n_gen    = args['n_gen'],
    )


if __name__ == '__main__':
    main()

