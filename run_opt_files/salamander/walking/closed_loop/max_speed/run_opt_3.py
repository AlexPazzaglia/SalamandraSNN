''' Run optimization in closed loop with farms '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments import (
    snn_utils,
    snn_simulation_setup,
    snn_simulation,
    snn_optimization,
    snn_optimization_parser
)

def main():
    ''' Run optimization in closed loop with farms '''

    args = snn_optimization_parser.parse_arguments()

    results_path = args['results_path']
    model_name   = '4limb_1dof'

    # Process parameters
    n_processes     = args['n_processes']
    n_sub_processes = args['n_sub_processes']

    params_processes_shared = {
        'connectivity_limbs_filename': 'connectivity_limb_salamandra_bilateral',
        'limb_connectivity_scheme'   : 'inhibitory',
        'verboserun'                 : False,
        'load_connectivity_indices'  : False,
    }

    # CLOSED LOOP WALKING
    obj_optimization =  [
        ['speed_fwd', 'max'],
        [      'cot', 'min'],
    ]

    constr_optimization = {
        'freq_ax'  : [  0.5,  1.5],
        'freq_diff': [ None,  0.1],
        'ptcc_ax'  : [  1.0, None],
        'ptcc_lb'  : [  1.0, None],
        'speed_fwd': [ 0.01,  0.2],
        'speed_lat': [-0.01, 0.01],
    }

    params_processes_shared['gaitflag'] = 1
    params_processes_shared['simulation_data_file_tag'] = 'walking_closedloop_max_speed_3'
    pars_optimization = [
        (                   'ps_gain_axial',  0.0, 10.0),
        (           'mech_activation_delays',  0.0,  1.0),
        (               'gains_drives_axis',  0.5,  1.5),
        (              'gains_drives_limbs',  0.5,  1.5),
        (                   'mc_gain_axial',  0.1,  1.0),
        (                   'mc_gain_limbs',  0.1,  1.5),
        ('mech_lb_pairs_gains_mean_fe_lead',  0.1,  1.0),
        ('mech_lb_pairs_gains_mean_fe_foll',  0.1,  1.0),
        ('mech_lb_pairs_gains_asym_fe_lead', -1.0,  1.0),
        ('mech_lb_pairs_gains_asym_fe_foll', -1.0,  1.0),
    ]

    snn_optimization.run_optimization_closed_loop(
        modname          = snn_simulation.MODELS_FARMS[model_name][0],
        parsname         = snn_simulation.MODELS_FARMS[model_name][1],
        params_processes = snn_simulation_setup.get_params_processes(
                        params_processes_shared = params_processes_shared,
                        n_processes_copies      = n_processes,
                        np_random_seed          = args['np_random_seed'],
                    ),
        tag_folder          = snn_utils.prepend_date_to_tag('OPTIMIZATION'),
        pars_optimization   = pars_optimization,
        obj_optimization    = obj_optimization,
        results_path        = results_path,
        constr_optimization = constr_optimization,
        n_sub_processes     = n_sub_processes,

        pop_size = args['pop_size'],
        n_gen    = args['n_gen'],
    )


if __name__ == '__main__':
    main()
