''' Script to run multiple sensitivity analysis with the desired module '''

import sal_experiment

def main():
    ''' Run sensitivity analysis '''

    sal_experiment.run_experiment(
        simulation_data_file_tag = 'swimming_cpg_neural_parameters_alpha_05_ps_weight_100',
        ps_weight_nominal        = 1.00,
        ps_weight_range          = [0.95, 1.05]
    )

if __name__ == '__main__':
    main()