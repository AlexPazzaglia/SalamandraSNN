''' Script to run multiple sensitivity analysis with the desired module '''

import sal_experiment

def main():
    ''' Run sensitivity analysis '''

    sal_experiment.run_experiment(
        simulation_data_file_tag = 'swimming_cpg_neural_parameters_alpha_03_ps_weight_150',
        ps_weight_nominal        = 1.50,
        ps_weight_range          = [1.45, 1.55]
    )

if __name__ == '__main__':
    main()