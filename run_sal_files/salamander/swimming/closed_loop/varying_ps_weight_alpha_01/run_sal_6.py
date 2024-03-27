''' Script to run multiple sensitivity analysis with the desired module '''

import sal_experiment

def main():
    ''' Run sensitivity analysis '''

    sal_experiment.run_experiment(
        simulation_data_file_tag = 'swimming_cpg_neural_parameters_alpha_01_ps_weight_125',
        ps_weight_nominal        = 1.25,
        ps_weight_range          = [1.20, 1.30]
    )

if __name__ == '__main__':
    main()