''' Script to run multiple sensitivity analysis with the desired module '''

import sal_experiment

def main():
    ''' Run sensitivity analysis '''

    sal_experiment.run_experiment(
        simulation_data_file_tag = 'swimming_cpg_neural_parameters_ps_095',
        alpha_fraction_th        = 0.95,
    )

if __name__ == '__main__':
    main()
