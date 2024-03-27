''' Script to run multiple sensitivity analysis with the desired module '''

import sal_experiment

def main():
    ''' Run sensitivity analysis '''

    alpha_fraction_th = 0.5
    ps_weight_nominal = 0.5

    sal_experiment.run_experiment(
        alpha_fraction_th        = alpha_fraction_th,
        ps_weight_nominal        = ps_weight_nominal,
    )

if __name__ == '__main__':
    main()