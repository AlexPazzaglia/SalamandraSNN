''' Run the spinal cord model together with the mechanical simulator '''

import experiment

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    ps_gain_axial            = 0.3
    simulation_data_file_tag = f'noise_effect_vs_feedback_weight_th_{round(ps_gain_axial * 100) :03d}'

    experiment.run_experiment(
        ps_gain_axial            = ps_gain_axial,
        simulation_data_file_tag = simulation_data_file_tag,
    )

if __name__ == '__main__':
    main()
