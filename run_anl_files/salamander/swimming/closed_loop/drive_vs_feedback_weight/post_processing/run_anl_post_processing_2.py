''' Run the analysis-specific post-processing script '''

import numpy as np
import post_processing

def main():
    ''' Run the analysis-specific post-processing script '''

    folder_name = 'net_farms_limbless_ps_weight_drive_vs_feedback_weight_alpha_fraction_th_030_100_2023_12_21_22_49_33_ANALYSIS'
    stim_a_amp  = np.linspace(-2, 5, 41) + 5.0
    ps_weight   = np.linspace(0.0, 3.0, 31)

    post_processing.run_post_processing(
        folder_name = folder_name,
        stim_a_amp  = stim_a_amp,
        ps_weight   = ps_weight,
    )


if __name__ == '__main__':
    main()