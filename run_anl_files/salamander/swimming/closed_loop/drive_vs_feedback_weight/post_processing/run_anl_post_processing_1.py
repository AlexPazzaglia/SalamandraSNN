''' Run the analysis-specific post-processing script '''

import numpy as np
import post_processing

def main():
    ''' Run the analysis-specific post-processing script '''

    folder_name = 'net_farms_limbless_ps_weight_drive_vs_feedback_weight_alpha_fraction_th_010_zoom_100_2023_11_08_20_54_43_ANALYSIS'
    post_processing.run_post_processing(folder_name)


if __name__ == '__main__':
    main()