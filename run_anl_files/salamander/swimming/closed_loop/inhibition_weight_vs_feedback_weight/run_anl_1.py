''' Run the spinal cord model together with the mechanical simulator '''

import numpy as np
import experiment

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    ps_gain_axial        = 0.1
    inhibition_gain_type = 'wide'

    experiment.run_experiment(
        ps_gain_axial            = ps_gain_axial,
        inhibition_gain_type     = inhibition_gain_type,
    )

if __name__ == '__main__':
    main()
