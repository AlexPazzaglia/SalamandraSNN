''' Run the spinal cord model together with the mechanical simulator '''

import experiment

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    ps_gain_axial = 0.1
    ps_type       = 'in_up'

    experiment.run_experiment(
        ps_gain_axial = ps_gain_axial,
        ps_type       = ps_type,
    )

if __name__ == '__main__':
    main()
