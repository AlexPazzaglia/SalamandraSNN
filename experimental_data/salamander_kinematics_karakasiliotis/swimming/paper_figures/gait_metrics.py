import numpy as np
import matplotlib.pyplot as plt


def main():

    # Experimental data
    mu_kin_water        = 1.0034 * 1e-6
    body_length_m       = 0.176
    snout_vent_length_m = 0.085

    frequencies_hz = np.array(
        [
            2.6,
            2.9,
            3.4,
            3.8,
        ]
    )

    stride_lengths_bl = np.array(
        [
            0.18,
            0.21,
            0.23,
            0.28,
        ]
    )

    displacements_bl = np.array(
        [
            4.0,
            2.5,
            2.2,
            3.5,
            4.3,
            5.0,
            6.5,
            8.0,
            9.5,
            11.0,
            13.0,
        ]
    ) / 100

    # Derived values
    stride_lengths_m = stride_lengths_bl * body_length_m

    mean_frequency_hz     = np.mean(frequencies_hz)
    mean_stride_length_bl = np.mean(stride_lengths_bl)
    mean_stride_length_m  = np.mean(stride_lengths_m)

    mean_velocity_bl = mean_stride_length_bl * mean_frequency_hz
    mean_velocity_m  = mean_stride_length_m  * mean_frequency_hz

    displacements_m = displacements_bl * body_length_m

    tail_amplitude_bl = displacements_bl[-1]
    tail_amplitude_m  = displacements_m[-1]

    strouhal_number = mean_frequency_hz * tail_amplitude_m / mean_velocity_m
    reynolds_number = mean_velocity_m * body_length_m / mu_kin_water
    swimming_number = 2*np.pi* mean_frequency_hz * tail_amplitude_m * body_length_m / mu_kin_water

    # Print to file
    folder_path = 'experimental_data/salamander_kinematics_karakasiliotis'
    with open(f'{folder_path}/swimming/paper_figures/gait_metrics.txt', 'w') as outfile:
        outfile.write(f'Experimental values:\n')
        outfile.write(f'  Body length:      {body_length_m:.3f} m\n')
        outfile.write(f'  Frequency:        {mean_frequency_hz:.3f} Hz\n')

        outfile.write('\n')
        outfile.write(f'  Stride length:    {mean_stride_length_m:.3f} m\n')
        outfile.write(f'  Velocity:         {mean_velocity_m:.3f} m/s\n')
        outfile.write(f'  Tail amplitude    {tail_amplitude_m:.3f} m\n')

        outfile.write('\n')
        outfile.write(f'  Stride length bl: {mean_stride_length_bl:.3f} BL\n')
        outfile.write(f'  Velocity_bl:      {mean_velocity_bl:.3f} BL/s\n')
        outfile.write(f'  Tail amplitude bl {tail_amplitude_bl:.3f} BL\n')

        outfile.write('\n')
        outfile.write(f'  Strouhal number   {strouhal_number:.3f}\n')
        outfile.write(f'  Reynolds number   {reynolds_number:.3f}\n')
        outfile.write(f'  Swimming number   {swimming_number:.3f}\n')

        outfile.write('\n')
        displacements_m_str = ' '.join([f'{displacement_m:.3f}' for displacement_m in displacements_m])
        displacements_bl_str = ' '.join([f'{displacement_bl:.3f}' for displacement_bl in displacements_bl])
        outfile.write(f'  Displacements     [ {displacements_m_str} ]\n')
        outfile.write(f'  Displacements bl  [ {displacements_bl_str} ]\n')


if __name__ == '__main__':
    main()