import numpy as np
import matplotlib.pyplot as plt

def main():

    # REFERENCE
    frequencies_ref = np.array(
        [
            2.6,
            2.9,
            3.4,
            3.8,
        ]
    )

    stride_lengths_ref = np.array(
        [
            0.18,
            0.28,
            0.23,
            0.21,
        ]
    )

    # MODEL
    frequencies_model = np.array(
        [
            2.50,
            2.75,
            3.00,
            3.25,
            3.5,
            3.75,
            4.00
        ]
    )
    stride_lengths_model = np.array(
        [
            0.21627545265067258,
            0.21700204944261414,
            0.22674602120789394,
            0.23523420106758955,
            0.24486936631470338,
            0.23913564330593695,
            0.24698187222183596,
        ]
    )

    # Scatter plot
    plt.figure()


    zorder = 2
    plt.scatter(
        frequencies_ref,
        stride_lengths_ref,
        c      = 'blue',
        marker = 'o',
        s      = 70,
        zorder = zorder,
        label  = 'reference',
    )

    plt.scatter(
        frequencies_model,
        stride_lengths_model,
        c      = 'red',
        marker = '*',
        s      = 100,
        zorder = zorder,
        label  = 'simulation',
    )

    plt.xlim((2.4, 4.1))
    plt.ylim((0.15, 0.30))

    plt.xticks([2.5, 3.0, 3.5, 4.0], fontsize=15)
    plt.yticks([0.15, 0.20, 0.25, 0.30], fontsize=15)

    plt.grid(linestyle='--')
    plt.xlabel('frequency [Hz]', fontsize=15)
    plt.ylabel('stride length [BL/cycle]', fontsize=15)
    plt.legend( loc='lower right', fontsize=15)
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()