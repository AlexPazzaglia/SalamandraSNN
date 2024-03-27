''' Read kinematics data '''
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

def main():
    ''' Main function '''

    filename = 'walking/data_karakasiliotis.csv'

    # Column labels
    n_joints_axial     = 8
    names_joints_axial = [ f'Axis {ind}' for ind in range(1, 1 + n_joints_axial) ]
    names_joints_limbs = [
        "Left Shoulder-yaw",
        "Left Shoulder-pitch",
        "Left Shoulder-roll",
        "Left Elbow",
        "Right Shoulder-yaw",
        "Right Shoulder-pitch",
        "Right Shoulder-roll",
        "Right Elbow",
        "Left Hip-yaw",
        "Left Hip-pitch",
        "Left Hip-roll",
        "Left Knee",
        "Right Hip-yaw",
        "Right Hip-pitch",
        "Right Hip-roll",
        "Right Knee"
    ]

    kinematics_names = ['time'] + names_joints_axial + names_joints_limbs

    # Load data
    kinematics_data = pd.read_csv(
        filename,
        names  = kinematics_names,
        header = None,
    )

    # Save data with labels
    kinematics_data.to_csv(
        'walking/data_karakasiliotis_processed.csv',
        index = False,
    )

    return


if __name__ == '__main__':
    main()