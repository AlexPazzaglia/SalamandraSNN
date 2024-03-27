''' Read kinematics data '''
import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline

def interpolate_angles(kinematics_data):
    ''' Interpolate angles '''

    time = np.linspace(0, 1, kinematics_data.shape[0])


    # Interpolated dataframe
    new_time        = np.linspace(0, 1, num=1000, endpoint=False)
    interpolated_df = pd.DataFrame({'time': new_time})

    for col in kinematics_data.columns:
        angle_values = kinematics_data[col]

        # Cubic interpolation
        func_interpolate     = CubicSpline(time, angle_values)
        interpolated_values  = func_interpolate(new_time)
        interpolated_df[col] = interpolated_values

    return interpolated_df

def main():
    filename = 'experimental_data/salamander_kinematics_karakasiliotis/swimming/spine_reduced.csv'

    n_joints_axial    = 8
    kinematics_names = [ f'Axis {ind + 1}' for ind in range(n_joints_axial) ]

    # Load data
    kinematics_data = pd.read_csv(
        filename,
        names  = kinematics_names,
        header = None,
    )

    # Interpolate angles
    kinematics_data = interpolate_angles(kinematics_data)

    # Save the dataframe as a csv file
    kinematics_data.to_csv(
        'experimental_data/kinematics_karakasiliotis/swimming/data_karakasiliotis_processed.csv',
        index = False,
    )

    return


if __name__ == '__main__':
    main()