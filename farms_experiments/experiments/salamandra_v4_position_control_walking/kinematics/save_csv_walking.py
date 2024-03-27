
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pylab as plt

LIMB_NAMES = (
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
)

def main():
    '''Main'''

    n_joints_spine = 8
    n_copies       = 10
    n_samples      = 100
    period         = 1

    folder_name = (
        'farms_experiments/experiments/'
        'salamandra_v4_position_control_walking/kinematics'
    )

    # time_samples = np.linspace(0, period*n_copies, n_samples*n_copies)
    time_samples_spine = np.linspace(0, period*n_copies, (n_samples-1)*n_copies)
    time_samples_limbs = np.linspace(0, period*n_copies, ( n_samples )*n_copies)
    time_interp        = np.arange(0, period*n_copies, 0.001)

    # Spinal data
    angles_mean_spine = np.genfromtxt(
                f'{folder_name}/spine_reduced.csv',
                delimiter=',',
            )
    angles_mean_spine_rep = np.tile( angles_mean_spine, (n_copies,1))


    angles_mean_spine_interp = interp1d(
        time_samples_spine,
        angles_mean_spine_rep,
        kind='linear',
        axis=0
    )(time_interp)

    # Limb data
    angles_mean_2_limbs = np.genfromtxt(
        f'{folder_name}/mean_limbs.csv',
        delimiter=',',
    )
    angles_std_2_limbs = np.genfromtxt(
        f'{folder_name}/std_limbs.csv',
        delimiter=',',
    )

    # Adjust for DOF convention
    angles_mean_2_limbs[:, [0,4]] = +angles_mean_2_limbs[:, [0,4]] - 90.0
    angles_mean_2_limbs[:, [1,5]] = -angles_mean_2_limbs[:, [1,5]] - 0
    angles_mean_2_limbs[:, [2,6]] = -angles_mean_2_limbs[:, [2,6]] - 0
    angles_mean_2_limbs[:, [3,7]] = -angles_mean_2_limbs[:, [3,7]] + 180.0

    # Adjust for multiple limbs
    angles_mean_LF = angles_mean_2_limbs[:,:4]
    angles_mean_RH = angles_mean_2_limbs[:,4:]

    angles_mean_4_limbs=np.c_[
        angles_mean_LF,
        np.r_[angles_mean_LF[n_samples//2:],angles_mean_LF[0:n_samples//2]],
        np.r_[angles_mean_RH[n_samples//2:],angles_mean_RH[0:n_samples//2]],
        angles_mean_RH,
    ]

    angles_std_LF = angles_std_2_limbs[:,:4]
    angles_std_RH = angles_std_2_limbs[:,4:]

    angles_std_4_limbs=np.c_[
        angles_std_LF,
        np.r_[angles_std_LF[n_samples//2:],angles_std_LF[0:n_samples//2]],
        np.r_[angles_std_RH[n_samples//2:],angles_std_RH[0:n_samples//2]],
        angles_std_RH,
    ]

    angles_mean_limbs_rep = np.tile( angles_mean_4_limbs, (n_copies,1))
    angles_std_limbs_rep  = np.tile( angles_std_4_limbs, (n_copies,1))

    angles_mean_limbs_interp = interp1d(
        time_samples_limbs,
        angles_mean_limbs_rep,
        kind = 'linear',
        axis = 0
    )(time_interp)

    angles_std_limbs_interp  = interp1d(
        time_samples_limbs,
        angles_std_limbs_rep,
        kind = 'linear',
        axis = 0
    )(time_interp)

    # Save data
    C=np.c_[time_interp, angles_mean_spine_interp, angles_mean_limbs_interp]
    D=np.c_[time_interp, angles_mean_spine_interp, angles_std_limbs_interp]
    np.savetxt(f'{folder_name}/data_karakasiliotis.csv', C, delimiter=",")
    np.savetxt(f'{folder_name}/data_karakasiliotis_std.csv', D, delimiter=",")

    # Plotting

    # Spinal data
    plt.figure(1, figsize=(5,5))
    for i in range(1,n_joints_spine+1):
        plt.plot(time_interp, angles_mean_spine_interp[:,i-1]-30*i,)
        plt.xlim([0,max(time_interp)])

    plt.xlabel("% cycle")
    plt.ylabel("Angle (deg)")
    plt.savefig(f"{folder_name}/spine_data.png")

    # Limb data
    plt.figure(2)
    plt.plot(np.arange(n_samples),angles_mean_2_limbs[:,1])
    plt.plot(np.arange(n_samples),angles_std_2_limbs[:,1])


    # Limb data
    plt.figure(3, figsize=(20,10))
    for i in range(1,17):
        y2=angles_mean_limbs_interp[:,i-1]+angles_std_limbs_interp[:,i-1]
        y1=angles_mean_limbs_interp[:,i-1]-angles_std_limbs_interp[:,i-1]

        plt.subplot(4,4,i)
        plt.plot(time_interp, angles_mean_limbs_interp[:,i-1])
        plt.fill_between(
            time_interp,
            y1,
            y2,
            where       = y2 >= y1,
            facecolor   = 'black',
            alpha       = 0.1,
            interpolate = True
        )

        plt.title(LIMB_NAMES[i-1])
        plt.xlim([0,max(time_interp)])
        plt.ylabel("Angle (deg)")

    plt.xlabel("% cycle")
    plt.savefig(f"{folder_name}/limb_data.png")


    plt.show()

if __name__ == "__main__":
    main()

