
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

    folder_name = (
        'farms_experiments/experiments/'
        'salamandra_v4_position_control_swimming/kinematics'
    )

    n_copies  = 10
    n_samples = 21
    period    = 1/2.13

    time_samples = np.linspace(0, period*n_copies, (n_samples-1)*n_copies)
    time_interp  = np.arange(0, period*n_copies, 0.001)

    # Spinal data
    angles_axis = np.genfromtxt(
        f'{folder_name}/spine_reduced.csv',
        delimiter=',',
    )
    angles_axis_rep = np.tile( angles_axis, (n_copies,1))

    angles_axis_interp = interp1d(
        time_samples,
        angles_axis_rep,
        kind='cubic',
        axis=0
    )(time_interp)

    # Limb data
    angles_limbs_interp   = np.zeros((len(time_interp),16))
    for i in [0,4,8,12]:
        angles_limbs_interp[:,i] = (1-np.exp(-3*time_interp))*(-70)


    # Save data
    np.savetxt(
        f'{folder_name}/data_karakasiliotis.csv',
        np.c_[time_interp, angles_axis_interp, angles_limbs_interp],
        delimiter = ","
    )

    # Plotting
    n=len(angles_axis_interp[0])

    # Spinal data
    plt.figure(1, figsize=(5,5))
    for i in range(1,n+1):
        plt.plot(time_interp, angles_axis_interp[:,i-1]-30*i,)
        plt.xlim([0,max(time_interp)])

    plt.xlabel("% cycle")
    plt.ylabel("Angle (deg)")
    plt.savefig(f"{folder_name}/spine_data.png")

    # Limb data
    plt.figure(2, figsize=(20,10))
    for i in range(1,17):
        plt.subplot(4,4,i)
        plt.plot(time_interp, angles_limbs_interp[:,i-1])

        plt.title(LIMB_NAMES[i-1])
        plt.xlim([0,max(time_interp)])
        plt.ylabel("Angle (deg)")

    plt.xlabel("% cycle")
    plt.savefig(f"{folder_name}/limb_data.png")


    plt.show()

if __name__ == "__main__":
    main()

