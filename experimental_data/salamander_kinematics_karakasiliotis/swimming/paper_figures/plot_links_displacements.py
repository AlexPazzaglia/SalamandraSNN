import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def main():

    # REFERENCE
    pos_bl_ref = np.linspace(0, 1, 11)
    disp_bl_ref = np.array(
        [
            0.040,
            0.025,
            0.022,
            0.035,
            0.043,
            0.050,
            0.065,
            0.080,
            0.095,
            0.110,
            0.130,
        ]
    )

    #MODEL
    body_length = 0.1
    pos_bl = np.array(
        [
            0.0000,
            0.0150,
            0.0257,
            0.0364,
            0.0471,
            0.0578,
            0.0685,
            0.0792,
            0.0899,
            0.1000,
        ]
    ) / body_length

    disp_bl_model_list = np.array(
        [
            # Freq = 2.50
            [
                0.03587516919401741,
                0.01677533200226144,
                0.01277463152646766,
                0.023667039858775774,
                0.03230437445287473,
                0.03902224195312948,
                0.041752852583129385,
                0.05398233710282402,
                0.08436900118317653,
                0.11596462647748929
            ],
            # Freq = 2.75
            [
                0.05046265125141421,
                0.024324941197048408,
                0.017555420781504683,
                0.033785471078608,
                0.04688759912939182,
                0.05482010641845458,
                0.05939996314335659,
                0.07740599058156801,
                0.12076658073855137,
                0.1639249736835734
            ],
            # Freq = 3.00
            [
                0.057078002708691455,
                0.028366346641757145,
                0.019912477030973024,
                0.03887252837954606,
                0.054277500942535896,
                0.06269409287019,
                0.06717781831759528,
                0.09036882921093108,
                0.14241609366121277,
                0.19249284921736762
            ],
            # Freq = 3.25
            [
                0.051767757879611215,
                0.026184018017908716,
                0.01798269575514872,
                0.03585876866202793,
                0.050883993756935864,
                0.05905809657059736,
                0.06290391629661266,
                0.0843433476988759,
                0.1342556384552947,
                0.1832397225726294
            ],
            # Freq = 3.50
            [
                0.03703603083546259,
                0.01955514822668227,
                0.012465249956462972,
                0.026122913700549852,
                0.03808756801193894,
                0.043904960608503234,
                0.0453012068889763,
                0.06266007645302053,
                0.10238219444514533,
                0.13997587978912485
            ],
            # Freq = 3.75
            [
                0.04783661588799461,
                0.02607404893007669,
                0.016238848704644718,
                0.03439061049530874,
                0.05118753447432038,
                0.05982840025642346,
                0.06253212052161726,
                0.08535445003384791,
                0.13930157093815,
                0.19280838698563396
            ],
            # Freq = 4.00
            [
                0.051716455322842955,
                0.02974060049815688,
                0.01760327745085993,
                0.03824274598149952,
                0.05837276336037962,
                0.06846971391766143,
                0.07125041115569725,
                0.0992322240819615,
                0.16345717005774532,
                0.22705496464753924
            ],
            # Freq = 3.17
            # [
            #     0.054303981438637274,
            #     0.027098077776900303,
            #     0.01890404847704878,
            #     0.03749638995700268,
            #     0.052467609329834,
            #     0.06039880848858961,
            #     0.06483746080903162,
            #     0.08765202235749951,
            #     0.13798597436734042
            # ]
        ]
    )

    disp_bl_model = np.mean(disp_bl_model_list, axis= 0)

    fig, ax1 = plt.subplots( figsize=(15, 8) )
    # ax1 = plt.axes()
    ax1.plot(
        pos_bl_ref * 100,
        disp_bl_ref * 100,
        'b',
        linestyle = '--',
        label     = 'reference'
    )
    # Add shaded area around reference, progressively increase the range of the shaded area from the right to the left
    ax1.fill_between(
        pos_bl_ref * 100,
        (disp_bl_ref - 0.01) * 100 - 7 * (pos_bl_ref - 0.2)**2,
        (disp_bl_ref + 0.01) * 100 + 7 * (pos_bl_ref - 0.2)**2,
        color='b',
        alpha=0.1
    )

    ax1.plot(
        pos_bl * 100,
        disp_bl_model * 100,
        'r',
        marker = 'D',
        label  = 'simulation'
    )

    # Labels
    ax1.set_xlabel('Body position [%BL]', fontsize=15)
    ax1.set_ylabel('Lateral displacement [%BL]', fontsize=15)

    # Ticks
    ax1.tick_params(direction='in')

    # Horizontal ticks
    ax1.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax1.set_xticklabels([0, '', 20, '', 40, '', 60, '', 80, '', 100], fontsize=15)

    axt = ax1.secondary_xaxis('top')
    axt.tick_params(direction='in')
    axt.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    axt.set_xticklabels(['', '', '', '', '', '', '', '', '', '', ''], fontsize=15)

    # Vertical ticks
    ax1.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax1.set_yticklabels([0, '', 4, '', 8, '', 12, '', 16, '', 20])

    axr = ax1.secondary_yaxis('right')
    axr.tick_params(direction='in')
    axr.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    axr.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])

    # Add arrow and text for "rostral"
    arrow_caudal = patches.FancyArrowPatch((78, 18), (98, 18), mutation_scale=15, color='black')
    ax1.add_patch(arrow_caudal)
    ax1.text(23, 18, 'rostral', fontsize=15, va='center', ha='left')

    # Add arrow and text for "caudal"
    arrow_rostral = patches.FancyArrowPatch((22, 18), (2, 18), mutation_scale=15, color='black')
    ax1.add_patch(arrow_rostral)
    ax1.text(77, 18, 'caudal', fontsize=15, va='center', ha='right')

    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 20])
    ax1.legend( loc='lower right', fontsize=15)


    plt.show()


if __name__ == '__main__':
    main()
