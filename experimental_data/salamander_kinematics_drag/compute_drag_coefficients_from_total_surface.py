import numpy as np
import pickle

N_LINKS_AXIS = 9
LENGTH_AXIS  = 0.1

# Relative extensions of the links
LINKS_TOTAL_SURFACE = np.array(
    [
        3.44,
        4.59,
        5.04,
        4.23,
        2.77,
        2.41,
        2.11,
        1.92,
        1.21,
    ]
) * 1e-4

LINKS_CYLINDERS_SURFACE = np.array(
    [
        1.52,
        2.54,
        1.88,
        1.09,
        0.68,
        0.50,
        0.36,
        0.32,
    ]
) * 1e-4

def get_farms_drag_coefficients(
    coeff_x        = np.array( [ 0.07, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 ]),
    coeff_y        = np.array( [ 0.70, 0.70, 0.70, 0.70, 1.00, 1.00, 1.00, 1.00, 1.00 ]),
    coeff_z        = np.array( [ 1.00, 1.00, 1.00, 1.00, 0.70, 0.70, 0.70, 0.70, 0.70 ]),
    save_data      = True,
):
    ''' Compute the drag coefficients for the salamander model '''

    links_surfaces = np.zeros_like(LINKS_TOTAL_SURFACE)
    for link in range(N_LINKS_AXIS):

        link_surface = LINKS_TOTAL_SURFACE[link]
        if link > 0:
            link_surface -= LINKS_CYLINDERS_SURFACE[link-1] / 2

        if link < N_LINKS_AXIS - 1:
            link_surface -= LINKS_CYLINDERS_SURFACE[link] / 2

        links_surfaces[link] = link_surface

    # Overall drag coefficients
    water_density      = 1000.0
    water_surfaces_aux = 0.5 * water_density * links_surfaces

    overall_coeff_x = coeff_x * water_surfaces_aux
    overall_coeff_y = coeff_y * water_surfaces_aux
    overall_coeff_z = coeff_z * water_surfaces_aux

    overall_coeff = np.array([overall_coeff_x, overall_coeff_y, overall_coeff_z]).T

    if not save_data:
        return overall_coeff

    pre_path = 'experimental_data/salamander_kinematics_drag/'

    # Pickle file
    with open(f'{pre_path}/drag_coefficients.pickle', 'wb') as outfile:
        pickle.dump(
            {
                'links_surfaces' : links_surfaces,

                'water_surfaces_aux' : water_surfaces_aux,

                'coeff_x' : coeff_x,
                'coeff_y' : coeff_y,
                'coeff_z' : coeff_z,

                'overall_coeff_x' : overall_coeff_x,
                'overall_coeff_y' : overall_coeff_y,
                'overall_coeff_z' : overall_coeff_z,

                'overall_coeff' : overall_coeff,
            },
            outfile
        )

    # TXT file
    get_str_vec = lambda vec : ', '.join([f'{val:.5f}' for val in vec])
    with open(f'{pre_path}/drag_coefficients.txt', 'w') as outfile:

        outfile.write('\n')
        outfile.write(f'links_surfaces = np.array( [ {get_str_vec(links_surfaces)} ] )\n')

        outfile.write('\n')
        outfile.write(f'water_surfaces_aux = np.array( [ {get_str_vec(water_surfaces_aux)} ] )\n')

        outfile.write('\n')
        outfile.write(f'coeff_x = np.array( [ {get_str_vec(coeff_x)} ] )\n')
        outfile.write(f'coeff_y = np.array( [ {get_str_vec(coeff_y)} ] )\n')
        outfile.write(f'coeff_z = np.array( [ {get_str_vec(coeff_z)} ] )\n')

        outfile.write('\n')
        outfile.write(f'overall_coeff_x = np.array( [ {get_str_vec(overall_coeff_x)} ] )\n')
        outfile.write(f'overall_coeff_y = np.array( [ {get_str_vec(overall_coeff_y)} ] )\n')
        outfile.write(f'overall_coeff_z = np.array( [ {get_str_vec(overall_coeff_z)} ] )\n')

        outfile.write('\n')
        outfile.write('Overall coefficients: \n')
        outfile.write(
            '\n'.join(
                [
                    f'-{drag_coeff:.5f}'
                    for drag_coeff in overall_coeff.reshape(-1)
                ]
            )
        )

    return overall_coeff

if __name__ == '__main__':
    get_farms_drag_coefficients()