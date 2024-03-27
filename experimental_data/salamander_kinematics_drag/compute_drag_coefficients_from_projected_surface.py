import numpy as np
import pickle

N_LINKS_AXIS = 9
LENGTH_AXIS  = 0.1

# Relative extensions of the links
LINKS_X_EXTENSIONS_REL = np.array(
    [
        0.11,
        0.10,
        0.11,
        0.12,
        0.12,
        0.12,
        0.11,
        0.11,
        0.11
    ]
)
LINKS_Y_EXTENSIONS_REL = np.array(
    [
        0.08,
        0.10,
        0.10,
        0.09,
        0.06,
        0.05,
        0.04,
        0.03,
        0.03
    ]
)
LINKS_Z_EXTENSIONS_REL = np.array(
    [
        0.07,
        0.08,
        0.08,
        0.08,
        0.07,
        0.06,
        0.06,
        0.06,
        0.06
    ]
)

def get_farms_drag_coefficients_original():
    ''' Return original drag coefficients '''
    return np.array([-0.003, -0.06, -0.06] * N_LINKS_AXIS).reshape(-1, 3)

def get_farms_drag_coefficients(
    coeff_x        = 0.5 * np.ones(N_LINKS_AXIS),
    coeff_y        = 0.5 * np.ones(N_LINKS_AXIS),
    coeff_z        = 0.5 * np.ones(N_LINKS_AXIS),
    scaling_x_body = 0.15,
    scaling_y_body = 1.00,
    scaling_z_body = 1.00,
    save_data      = True,
):
    ''' Compute the drag coefficients for the salamander model '''
    links_x_extensions = LINKS_X_EXTENSIONS_REL * LENGTH_AXIS
    links_y_extensions = LINKS_Y_EXTENSIONS_REL * LENGTH_AXIS
    links_z_extensions = LINKS_Z_EXTENSIONS_REL * LENGTH_AXIS

    # Ellipsoidal face
    links_yz_surface = np.pi * links_x_extensions * links_y_extensions / 4

    # Rectangular face
    links_xz_surface = links_x_extensions * links_z_extensions
    links_xy_surface = links_x_extensions * links_y_extensions

    # Triangular face
    links_xz_surface[-1] *= 0.5

    # Drag coefficients
    coeff_x[1:] *= scaling_x_body
    coeff_y[1:] *= scaling_y_body
    coeff_z[1:] *= scaling_z_body

    # Overall drag coefficients
    water_density = 1000.0

    water_surface2_x_aux = 0.5 * water_density * links_yz_surface
    water_surface2_y_aux = 0.5 * water_density * links_xz_surface
    water_surface2_z_aux = 0.5 * water_density * links_xy_surface

    overall_coeff_x = coeff_x * water_surface2_x_aux
    overall_coeff_y = coeff_y * water_surface2_y_aux
    overall_coeff_z = coeff_z * water_surface2_z_aux

    overall_coeff = np.array([overall_coeff_x, overall_coeff_y, overall_coeff_z]).T

    if not save_data:
        return overall_coeff

    pre_path = 'experimental_data/salamander_kinematics_drag/'

    # Pickle file
    with open(f'{pre_path}/drag_coefficients.pickle', 'wb') as outfile:
        pickle.dump(
            {
                'links_x_extensions' : links_x_extensions,
                'links_y_extensions' : links_y_extensions,
                'links_z_extensions' : links_z_extensions,

                'links_xy_surface' : links_xy_surface,
                'links_xz_surface' : links_xz_surface,
                'links_yz_surface' : links_yz_surface,

                'water_surface2_x' : water_surface2_x_aux,
                'water_surface2_y' : water_surface2_x_aux,
                'water_surface2_z' : water_surface2_x_aux,

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
        outfile.write(f'links_x_extensions = [ {get_str_vec(links_x_extensions)} ]\n')
        outfile.write(f'links_y_extensions = [ {get_str_vec(links_y_extensions)} ]\n')
        outfile.write(f'links_z_extensions = [ {get_str_vec(links_z_extensions)} ]\n')

        outfile.write('\n')
        outfile.write(f'links_xy_surface = [ {get_str_vec(links_xy_surface)} ]\n')
        outfile.write(f'links_xz_surface = [ {get_str_vec(links_xz_surface)} ]\n')
        outfile.write(f'links_yz_surface = [ {get_str_vec(links_yz_surface)} ]\n')

        outfile.write('\n')
        outfile.write(f'water_surface2_x = [ {get_str_vec(water_surface2_x_aux)} ]\n')
        outfile.write(f'water_surface2_y = [ {get_str_vec(water_surface2_x_aux)} ]\n')
        outfile.write(f'water_surface2_z = [ {get_str_vec(water_surface2_x_aux)} ]\n')

        outfile.write('\n')
        outfile.write(f'coeff_x = [ {get_str_vec(coeff_x)} ]\n')
        outfile.write(f'coeff_y = [ {get_str_vec(coeff_y)} ]\n')
        outfile.write(f'coeff_z = [ {get_str_vec(coeff_z)} ]\n')

        outfile.write('\n')
        outfile.write(f'overall_coeff_x = [ {get_str_vec(overall_coeff_x)} ]\n')
        outfile.write(f'overall_coeff_y = [ {get_str_vec(overall_coeff_y)} ]\n')
        outfile.write(f'overall_coeff_z = [ {get_str_vec(overall_coeff_z)} ]\n')

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