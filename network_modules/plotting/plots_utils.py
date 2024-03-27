import os
import logging
from typing import Union
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, FFMpegWriter

# SAVING
def _save_figure(
    figure     : Figure,
    folder_path: str,
    **kwargs
) -> None:
    """ Save figure """

    figname_pre = figure._label.replace(' ', '_').replace('.', 'dot')
    if figname_pre == '':
        figname_pre = kwargs.pop('fig_label', 'animation')

    for extension in kwargs.pop('extensions', ['pdf']):
        figname  = f'{figname_pre}.{extension}'
        filename = f'{folder_path}/{figname}'

        logging.info(f'Saving figure {figname} to {filename}')
        figure.savefig(filename)

def _save_animation(
        figure     : Figure,
        anim       : FuncAnimation,
        folder_path: str,
        **kwargs
    ) -> None:
    """ Save animation """

    figname = figure._label.replace(' ', '_').replace('.', 'dot')
    if figname == '':
        figname = kwargs.pop('fig_label', 'animation')

    figname = f'{figname}.mp4'
    filename = f'{folder_path}/{figname}'

    logging.info('Saving animation %s to %s', figname, filename)
    anim.save(
        filename,
        writer = FFMpegWriter(fps = 20)
    )

def _save_plots(
    figures_dict: dict[str, Union[Figure, list[Figure, FuncAnimation]]],
    folder_path : str
) -> None:
    ''' Save results of the simulation '''

    for fig_label, fig in figures_dict.items():
        if not isinstance(fig, list):
            # Image
            _save_figure(
                figure      = fig,
                folder_path = folder_path,
                fig_label   = fig_label,
            )
        else:
            # Animation
            _save_animation(
                figure      = fig[0],
                anim        = fig[1],
                folder_path = folder_path,
                fig_label   = fig_label,
            )

def save_prompt(
    figures_dict: dict[str, Union[Figure, list[Figure, FuncAnimation]]],
    folder_path : str,
) -> tuple[bool, str]:
    ''' Prompt user to choose whether to save the figures '''

    saveprompt = input('Save images? [y/n]  ')

    if not saveprompt in ('y','Y','1', 'yes'):
        return False, ''

    # Tag
    file_tag = input('''Tag to append to the figures ['']: ''')

    figures_path = f'{folder_path}/{file_tag}'
    os.makedirs(figures_path, exist_ok=True)

    # Save
    _save_plots(
        figures_dict = figures_dict,
        folder_path  = figures_path,
    )

    return True, figures_path

