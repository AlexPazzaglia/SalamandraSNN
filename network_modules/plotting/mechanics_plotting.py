'''
Plotting of the mechanical simulation
'''

import os

import numpy as np
import matplotlib.pyplot as plt
import network_modules.plotting.plots_mech as mech_plt

from network_modules.performance.mechanics_performance import MechPerformance
from farms_mujoco.sensors.camera import save_video

from network_modules.performance import signal_processor_mech_plot

# FIGURE PARAMETERS
SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.rc(  'font', size      = SMALL_SIZE )  # controls default text sizes
plt.rc(  'axes', titlesize = SMALL_SIZE )  # fontsize of the axes title
plt.rc(  'axes', labelsize = BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc( 'xtick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc( 'ytick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize  = SMALL_SIZE )  # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title

class MechPlotting(MechPerformance):
    '''
    Class used plot the results of mechanical simulations
    '''

    # POST PROCESSING
    def simulation_post_processing(self) -> dict[str, float]:
        '''
        Post-processing for the mechanical simulation
        Optionally save the video of the simulation
        '''
        mech_metrics = super().simulation_post_processing()

        # Saving video
        if self.mech_sim_options.video:
            os.makedirs(self.results_data_folder, exist_ok=True)
            save_video(
                camera      = self.camera,
                video_path  = f'{self.results_data_folder}/{self.mech_sim_options.video_name}',
                iteration   = self.sim.iteration,
            )

        return mech_metrics

    # ALL RESULTS
    def simulation_plots(self, figures_dict = None) -> None:
        ''' Plots showing the network's behavior '''

        # Change matplotlib logging level to avoid undesired messages
        plt.set_loglevel("warning")

        # Plot
        self._plot_farms_simulation(figures_dict)

    # ------ [ PLOTTING ] ------
    def _plot_farms_simulation(
        self,
        figures_dict   : dict = None,
        load_from_file : bool = False,
    ) -> None:
        '''
        When included, plot the movemente generated in the farms model
        '''
        if not self.snn_network.params.monitor.farms_simulation['active']:
            return

        self.load_mechanical_simulation_data(load_from_file)

        figures_dict = self.snn_network.figures if figures_dict is None else figures_dict

        # Plots
        self._plot_joints_angles(figures_dict)
        self._plot_com_trajectory(figures_dict)
        self._plot_fitted_trajectory(figures_dict)

        # Animations
        self._animate_links_trajectory(figures_dict)

    # \----- [ Mechanical simulation ] ------

    # ------ [ Plots ] ------
    def _plot_joints_angles(
        self,
        figures_dict: dict[str, plt.Figure] = None,
    ):
        ''' Plot joint angles evolution '''

        plotpars = self.snn_network.params.monitor.farms_simulation['plotpars']
        if not plotpars['joint_angles']:
            return

        joints_positions = np.array( self.sim_res_data.sensors.joints.positions_all() )

        if not joints_positions[-1].any():
            joints_positions  = joints_positions[:-1]

        timestep     = self.sim_res_data.timestep
        n_iterations = joints_positions.shape[0]
        times        = np.arange(0, timestep*n_iterations, timestep)

        fig_ja_dict = mech_plt.plot_joints_angles(
            times         = times,
            joints_angles = np.rad2deg( joints_positions ),
            params        = self.sim_res_pars,
        )
        figures_dict.update(fig_ja_dict)

        return

    def _plot_com_trajectory(
        self,
        figures_dict: dict[str, plt.Figure] = None,
    ):
        ''' Plot COM trajectory '''

        plotpars = self.snn_network.params.monitor.farms_simulation['plotpars']
        if not plotpars['com_trajectory']:
            return

        links_positions  = np.array( self.sim_res_data.sensors.links.urdf_positions() )

        if not links_positions[-1].any():
            links_positions  = links_positions[:-1]

        timestep     = self.sim_res_data.timestep
        n_iterations = links_positions.shape[0]
        times        = np.arange(0, timestep*n_iterations, timestep)

        axial_joints  = self.sim_res_pars['morphology']['n_joints_body']
        com_positions = np.mean( links_positions[:, :axial_joints], axis= 1 )

        fig_ht = mech_plt.plot_com_trajectory(
            times         = times,
            com_positions = com_positions,
        )
        figures_dict.update(fig_ht)

        return

    def _plot_fitted_trajectory(
        self,
        figures_dict: dict[str, plt.Figure] = None,
    ):
        ''' Plot trajectory'''

        plotpars = self.snn_network.params.monitor.farms_simulation['plotpars']
        if not plotpars['trajectory_fit']:
            return

        disp_data = self.all_metrics_data['links_displacements_data']
        figures_dict['fig_traj_fit'] = signal_processor_mech_plot.plot_trajectory_fit(
            links_pos_xy               = disp_data['links_positions'],
            direction_fwd              = disp_data['direction_fwd'],
            direction_left             = disp_data['direction_left'],
            quadratic_fit_coefficients = disp_data['quadratic_fit_coefficients'],
        )

    # \----- [ Plots ] ------

    # ------ [ Animations ] ------
    def _animate_links_trajectory(
        self,
        figures_dict: dict[str, plt.Figure] = None,
    ):
        ''' Animate links trajectory '''

        plotpars = self.snn_network.params.monitor.farms_simulation['plotpars']
        if not plotpars['animate']:
            return

        links_positions  = np.array( self.sim_res_data.sensors.links.urdf_positions() )

        if not links_positions[-1].any():
            links_positions  = links_positions[:-1]

        timestep     = self.sim_res_data.timestep
        n_iterations = links_positions.shape[0]
        times        = np.arange(0, timestep*n_iterations, timestep)

        fig_lt = plt.figure('Link trajectory animation')
        anim_lt = mech_plt.animate_links_trajectory(
            fig             = fig_lt,
            times           = times,
            links_positions = links_positions,
            params          = self.sim_res_pars,
        )
        figures_dict['fig_lt'] = [fig_lt, anim_lt]

        return

    # \----- [ Animations ] ------

    # ------ [ Data cleaning ] ------
    def clear_figures(self) -> None:
        ''' Clear figures '''
        plt.close('all')

    # \----- [ Data cleaning ] ------