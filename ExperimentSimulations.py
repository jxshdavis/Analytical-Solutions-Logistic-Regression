import seaborn as sns
import MultiSim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class ExperimentSimulations:
    """
    Easily create a simulation to compare performance of the proposed Analytical Solution to the usual MLE solution
    (sklearn.linear_model LogisticRegression with lib-linear solver):
    - Specify the number of regressors and the number of levels each regressor will have
    - Specify the number of trials to perform
    - Automatically performs each simulation with the MultiSim Class
    - Comparison is done through the MSE between the predicted beta's and the "true" beta which was used to generate the
      data response
    - Can use the plotting functions to generate nice plots which show the comparison of the errors as well as the time
      it took to fit the models.

    If the number of regressors is 1, we fit and compare both of the proposed analytical solutions. Otherwise, we
    only perform the multi-regressor solution.
    """
    def __init__(self, num_trials, num_regressors, num_levels, min_number_obs, max_number_obs, number_observations_sizes, logspace = True, use_mean = True):
        """
        :param num_trials: the number of simulations to run - the final plots will show the averages over all trials.
        :param num_regressors: the number of regressors the generated design matrix will have
        :param num_levels: the number of possible levels each regressor will have
        :param min_number_obs: the log of the smallest number of observations (rows of the design matrix) to generate.
        :param max_number_obs: the log of the largest number of observations (rows of the design matrix) to generate.
        :param number_observations_sizes: the number of different sample sizes to test.
        :param logspace: default is True. If False, will linearly space the observation sizes between min_number_obs and
        max_number_obs
        :param use_mean: default is true. If false, the plots will show Median_squared_error as opposed to mean squared
        error. Can be nice to set as False if there is a particular part of the simulation which has extreme outliers in
        the errors.
        """
        # set true to use means and set false to use medians
        self._USE_MEAN = False
        self._num_regressors = num_regressors
        self._num_levels = num_levels
        self._obs_counts = [int(x) for x in np.logspace(start=min_number_obs, stop=max_number_obs, num=number_observations_sizes)]
        self._num_trials = num_trials

        self._gamma_errors = []
        self._lib_lin_errors = []
        self._simple_errors = []

        self._analytical_times = []
        self._iterative_times = []

        self._color1 = "red"
        self._color2 = "green"
        self._color3 = "blue"

        self._x_boxplot = None
        self._obs_counts_strings = None

        # run the simulation
        self.run_sim()

    def run_sim(self):
        """
        Runs the actual simulation. Saves the errors from each trial as well as fitting times.
        :return: None
        """
        num_trials = self._num_trials
        obs_counts = self._obs_counts
        num_levels = self._num_levels
        num_regressors = self._num_regressors
        USE_MEAN = self._USE_MEAN
        for trial in range(num_trials):
            print(str(round((trial + 1) / num_trials * 100, 3)) + " percent finished.")
            trial_gamma_error = []
            trial_lib_lin_error = []
            trial_simple_error = []

            trial_analytical_time = []
            trial_iterative_time = []

            for num_obs in obs_counts:
                while True:
                    try:
                        sim1 = MultiSim.Simulation(num_observations=num_obs, num_regressors=num_regressors,
                                                   num_levels=num_levels)

                        true, gamma, lib_lin_sol = sim1.get_parameters()
                        analytical_time, iterative_time = sim1.get_times()
                        if num_regressors == 1:
                            simple_params = sim1.get_simple_parameters()

                        # compute sample errors
                        if USE_MEAN:
                            trial_gamma_error.append(((true - gamma) ** 2).mean(axis=0))
                            trial_lib_lin_error.append(((true - lib_lin_sol) ** 2).mean(axis=0))
                            if num_regressors == 1:
                                trial_simple_error.append(((true - simple_params) ** 2).mean(axis=0))
                        else:
                            trial_gamma_error.append(np.median((true - gamma) ** 2, axis=0))
                            trial_lib_lin_error.append(np.median((true - lib_lin_sol) ** 2, axis=0))
                            if num_regressors == 1:
                                trial_simple_error.append(np.median((true - simple_params) ** 2, axis=0))

                        # save experiment error
                        trial_analytical_time.append(analytical_time)
                        trial_iterative_time.append(iterative_time)

                        break  # If no error occurred, exit the loop and move to the next iteration
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        print("Retrying the iteration...")

            # save trial errors and times
            self._gamma_errors.append(trial_gamma_error)
            self._lib_lin_errors.append(trial_lib_lin_error)
            self._simple_errors.append(trial_simple_error)

            self._analytical_times.append(trial_analytical_time)
            self._iterative_times.append(trial_iterative_time)

        # convert errors and times to numpy arrays for plotting
        self._gamma_errors = np.array(self._gamma_errors)
        self._lib_lin_errors = np.array(self._lib_lin_errors)
        self._simple_errors = np.array(self._simple_errors)

        self._analytical_times = np.array(self._analytical_times)
        self._iterative_times = np.array(self._iterative_times)

    def plot_errors(self):
        """
        Saves the plot displaying information on errors obtained from the simulation.
        :return: None
        """
        num_trials = self._num_trials
        obs_counts = self._obs_counts
        num_levels = self._num_levels
        num_regressors = self._num_regressors
        USE_MEAN = self._USE_MEAN
        gamma_errors = self._gamma_errors
        lib_lin_errors = self._lib_lin_errors
        simple_errors = self._simple_errors
        # plot results
        sns.set_style('whitegrid')
        n_colors = 10
        custom_palette = sns.color_palette("viridis", n_colors)

        # color1 = custom_palette[int(0)]
        # color2 = custom_palette[int(3 * n_colors / 9)]
        # color3 = custom_palette[int(6 * n_colors / 9)]

        color1 = sns.color_palette("husl", 8)[0]
        color2 = sns.color_palette("husl", 8)[3]
        color3 = sns.color_palette("husl", 8)[5]

        self._color1 = color1
        self._color2 = color2
        self._color3 = color3

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        if num_regressors == 1:
            if USE_MEAN:
                max_value = max(np.max(np.mean(gamma_errors, axis=0)), np.max(np.mean(lib_lin_errors, axis=0)),
                                np.max(np.mean(simple_errors, axis=0)))
            else:
                max_value = max(np.max(np.median(simple_errors, axis=0)), np.max(np.median(gamma_errors, axis=0)),
                                np.max(np.median(lib_lin_errors, axis=0)))
        else:
            if USE_MEAN:
                max_value = max(np.max(np.mean(gamma_errors, axis=0)), np.max(np.mean(lib_lin_errors, axis=0)))
            else:
                max_value = max(np.max(np.median(gamma_errors, axis=0)),
                                np.max(np.median(lib_lin_errors, axis=0)))

        ax[0].set_yscale('log')  # Set y-axis to logarithmic scale
        # ax[0].set_ylim(0, 1.5 * max_value)  # Set y-axis limit to 130% of the maximum value


        if USE_MEAN:
            title = "Average Error over " + str(num_trials) + " trials\nvs. Number of Observations"
        else:
            title = "Median Error over " + str(num_trials) + " trials\nvs. Number of Observations"

        ax[0].text(x=0.5, y=1.1, s=title, fontsize=12, weight='bold', ha='center', va='bottom', transform=ax[0].transAxes)
        ax[0].text(x=0.5, y=1.05,
                   s="Data Randomly Generated with " + str(num_regressors) + " categorical regressors each with " + str(
                       num_levels) + " levels", fontsize=8, alpha=0.75, ha='center', va='bottom', transform=ax[0].transAxes)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
        # Create an array to hold the boxplot data
        box_data_gamma = []
        box_data_lib_lin = []

        for i, obs_count in enumerate(obs_counts):
            # Repeat the gamma_errors and lib_lin_errors values for each boxplot
            box_data_gamma.extend(gamma_errors[:, i])
            box_data_lib_lin.extend(lib_lin_errors[:, i])

        # Plot the boxplots

        x_boxplot = []
        obs_counts_strings = [str("{:.1e}".format(x)) for x in obs_counts]
        self._obs_counts_strings = obs_counts_strings
        for num_obs in obs_counts_strings:
            x_boxplot += [num_obs] * num_trials

        self._x_boxplot = x_boxplot
        y_boxplot_gamma = gamma_errors.T.flatten().tolist()
        y_boxplot_lib_lin = lib_lin_errors.T.flatten().tolist()
        y_boxplot_simple = simple_errors.T.flatten().tolist()

        df_gamma_box = pd.DataFrame({"x": np.array(x_boxplot), "y": np.array(y_boxplot_gamma)})
        df_lib_lin_box = pd.DataFrame({"x": np.array(x_boxplot), "y": np.array(y_boxplot_lib_lin)})

        if num_regressors == 1:
            df_simple_box = pd.DataFrame({"x": np.array(x_boxplot), "y": np.array(y_boxplot_simple)})
            sns.boxplot(x='x', y='y', ax=ax[0], data=df_simple_box, width=.1, color=color3, saturation=.5,
                        boxprops=dict(alpha=.4),
                        whiskerprops=dict(color=color3), capprops=dict(color=color3), medianprops=dict(color=color3),
                        flierprops=dict(markerfacecolor=color3, markeredgecolor=color3))

        sns.boxplot(x='x', y='y', ax=ax[0], data=df_gamma_box, width=.3, color=color1, saturation=.5,
                    boxprops=dict(alpha=.3),
                    whiskerprops=dict(color=color1), capprops=dict(color=color1), medianprops=dict(color=color1),
                    flierprops=dict(markerfacecolor=color1, markeredgecolor=color1))

        sns.boxplot(x='x', y='y', ax=ax[0], data=df_lib_lin_box, width=.2, color=color2, saturation=.5, boxprops=dict(alpha=.4),
                    whiskerprops=dict(color=color2), capprops=dict(color=color2), medianprops=dict(color=color2),
                    flierprops=dict(markerfacecolor=color2, markeredgecolor=color2))


        if USE_MEAN:
            y_gam = np.mean(gamma_errors, axis=0)
            y_iter = np.mean(lib_lin_errors, axis=0)
            y_simple = np.mean(simple_errors, axis=0)
        else:
            y_gam = np.median(gamma_errors, axis=0)
            y_iter = np.median(lib_lin_errors, axis=0)
            y_simple = np.median(simple_errors, axis=0)

        sns.lineplot(x=obs_counts_strings, y=y_gam, color=color1, label='Analytic', linewidth=2, ax=ax[0])
        sns.lineplot(x=obs_counts_strings, y=y_iter, color=color2, label='Standard Iterative', linewidth=2, ax=ax[0])
        if num_regressors == 1:
            sns.lineplot(x=obs_counts_strings, y=y_simple, color=color3, label='Simple Analytic MLE', linewidth=2, ax=ax[0])

        ax[0].set_xlabel("Number of Observations")
        ax[0].set_ylabel("Average MSE (Predicted Beta vs Actual Beta)")

        h, l = ax[0].get_legend_handles_labels()
        ax[0].legend(h[:4], l[:4], bbox_to_anchor=(1.05, 1), loc=2)

        # difference in MSE plot

        # Take the difference, leaving 'x' column unchanged
        df_diff_analytic = df_gamma_box.copy()  # Create a copy of the 'df_gamma_box' DataFrame
        df_diff_analytic['y'] = df_gamma_box['y'].subtract(df_lib_lin_box['y'])


        if num_regressors == 1:
            df_diff_simple = df_simple_box.copy()
            df_diff_simple['y'] = df_simple_box['y'].subtract(df_lib_lin_box['y'])
            sns.boxplot(x='x', y='y', ax=ax[1], data=df_diff_simple, width=.2, boxprops=dict(alpha=0.5), color=color3,
                        whiskerprops=dict(color=color3), capprops=dict(color=color3), medianprops=dict(color=color3),
                        flierprops=dict(markerfacecolor=color3, markeredgecolor=color3))

        sns.boxplot(x='x', y='y', ax=ax[1], data=df_diff_analytic, width=.2, boxprops=dict(alpha=0.5), color=color1,
                    whiskerprops=dict(color=color1), capprops=dict(color=color1), medianprops=dict(color=color1),
                    flierprops=dict(markerfacecolor=color1, markeredgecolor=color1))

        ax[1].axhline(y=0, linestyle='--', color="black")
        ax[1].set_xlabel("Number of Observations")

        y_lab = "Difference in Average MSE (Predicted Beta vs Actual Beta)"
        if not USE_MEAN:
            y_lab = "Difference in Average Median Squared Error (Predicted Beta vs Actual Beta)"

        ax[1].set_ylabel("Difference in Average MSE (Predicted Beta vs Actual Beta)")
        ax[1].set_yscale('symlog')

        if USE_MEAN:
            title2 = "Difference in Average Error over " + str(num_trials) + " trials\nvs. Number of Observations"
        else:
            title2 = "Difference in Median Error over " + str(num_trials) + " trials\nvs. Number of Observations"

        ax[1].text(x=1.95, y=1.1, s=title2, fontsize=12, weight='bold', ha='center', va='bottom', transform=ax[0].transAxes)
        ax[1].text(x=1.95, y=1.03,
                   s="Data Randomly Generated with " + str(num_regressors) + " categorical regressors each with " + str(
                       num_levels) + " levels.\nWe compare the error of the analytic solution to the error of iterative solution.",
                   fontsize=8, alpha=0.75, ha='center', va='bottom', transform=ax[0].transAxes)

        fig.tight_layout()  # Adjust plot layout to prevent overlapping
        plot_name = "Sim_MSE_"+str(self._num_regressors)+"regressors_"+str(self._num_levels)+"levels_"+str(self._num_trials)+"trials.png"
        plt.savefig(plot_name, dpi=300)

        plt.close()

    def plot_times(self):
        """
        Saves the plot displaying information on times obtained from the simulation.
        :return: None
        """
        color1 = self._color1
        color2 = self._color2
        color3 = self._color3

        x_boxplot = self._x_boxplot

        num_trials = self._num_trials
        num_regressors = self._num_regressors
        num_levels = self._num_levels
        obs_counts_strings = self._obs_counts_strings
        analytical_times = self._analytical_times
        iterative_times = self._iterative_times

        fig_time, ax_time = plt.subplots(figsize=(7, 5))

        y_boxplot_analytical_time = np.array(analytical_times).T.flatten().tolist()
        y_boxplot_iterative_time = np.array(iterative_times).T.flatten().tolist()

        df_a_times_box = pd.DataFrame({"x": np.array(x_boxplot), "y": np.array(y_boxplot_analytical_time)})
        df_i_times_box = pd.DataFrame({"x": np.array(x_boxplot), "y": np.array(y_boxplot_iterative_time)})

        sns.boxplot(x='x', y='y', ax=ax_time, data=df_a_times_box, width=.3, color=color1, saturation=.5,
                    boxprops=dict(alpha=.3),
                    whiskerprops=dict(color=color1), capprops=dict(color=color1), medianprops=dict(color=color1),
                    flierprops=dict(markerfacecolor=color1, markeredgecolor=color1))

        sns.boxplot(x='x', y='y', ax=ax_time, data=df_i_times_box, width=.2, color=color2, saturation=.5,
                    boxprops=dict(alpha=.4),
                    whiskerprops=dict(color=color2), capprops=dict(color=color2), medianprops=dict(color=color2),
                    flierprops=dict(markerfacecolor=color2, markeredgecolor=color2))

        sns.lineplot(x=obs_counts_strings, y=np.mean(analytical_times, axis=0), color=color1, label='Analytic', linewidth=2,
                     ax=ax_time)
        sns.lineplot(x=obs_counts_strings, y=np.mean(iterative_times, axis=0), color=color2, label='Standard Iterative',
                     linewidth=2, ax=ax_time)

        ax_time.set_xlabel("Number of Observations")
        ax_time.set_ylabel("Average Time Spent Fitting Model")

        h, l = ax_time.get_legend_handles_labels()

        ax_time.legend(h[:4], l[:4], bbox_to_anchor=(1.05, 1), loc=2)

        title_time = "Average Time to Fit Models over " + str(num_trials) + " trials\nvs. Number of Observations"

        ax_time.text(x=0.5, y=1.1, s=title_time, fontsize=12, weight='bold', ha='center', va='bottom',
                     transform=ax_time.transAxes)
        ax_time.text(x=0.5, y=1.05,
                     s="Data Randomly Generated with " + str(num_regressors) + " categorical regressors each with " + str(
                         num_levels) + " levels", fontsize=8, alpha=0.75, ha='center', va='bottom', transform=ax_time.transAxes)
        ax_time.set_yscale('log')
        fig_time.tight_layout()  # Adjust plot layout to prevent overlapping
        plot_name = "Fitting_Times_" + str(self._num_regressors) + "regressors_" + str(self._num_levels) + "levels_" + str(
            self._num_trials) + "trials.png"
        plt.savefig(plot_name, dpi=300)  # Save the plot with higher DPI


