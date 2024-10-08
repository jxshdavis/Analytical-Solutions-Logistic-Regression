import seaborn as sns
import Simulation
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

    def __init__(self,
                 num_trials,
                 num_regressors, num_levels,
                 log_min_x,
                 log_max_x,
                 number_x_ticks,
                 max_iter=100,
                 mixing_percentage=0,
                 large_samples_size=30,
                 num_obs=10**3,
                 nudge=10 ** (-5),
                 beta_range=[-10, 10],
                 lamb=0,
                 penalty="l2",
                 logspace=True,
                 use_mean=True,
                 drop_under_count=[0]):
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
        self._USE_MEAN = use_mean
        self._mixing_percentage = mixing_percentage
        self._large_samples_size = large_samples_size
        self._num_regressors = num_regressors
        self._num_levels = num_levels
        self._num_obs = num_obs
        self._x_ax_counts = [int(x) for x in
                             np.logspace(start=log_min_x, stop=log_max_x, num=number_x_ticks)]
        self._num_trials = num_trials
        self._nudge = nudge
        self._gamma_errors = []
        self._lib_lin_errors = []
        self._simple_errors = []
        self._beta_range = beta_range
        self._analytical_transform_times = []
        self._analytical_fit_times = []
        self._iterative_times = []
        self._remaining_mixed_proportions = []
        self._lambda = lamb
        self._penalty = penalty
        self._color1 = "red"
        self._color2 = "green"
        self._color3 = "blue"
        self._x_boxplot = None
        self._x_ax_counts_strings = None
        self._drop_under_count = drop_under_count
        self.max_iter = max_iter

    def run_observation_sim(self):
        self.run_sim("observations")

    def run_regressors_sim(self):
        self.run_sim("regressors")

    def run_regressors_sim(self):
        self.run_sim("regressors")

    def run_sim(self, independent_var):
        """
        Runs the actual simulation. Saves the errors from each trial as well as fitting times as plots.
        :return: None
        """

        # grab parameter values for easier to read code
        num_trials = self._num_trials
        x_ax_counts = self._x_ax_counts
        num_levels = self._num_levels
        num_regressors = self._num_regressors
        num_obs = self._num_obs
        USE_MEAN = self._USE_MEAN

        # perform each trial

        mixed_proportions = []
        for count in x_ax_counts:
            N_prime = int(count * self._mixing_percentage /
                          self._large_samples_size)

            a = 0
            b = 2**num_regressors
            if b - a + 1 < N_prime:
                N_prime = b-a

            N = int(count * (1-self._mixing_percentage))

            mixed_proportions.append(N/(N+N_prime*self._large_samples_size))

        for trial in range(num_trials):
            # print percent of trials left
            print(str(round((trial) / num_trials * 100, 3)) + " percent finished.")

            # initialize empty lists to store errors and times for the trial
            trial_gamma_error, trial_lib_lin_error, trial_simple_error, trial_analytical_transform_time, \
                trial_analytical_fit_time, trial_iterative_time = [], [], [], [], [], []

            # intialize lists to store mixing proportions after deletion

            trial_remaining_mixed_proportions = []

            # loop through the different levels of the independent variable
            for count in x_ax_counts:

                # set independent variable
                if independent_var == "observations":
                    num_obs = count
                elif independent_var == "regressors":
                    num_regressors = count
                elif independent_var == "levels":
                    num_levels = count

                # save mixing information for the fixing plot

                # if our data is not diverse enough for the solver we will try again with new data, hence the while loop
                while True:
                    try:
                        # run the simulation on the current inputs

                        sim1 = Simulation.Simulation(num_observations=num_obs, num_regressors=num_regressors,
                                                     num_levels=num_levels, nudge=self._nudge,
                                                     beta_range=self._beta_range,
                                                     lamb=self._lambda, penalty=self._penalty,
                                                     mixing_percentage=self._mixing_percentage,
                                                     large_samples_size=self._large_samples_size,
                                                     drop_under_count=self._drop_under_count,
                                                     max_iter = self.max_iter)

                        analytical_model = sim1.get_analytical_model()
                        row_counts, remaining_num_batch_samples = analytical_model.get_mixing_info()
                        total_samples = row_counts.sum()

                        # N = total_samples - remaining_num_batch_samples * self._large_samples_size
                        N = total_samples

                        trial_remaining_mixed_proportions.append(
                            N / total_samples)

                        # get the true and estimated parameter values
                        true, gammas, lib_lin_sol = sim1.get_parameters()

                        number_of_drop_row = len(self._drop_under_count)

                        multi_true = np.array([true]*number_of_drop_row)
                        # gamma = np.array(gamma)

                        # get the model fitting times
                        analytical_transform_time, analytical_fit_time, iterative_time = sim1.get_times()

                        # if a single repressor get the analytic MLE estimations
                        if num_regressors == 1:
                            simple_params = sim1.get_simple_parameters()

                        # compute sample errors with mean or median
                        if USE_MEAN:

                            # paused work here. Now trial_gamma_error contains a list of
                            #  mse of gamma and true for each drop count
                            trial_gamma_error.append(
                                ((multi_true - gammas) ** 2).mean(axis=1))
                            trial_lib_lin_error.append(
                                ((true - lib_lin_sol) ** 2).mean(axis=0))
                            if num_regressors == 1:
                                trial_simple_error.append(
                                    ((true - simple_params) ** 2).mean(axis=0))
                        else:
                            trial_gamma_error.append(
                                np.median((multi_true - gammas) ** 2, axis=1))
                            trial_lib_lin_error.append(
                                np.median((true - lib_lin_sol) ** 2, axis=0))
                            if num_regressors == 1:
                                trial_simple_error.append(
                                    np.median((true - simple_params) ** 2, axis=0))

                        # save experiment error
                        trial_analytical_transform_time.append(
                            analytical_transform_time)
                        trial_analytical_fit_time.append(analytical_fit_time)
                        trial_iterative_time.append(iterative_time)

                        break  # If no error occurred, exit the loop and move to the next iteration
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        print("Retrying the iteration...")

            # save trial errors and times to a list containing all the trial information
            self._gamma_errors.append(trial_gamma_error)
            self._lib_lin_errors.append(trial_lib_lin_error)
            self._simple_errors.append(trial_simple_error)
            self._analytical_transform_times.append(
                trial_analytical_transform_time)
            self._analytical_fit_times.append(trial_analytical_fit_time)
            self._iterative_times.append(trial_iterative_time)

            self._remaining_mixed_proportions.append(
                trial_remaining_mixed_proportions)

        # convert errors and times to numpy arrays for plotting
        self._gamma_errors = np.array(self._gamma_errors)
        self._lib_lin_errors = np.array(self._lib_lin_errors)
        self._simple_errors = np.array(self._simple_errors)

        self._analytical_transform_times = np.array(
            self._analytical_transform_times)
        self._analytical_fit_times = np.array(self._analytical_fit_times)
        self._iterative_times = np.array(self._iterative_times)

        self._remaining_mixed_proportions = np.array(
            self._remaining_mixed_proportions)

        self.plot_errors(independent_var, mixed_proportions)
        self.plot_times(independent_var)

    def plot_errors(self, independent_var, mixed_proportions):
        """
        Saves the plot displaying information on errors obtained from the simulation.
        :return: None
        """
        num_trials = self._num_trials
        x_ax_counts = self._x_ax_counts
        num_levels = self._num_levels
        num_regressors = self._num_regressors
        num_obs = self._num_obs
        USE_MEAN = self._USE_MEAN
        gamma_errors = self._gamma_errors
        lib_lin_errors = self._lib_lin_errors
        simple_errors = self._simple_errors

        mean_remaining_mixed_proportions = np.mean(
            self._remaining_mixed_proportions, axis=0)

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


        # Saved for plotting blovk sampling data
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # fig, ax = plt.subplots(ncols = 2, figsize=(14, 10))
        

        if num_regressors == 1:
            if USE_MEAN:
                max_value = max(np.max(np.mean(gamma_errors, axis=0)), np.max(np.mean(lib_lin_errors, axis=0)),
                                np.max(np.mean(simple_errors, axis=0)))
            else:
                max_value = max(np.max(np.median(simple_errors, axis=0)), np.max(np.median(gamma_errors, axis=0)),
                                np.max(np.median(lib_lin_errors, axis=0)))
        else:
            if USE_MEAN:
                max_value = max(np.max(np.mean(gamma_errors, axis=0)), np.max(
                    np.mean(lib_lin_errors, axis=0)))
            else:
                max_value = max(np.max(np.median(gamma_errors, axis=0)),
                                np.max(np.median(lib_lin_errors, axis=0)))

        ax[0][0].set_yscale('log')  # Set y-axis to logarithmic scale

        if USE_MEAN:
            title = "Average Error over " + \
                str(num_trials) + " trials vs. Number of "+str(independent_var)
        else:
            title = "Median Error over " + \
                str(num_trials) + " trials vs. Number of "+str(independent_var)

        ax[0][0].text(x=0.5, y=1.1, s=title, fontsize=12, weight='bold', ha='center', va='bottom',
                      transform=ax[0][0].transAxes)
        ax[0][0].text(x=0.5, y=1.00,
                      s="Data Randomly Generated with " + str(num_regressors) + " categorical regressors each with " + str(
                          num_levels) + " levels.\nElements of the true"
                      "beta are uniformly selected from the range [" + str(
                          self._beta_range[0]) + ", "
                      + str(self._beta_range[1]) + "].", fontsize=8, alpha=0.75, ha='center',
                      va='bottom', transform=ax[0][0].transAxes)
        ax[0][0].set_xticklabels(ax[0][0].get_xticklabels(), rotation=90)

        # Create an array to hold the boxplot data
        box_data_gamma = []
        box_data_lib_lin = []

        for i, obs_count in enumerate(x_ax_counts):
            # Repeat the gamma_errors and lib_lin_errors values for each boxplot
            box_data_gamma.extend(gamma_errors[:, i])
            box_data_lib_lin.extend(lib_lin_errors[:, i])

        # Plot the boxplots

        x_boxplot = []
        x_ax_counts_strings = [str("{:.1e}".format(x)) for x in x_ax_counts]
        self._x_ax_counts_strings = x_ax_counts_strings

        for count in x_ax_counts_strings:
            x_boxplot += [count] * num_trials

        # DO NOT DELETE  # DO NOT DELETE  # DO NOT DELETE  # DO NOT DELETE

        self._x_boxplot = x_boxplot

        # y_boxplot_gamma = np.array(gamma_errors.T.flatten().tolist())
        y_boxplot_lib_lin = lib_lin_errors.T.flatten().tolist()
        y_boxplot_simple = simple_errors.T.flatten().tolist()

        # DO NOT DELETE  # DO NOT DELETE  # DO NOT DELETE  # DO NOT DELETE

        # self._x_boxplot = x_boxplot
        # y_boxplot_gamma = gamma_errors.T.flatten().tolist()
        # y_boxplot_lib_lin = lib_lin_errors.T.flatten().tolist()
        # y_boxplot_simple = simple_errors.T.flatten().tolist()

        # df_gamma_box = pd.DataFrame(
        #     {"x": np.array(x_boxplot), "y": np.array(y_boxplot_gamma)})

        # DO NOT DELETE  # DO NOT DELETE  # DO NOT DELETE  # DO NOT DELETE

        df_lib_lin_box = pd.DataFrame(
            {"x": np.array(x_boxplot), "y": np.array(y_boxplot_lib_lin)})

        if num_regressors == 1:
            df_simple_box = pd.DataFrame(
                {"x": np.array(x_boxplot), "y": np.array(y_boxplot_simple)})
            sns.boxplot(x='x', y='y', ax=ax[0][0], data=df_simple_box, width=.1, color=color3, saturation=.5,
                        boxprops=dict(alpha=.4),
                        whiskerprops=dict(color=color3), capprops=dict(color=color3), medianprops=dict(color=color3),
                        flierprops=dict(markerfacecolor=color3, markeredgecolor=color3))

        # for each drop row count we extract the correponding errors for the gama estimate
        num_columns = len(self._drop_under_count)

        import matplotlib.colors as mcolors
        base_rgb = mcolors.to_rgb(color1)



        # Create a linear gradient of shades
        shades = [mcolors.rgb2hex(np.clip(
            np.array(base_rgb) * (1 - i/num_columns), 0, 1)) for i in range(num_columns)]

        for i in range(num_columns):
            color1 = shades[i]
            # extract the errors for the given value of drop count and then plot
            y_boxplot_gamma = np.array(
                gamma_errors[:, :, i].T.flatten().tolist())
                
            df_gamma_box = pd.DataFrame(
                {"x": np.array(x_boxplot), "y": np.array(y_boxplot_gamma)})

            sns.boxplot(x='x', y='y', ax=ax[0][0], data=df_gamma_box, width=.3, color=color1, saturation=.5,
                        boxprops=dict(alpha=.3),
                        whiskerprops=dict(color=color1), capprops=dict(color=color1), medianprops=dict(color=color1),
                        flierprops=dict(markerfacecolor=color1, markeredgecolor=color1))

            # do the line plot for mean / median here since looping is needed
            if USE_MEAN:
                y_gam = np.mean(gamma_errors[:, :, i], axis=0)
            else:
                y_gam = np.median(gamma_errors[:, :, i], axis=0)

            sns.lineplot(x=x_ax_counts_strings, y=y_gam, color=color1,
                         label='Analytic: DC='+str(self._drop_under_count[i]), linewidth=2, ax=ax[0][0])

        # plot the iterative solution errors
        sns.boxplot(x='x', y='y', ax=ax[0][0], data=df_lib_lin_box, width=.2, color=color2, saturation=.5,
                    boxprops=dict(alpha=.4),
                    whiskerprops=dict(color=color2), capprops=dict(color=color2), medianprops=dict(color=color2),
                    flierprops=dict(markerfacecolor=color2, markeredgecolor=color2))

        if USE_MEAN:
            y_iter = np.mean(lib_lin_errors, axis=0)
            y_simple = np.mean(simple_errors, axis=0)
        else:
            y_iter = np.median(lib_lin_errors, axis=0)
            y_simple = np.median(simple_errors, axis=0)

        sns.lineplot(x=x_ax_counts_strings, y=y_iter, color=color2,
                     label='Standard Iterative', linewidth=2, ax=ax[0][0])
        if num_regressors == 1:
            sns.lineplot(x=x_ax_counts_strings, y=y_simple, color=color3, label='Simple Analytic MLE', linewidth=2,
                         ax=ax[0][0])

        ax[0][0].set_xlabel("Number of "+str(independent_var))
        ax[0][0].set_ylabel("Average MSE (Predicted Beta vs Actual Beta)")

        h, l = ax[0][0].get_legend_handles_labels()

        ax[0][0].legend(h, l, bbox_to_anchor=(1.05, 1), loc=2)

        # difference in MSE plot

        # Take the difference, leaving 'x' column unchanged
        # Create a copy of the 'df_gamma_box' DataFrame
        df_diff_analytic = df_gamma_box.copy()
        df_diff_analytic['y'] = df_gamma_box['y'].subtract(df_lib_lin_box['y'])

        if num_regressors == 1:
            df_diff_simple = df_simple_box.copy()
            df_diff_simple['y'] = df_simple_box['y'].subtract(
                df_lib_lin_box['y'])
            sns.boxplot(x='x', y='y', ax=ax[0][1], data=df_diff_simple, width=.2, boxprops=dict(alpha=0.5), color=color3,
                        whiskerprops=dict(color=color3), capprops=dict(color=color3), medianprops=dict(color=color3),
                        flierprops=dict(markerfacecolor=color3, markeredgecolor=color3))

        sns.boxplot(x='x', y='y', ax=ax[0][1], data=df_diff_analytic, width=.2, boxprops=dict(alpha=0.5), color=color1,
                    whiskerprops=dict(color=color1), capprops=dict(color=color1), medianprops=dict(color=color1),
                    flierprops=dict(markerfacecolor=color1, markeredgecolor=color1))

        ax[0][1].axhline(y=0, linestyle='--', color="black")
        ax[0][1].set_xlabel("Number of "+str(independent_var))

        y_lab = "Difference in Average MSE (Predicted Beta vs Actual Beta)"
        if not USE_MEAN:
            y_lab = "Difference in Average Median Squared Error (Predicted Beta vs Actual Beta)"

        ax[0][1].set_ylabel(
            "Difference in Average MSE (Predicted Beta vs Actual Beta)")
        ax[0][1].set_yscale('symlog')

        if USE_MEAN:
            title2 = "Difference in Average Error over " + \
                str(num_trials) + " trials vs. Number of "+str(independent_var)
        else:
            title2 = "Difference in Median Error over " + \
                str(num_trials) + " trials vs. Number of "+str(independent_var)

        if independent_var == "observations":
            subtitle = "Data Randomly Generated with " + str(
                num_regressors) + " categorical regressors each with " + str(
                num_levels) + " levels.\nWe compare the error of the analytic solution to the error of iterative " \
                              "solution. Nudge parameter is set to " + \
                str(self._nudge) + "."
        elif independent_var == "regressors":
            subtitle = "Data Randomly Generated: " + str(
                num_obs) + " observations each regressor with " + str(
                num_levels) + " levels.\nWe compare the error of the analytic solution to the error of iterative " \
                              "solution. Nudge parameter is set to " + \
                str(self._nudge) + "."
        elif independent_var == "levels":
            subtitle = "Data Randomly Generated: " + str(
                num_obs) + " observations and  " + str(
                num_regressors) + " regressors.\nWe compare the error of the analytic solution to the error of iterative " \
                "solution. Nudge parameter is set to " + str(self._nudge) + "."

        subtitle3 = "Mixing percentage: " + \
            str(self._mixing_percentage) + "\n. Large sampling batch size of " + \
            str(self._large_samples_size) + "."

        ax[0][1].text(x=1.95, y=1.1, s=title2, fontsize=12, weight='bold', ha='center', va='bottom',
                      transform=ax[0][0].transAxes)
        ax[0][1].text(x=1.95, y=1.03,
                      s=subtitle,
                      fontsize=8, alpha=0.75, ha='center', va='bottom', transform=ax[0][0].transAxes)

        x_middle = 0.35
        y_middle = 0.5 * (ax[0][0].get_position().ymax +
                          ax[0][1].get_position().ymin)

        # fig.text(x_middle, y_middle, subtitle3,
        #          fontsize=10)

        bottom_row_data = [mixed_proportions, mean_remaining_mixed_proportions]
        bottom_row_titles = ['Proportion of Data sampled with indivudual vs block sampling',
                             'Mean Resulting Proportion of Data sampled with\n indivudual vs block sampling after Row Deletion']
        
    
        for col in [0, 1]:
            data = bottom_row_data[col]
            title = bottom_row_titles[col]

            ax[1][col].set_frame_on(False)
            ax[1][col].xaxis.set_visible(False)
            ax[1][col].yaxis.set_visible(False)

            # Set the bar width and space between bars
            bar_width = 0.25  # Make the bars twice as thin
            bar_space = 0.2  # Adjust the space between bars as needed

            # Teal color using RGB values
            teal_color_rgb = (89, 156, 156)
            teal_color_normalized = tuple(
                val / 255.0 for val in teal_color_rgb)

            # Calculate positions for bars
            bar_positions = np.arange(0, len(data) *
                                    (bar_width + bar_space), (bar_width + bar_space))

            # Create bars with two stacked sections
            blue_heights = np.array(data)
            pink_heights = 1 - blue_heights

            # Blue section with teal color
            ax[1][col].bar(bar_positions, blue_heights, bar_width,
                        color=teal_color_normalized, edgecolor='none', label='Individual Samples')  # Remove the border

            # Pink section
            ax[1][col].bar(bar_positions, pink_heights, bar_width, bottom=blue_heights,
                        color='salmon', edgecolor='none', label='Block Samples')  # Remove the border

            # Place the title below the bars
            ax[1][col].text(0.5, -0.1, title,
                            ha='center', va='center', transform=ax[1][col].transAxes)

            # ax[1][col].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            h, l = ax[1][col].get_legend_handles_labels()
            ax[1][col].legend(h[:4], l[:4], bbox_to_anchor=(1.05, 1), loc=2)

        fig.tight_layout()  # Adjust plot layout to prevent overlapping

        plot_name = "Sim_MSE_" + str(self._num_regressors) + "regressors_" + str(self._num_levels) + "levels_" + str(
            self._num_trials) + "trials. MP="+str(self._mixing_percentage)+".png"

        plt.savefig(plot_name, dpi=300)

        plt.close()





    def plot_times(self, independent_var):
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
        x_ax_counts_strings = self._x_ax_counts_strings
        analytical_transform_times = self._analytical_transform_times
        analytical_fit_times = self._analytical_fit_times

        analytical_times = analytical_transform_times + analytical_fit_times
        iterative_times = self._iterative_times

        time_ratio = iterative_times / analytical_fit_times

        fig_time, ax_time = plt.subplots(figsize=(7, 5))

        # add the transform times to the fit times for the analytical box plot
        y_boxplot_analytical_time = np.array(
            analytical_times).T.flatten().tolist()

        y_boxplot_iterative_time = np.array(
            iterative_times).T.flatten().tolist()

        y_boxplot_time_ratio = np.array(
            time_ratio).T.flatten().tolist()

        df_a_times_box = pd.DataFrame(
            {"x": np.array(x_boxplot), "y": np.array(y_boxplot_analytical_time)})
        df_i_times_box = pd.DataFrame(
            {"x": np.array(x_boxplot), "y": np.array(y_boxplot_iterative_time)})

        df_time_ratio_box = pd.DataFrame(
            {"x": np.array(x_boxplot), "y": np.array(y_boxplot_time_ratio)})

        sns.boxplot(x='x', y='y', ax=ax_time, data=df_a_times_box, width=.3, color=color1, saturation=.5,
                    boxprops=dict(alpha=.3),
                    whiskerprops=dict(color=color1), capprops=dict(color=color1), medianprops=dict(color=color1),
                    flierprops=dict(markerfacecolor=color1, markeredgecolor=color1))

        sns.boxplot(x='x', y='y', ax=ax_time, data=df_i_times_box, width=.2, color=color2, saturation=.5,
                    boxprops=dict(alpha=.4),
                    whiskerprops=dict(color=color2), capprops=dict(color=color2), medianprops=dict(color=color2),
                    flierprops=dict(markerfacecolor=color2, markeredgecolor=color2))

        sns.boxplot(x='x', y='y', ax=ax_time, data=df_time_ratio_box, width=.2, color="black", saturation=.5,
                    boxprops=dict(alpha=.4),
                    whiskerprops=dict(color="black"), capprops=dict(color="black"), medianprops=dict(color="black"),
                    flierprops=dict(markerfacecolor="black", markeredgecolor="black"))

        sns.lineplot(x=x_ax_counts_strings, y=np.mean(analytical_times, axis=0), color=color1, label='Analytic',
                     linewidth=2,
                     ax=ax_time)
        sns.lineplot(x=x_ax_counts_strings, y=np.mean(iterative_times, axis=0), color=color2, label='Standard Iterative',
                     linewidth=2, ax=ax_time)

        sns.lineplot(x=x_ax_counts_strings, y=np.mean(time_ratio, axis=0), color="black", label='Time Ratio',
                     linewidth=2, ax=ax_time)

        stacked_times = {"x": x_ax_counts_strings, "transform times": np.mean(
            analytical_transform_times, axis=0), "fit times": np.mean(analytical_fit_times, axis=0)}
        stacked_times = pd.DataFrame(stacked_times)

        stacked_times.set_index('x').plot(kind='bar', stacked=True, color=[
            'orange', 'blue'], ax=ax_time, alpha=.1)

        ax_time.set_xlabel("Number of "+str(independent_var))
        ax_time.set_ylabel("Average Time Spent Fitting Model")

        h, l = ax_time.get_legend_handles_labels()

        ax_time.legend(h[:4], l[:4], bbox_to_anchor=(1.05, 1), loc=2)

        title_time = "Average Time to Fit Models over " + \
            str(num_trials) + " trials\nvs. Number of "+str(independent_var)

        ax_time.text(x=0.5, y=1.1, s=title_time, fontsize=12, weight='bold', ha='center', va='bottom',
                     transform=ax_time.transAxes)
        ax_time.text(x=0.5, y=1.05,
                     s="Data Randomly Generated with " + str(
                         num_regressors) + " categorical regressors each with " + str(
                         num_levels) + " levels", fontsize=8, alpha=0.75, ha='center', va='bottom',
                     transform=ax_time.transAxes)
        ax_time.set_yscale('log')
        fig_time.tight_layout()  # Adjust plot layout to prevent overlapping
        plot_name = "Fitting_Times_" + str(self._num_regressors) + "regressors_" + str(
            self._num_levels) + "levels_" + str(
            self._num_trials) + "trials.png"
        plt.savefig(plot_name, dpi=300)  # Save the plot with higher DPI
