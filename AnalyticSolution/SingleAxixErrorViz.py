import seaborn as sns
import Simulation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv
import AnalyticalSolution
import ExperimentSimulations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
import random

# for binomial glm
import statsmodels.api as sm


class SingleAxisErrorViz:

    def __init__(self,
                 num_trials,
                 num_regressors, 
                 num_levels,
                 number_x_ticks,
                 num_obs_per_row=100,
                 beta_range=[-1, 1],
                 lamb=0,
                 penalty="l2"):

        self._num_trials = num_trials
        self._num_regressors = num_regressors
        self._num_levels = num_levels
        self._number_x_ticks = number_x_ticks
        self._num_obs_per_row = num_obs_per_row
        self._beta_range = beta_range
        self._lamb = lamb
        self._penalty = penalty


    def run_sim(self):
        # for each probability in the predicted probabilities we want the error in the estimate to vary!

        # There will be num_levels^num_regressors predicted probabitlies to conisder.

        # So in the binary, 2 predictor case there will be 4 probabilities!

        # so for a given probability the x_axis will be the error in the datas estimate of this probability,
        # and the y-axis will be the distance between the prdeicted beta and the true beta, and also the ditance between 
        # the predicted beta and the MLE


        # Get Beta

        num_bits = (self._num_levels-1)*self._num_regressors

        true_beta = [0] * (num_bits + 1)

        for index in range(len(true_beta)):
            true_beta[index] = round(random.uniform(
                self._beta_range[0], self._beta_range[1]), 2)


        
        # get tilde X and true success probabilities
        
        new_unq = np.array([i for i in range(self._num_levels**self._num_regressors)])

        new_unq = ((new_unq[:, None] & (1 << np.arange(num_bits)[::-1]))
                    > 0).astype(int)

        # add intercept column to unq
        ones_column = np.ones((new_unq.shape[0], 1), dtype=int)
        tilde_X = np.hstack((ones_column, new_unq))


        probabilities = 1/(1+np.exp(-1*(tilde_X @ np.array(true_beta))))
        
        import math
        num_plots = len(probabilities)
        num_cols = math.ceil(math.sqrt(num_plots))
        num_rows = math.ceil(num_plots / num_cols)
        # fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 8))

        for row_idx, p in  enumerate(probabilities):


            # generate list of errors we want to plot
            
            buffer = .01

            errors = np.linspace(-p+buffer, 1-p-buffer, self._number_x_ticks)


            mle_probabilities_mse = []
            analytic_probabilities_mse = []

            non_weighted_analytic_probabilities_mse = []

            for error in errors:
                f = np.copy(probabilities)


                mean = 0  # Mean of the normal distribution
                std_dev = 0.005  # Standard deviation of the normal distribution

                # Generate normal noise with the same length as the vector
                noise = np.random.normal(loc=mean, scale=std_dev, size=len(f))

                f += noise

                f[row_idx] = probabilities[row_idx] + error

                successes = np.round(self._num_obs_per_row*f).astype(int)
                failures = self._num_obs_per_row - successes

                # Create a DataFrame from the rounded numpy arrays
                response = pd.DataFrame({'Successes': successes, 'Failures': failures})


                binom_model = sm.GLM(response, tilde_X, family=sm.families.Binomial())
 
                binom_model_results = binom_model.fit()

                iterative_params = binom_model_results.params

                # Get the predicted response
                mle_predicted_probabilities = 1/(1+np.exp(-1*(tilde_X @ np.array(iterative_params))))

                sample_weights = self._num_obs_per_row * f * (1-f)

                linear_model = LinearRegression(fit_intercept=False).fit(tilde_X, np.log(f/(1-f)), sample_weight=sample_weights)

                g_new = linear_model.coef_

                analytic_predicted_probabilities = 1/(1+np.exp(-1*(tilde_X @ g_new)))


                linear_model2 = LinearRegression(fit_intercept=False).fit(tilde_X, np.log(f/(1-f)))

                g_new =  linear_model2.coef_

                non_weighted_analytic_predicted_probabilities = 1/(1+np.exp(-1*(tilde_X @ g_new)))



               
                mle_probabilities_mse.append(np.mean(mle_predicted_probabilities-probabilities)**2)
                analytic_probabilities_mse.append(np.mean(analytic_predicted_probabilities-probabilities)**2)
                non_weighted_analytic_probabilities_mse.append(np.mean(non_weighted_analytic_predicted_probabilities-probabilities)**2)

                print("Estimation for f", str(f))
                print(analytic_probabilities_mse)
                print()
                

            # Set Seaborn style
            plt.clf()
            sns.set_style("whitegrid")
            colors = sns.color_palette("husl", 3)  # Using the 'husl' color palette with 2 colors

            sns.lineplot(x=errors, y=mle_probabilities_mse, 
            label='Error in MLE Predicted Success P',
            color=colors[0])

            # Plot the second line plot using Seaborn
            sns.lineplot(x=errors, y=analytic_probabilities_mse, 
            label='Error in Analytic Predicted Success P',
            color=colors[1])


            # Plot the non_weighted line plot using Seaborn
            # sns.lineplot(x=errors, y=non_weighted_analytic_probabilities_mse, 
            # label='Error in Analy-Nonweighted Predicted Success P',
            # color=colors[2])

            # Access the current Axes object
            ax = plt.gca()

            # Customize the plot if needed (e.g., axes labels, title, etc.)
            ax.set_xlabel('Error on Measures Success Frequency')
            ax.set_ylabel('MSE of Prediction vs Acual Probabilities')
            ax.set_title('Errors the index '+str(row_idx+1)+" predicted Probability")

            # Show the legend
            plt.legend()

            # Show the plot
            plt.savefig("Error Vs Prob Error Row:"+str(row_idx+1)+".png", dpi = 300)


        print("done")

        
experiment = SingleAxisErrorViz(num_trials=1,
                 num_regressors=3, 
                 num_levels=2,
                 number_x_ticks = 100)

experiment.run_sim()





    