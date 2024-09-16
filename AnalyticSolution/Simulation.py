import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import bernoulli
from scipy.special import expit
import AnalyticalSolution
import DataGenerator
import pandas as pd
from timeit import default_timer as timer


class Simulation:
    """
    This Class generates synthetic data for logistic regression and computes iteratee MLE and analytic heuristic
    estimates of the underlying model using the DataGenerator class and sklearn's Logsitc Regression model.
    """

    def __init__(self, num_observations, num_regressors, num_levels, nudge, beta_range, lamb, penalty, mixing_percentage, large_samples_size, drop_under_count, max_iter, random_data=True):
        self._iterative_time = None
        self._mixing_percentage = mixing_percentage
        self._large_samples_size = large_samples_size
        self._analytical_transform_time = None
        self._analytical_fit_time = None
        self._num_observations = num_observations
        self._lambda = lamb
        self._penalty = penalty
        self._num_regressors = num_regressors
        self._num_levels = num_levels
        self._nudge = nudge
        self._beta_range = beta_range
        self._drop_under_count = drop_under_count
        self._iterative_design = None
        self.max_iter = max_iter
        # get the generated data

        start = timer()
        self._generator, self._sim_data, self._sim_y = self.generate_data()
        end = timer()

        print("data gen time")
        print(end - start)

        # set the two models
        self._analytical_model = self.set_analytical_model()
        self._iterative_model = self.set_iterative_model()

        if num_regressors == 1:
            self._simple_params = self.gen_simple_parameters()

    def get_iterative_design(self):
        return self._iterative_design

    def generate_data(self):
        generator = DataGenerator.Generator(self._num_observations, self._num_regressors, self._num_levels,
                                            self._nudge, self._beta_range, mixing_percentage=self._mixing_percentage,
                                            large_samples_size=self._large_samples_size,
                                            drop_under_count=self._drop_under_count)
        sim_data_1 = generator.get_design_matrix()
        sim_y_1 = generator.get_response()

        return generator, sim_data_1, sim_y_1

    def set_analytical_model(self):

        analytical_model = self._generator.get_analytical_model()

        analytical_model.set_y(self._sim_y)
        analytical_model.set_lambda(self._lambda)
        analytical_model.set_penalty(self._penalty)

        fit_time, transform_time = analytical_model.transform_response()

        # z = analytical_model.get_transformed()[0].values
        # x_tilde = analytical_model.get_transformed()[1]

        # self._analytical_transform_time = transform_time
        self._analytical_transform_time = 0
        self._analytical_fit_time = fit_time

        return analytical_model

    def set_iterative_model(self):

        # Start the timer
        start_time = timer()
        if self._lambda == 0:
            iterative_model = LogisticRegression(
                solver='lbfgs', penalty=None)

        else:
            max_iter = self.max_iter
            iterative_model = LogisticRegression(
                solver='liblinear', C=1/self._lambda,max_iter=max_iter)
        df_encoded = pd.get_dummies(self._sim_data, drop_first=True)
        df_encoded_1 = df_encoded.astype(int)

        iterative_model.fit(df_encoded_1, self._sim_y.ravel())
        self._iterative_design = df_encoded_1
        end_time = timer()

        # set the time the model took to run
        print("Iterative model time")
        print(end_time - start_time)
        self._iterative_time = end_time - start_time
        return iterative_model

    def get_parameters(self):
        true = self._generator.get_params()

        gamma = self._analytical_model.get_gamma()

        lib_lin_sol = list(self._iterative_model.intercept_) + \
            list(self._iterative_model.coef_[0])

        lib_lin_sol = [x for x in lib_lin_sol]

        return np.array(true), np.array(gamma), np.array(lib_lin_sol)

    def get_times(self):
        """
        :return: Time Analytical Model Took to fit, Time Iterative Model Took to fit
        """
        return self._analytical_transform_time, self._analytical_fit_time, self._iterative_time

    def gen_simple_parameters(self):
        success_count, combo_counts = self._analytical_model.get_simple_counts()

        E = 10**2
        params = []
        # add the intercept -> when x_i = (0,..., 0) this corresponds to when the non 1-hot encoded x is equal
        # to its first level. This is counted in the last position of the combo_counts vector.
        if combo_counts[-1]-success_count[-1] == 0:
            params.append(E)
        elif success_count[-1] == 0:
            params.append(1 / E)
        else:
            params.append(
                np.log(success_count[-1]/(combo_counts[-1]-success_count[-1])))

        for index in range(0, len(success_count)-1):

            if combo_counts[index] - success_count[index] == 0:
                ci = E
            elif success_count[index] == 0:
                ci = 1 / E
            else:
                ci = success_count[index] / \
                    (combo_counts[index]-success_count[index])
            params.append(np.log(ci) - params[0])
        return np.array(params)

    def get_simple_parameters(self):
        return self._simple_params

    def get_analytical_model(self):
        return self._analytical_model
