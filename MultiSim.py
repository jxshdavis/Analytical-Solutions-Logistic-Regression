import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import bernoulli
from scipy.special import expit
import AnalyticalSolution
import DataGenerator
import pandas as pd
from sklearn.linear_model import LogisticRegression
from timeit import default_timer as timer



class Simulation:

    def __init__(self, num_observations, num_regressors, num_levels):
        self._iterative_time = None
        self._analytical_time = None
        self._num_observations = num_observations
        self._num_regressors = num_regressors
        self._num_levels = num_levels

        # get the generated data
        self._generator, self._sim_data, self._sim_y = self.generate_data()

        # set the two models

        self._analytical_model = self.set_analytical_model()



        self._iterative_model = self.set_iterative_model()

        if num_regressors == 1:
            self._simple_params = self.gen_simple_parameters()


        # time the two models took to run

    def generate_data(self):
        generator = DataGenerator.Generator(self._num_observations, self._num_regressors, self._num_levels)
        sim_data_1 = generator.get_design_matrix()
        sim_y_1 = generator.get_response()

        return generator, sim_data_1, sim_y_1

    def set_analytical_model(self):
        start_time = timer()
        analytical_model = self._generator.get_analytical_model()
        analytical_model.set_y(self._sim_y)
        analytical_model.transform_response()
        # z = analytical_model.get_transformed()[0].values
        # x_tilde = analytical_model.get_transformed()[1]
        analytical_model.fit()

        end_time = timer()
        self._analytical_time = end_time - start_time

        return analytical_model

    def set_iterative_model(self):
        # Start the timer
        start_time = timer()
        iterative_model = LogisticRegression(solver='liblinear')
        df_encoded = pd.get_dummies(self._sim_data, drop_first=True)
        df_encoded_1 = df_encoded.astype(int)
        iterative_model.fit(df_encoded_1, self._sim_y)
        end_time = timer()

        # set the time the model took to run
        self._iterative_time = end_time - start_time
        return iterative_model

    def get_parameters(self):
        true = self._generator.get_params()

        gamma = [round(x, 3) for x in self._analytical_model.get_gamma()[0].tolist()]

        lib_lin_sol = list(self._iterative_model.intercept_) + list(self._iterative_model.coef_[0])
        lib_lin_sol = [round(x, 3) for x in lib_lin_sol]

        # print("true parameters")
        # print(true)
        # print("Analytically Predicted Gamma")
        # print(gamma)
        # print("Lib Linear solution:")
        # print(lib_lin_sol)

        return np.array(true), np.array(gamma), np.array(lib_lin_sol)

    def get_times(self):
        """
        :return: Time Analytical Model Took to fit, Time Iterative Model Took to fit
        """
        return self._analytical_time, self._iterative_time

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
            params.append(np.log(success_count[-1]/(combo_counts[-1]-success_count[-1])))

        for index in range(0, len(success_count)-1):


            if combo_counts[index] - success_count[index] == 0:
                ci = E
            elif success_count[index] == 0:
                ci = 1 / E
            else:
                ci = success_count[index]/(combo_counts[index]-success_count[index])
            params.append(np.log(ci) - params[0])
        return np.array(params)


    def get_simple_parameters(self):
        return self._simple_params

