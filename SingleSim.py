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




class SingleSimulation:
    """
    Used to test the analytical solution presented when there is a single regressor.
    """

    def __init__(self, num_observations, num_levels):

        self._analytical_time = None
        self._num_observations = num_observations
        self._num_regressors = 1
        self._num_levels = num_levels

        # get the generated data
        self._generator, self._sim_data, self._sim_y = self.generate_data()

        start = timer()
        self._params = np.log()
        end = timer()
        self._analytical_time = end - start



        # time the two models took to run

    def generate_data(self):
        generator = DataGenerator.Generator(self._num_observations, self._num_regressors, self._num_levels)
        sim_data_1 = generator.get_design_matrix()
        sim_y_1 = generator.get_response()
        return generator, sim_data_1, sim_y_1


    def get_parameters(self):
        return




    def get_times(self):
        """
        :return: Time Analytical Model Took to fit, Time Iterative Model Took to fit
        """
        return self._analytical_time
