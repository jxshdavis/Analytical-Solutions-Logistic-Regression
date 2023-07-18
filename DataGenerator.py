import random
import pandas as pd
import AnalyticalSolution
import numpy as np
from scipy.stats import bernoulli
from scipy.special import expit
import copy

class Generator:
    """
    Randomly Generates the design matrix first and then generates the parameters. The number of regressors in the
    1-hot encoded data will depend on the number of levels for each initial regressor chosen during the random
    generator.
    """

    def __init__(self, num_observations, num_regressors, num_levels, nudge, beta_range):
        """
        :param num_observations:
        :param num_regressors:
        :param num_levels_size:  determines the number of "levels" each regressor
        can have.
        """
        self._num_observations = num_observations
        self._num_regressors = num_regressors
        self._num_levels = num_levels
        self._design_matrix = self.gen_design_matrix()
        self._nudge = nudge
        self._beta_range = beta_range
        self._params = self.gen_model_params()
        # print(self._design_matrix)
        self._analytical_model = None
        self._encoded_x = self.transform_design()
        self._response = self.gen_response()

    def gen_model_params(self):
        """
        :return: randomly choose the Betas for the transformed
        data with all betas between -10 and 10
        """
        # Count the number of unique values in each column
        unique_counts = self._design_matrix.nunique(axis=0)

        # Convert unique_counts to a list
        unique_counts_list = unique_counts.tolist()

        # Sum all the entries in the list
        total_unique_counts = sum(unique_counts_list)
        params = [0] * (total_unique_counts - self._num_regressors + 1)

        for index in range(len(params)):
            params[index] = round(random.uniform(self._beta_range[0], self._beta_range[1]), 2)
        return params

    def gen_design_matrix(self):
        """
        :return: a randomly generated design matrix
        """
        n = self._num_regressors
        N = self._num_observations
        K = self._num_levels
        design = []
        levels = list(range(K))
        for observation in range(N):
            obs = []
            for i in range(n):
                obs.append(str(random.choice(levels)))
            design.append(obs)

        col_names = ["col" + str(x) for x in range(n)]

        X = pd.DataFrame(design, columns=col_names)
        # print("Design Matrix:")
        # print(X)

        return X

    def transform_design(self):
        model = AnalyticalSolution.AnalyticalSolution(self.get_design_matrix(), nudge = self._nudge)

        # save the analytical model so we do not have to re-encode x in the future!
        self._analytical_model = model
        return model.get_encoded_x()

    def gen_response(self):
        """
        :return: the response vector Y as a numpy array.
        """
        probabilities = []
        X = copy.deepcopy(self._encoded_x)
        X.insert(0, 'intercept', 1)
        Beta = np.array(self._params)
        probabilities = expit(X.dot(Beta))
        response = np.array([[bernoulli.rvs(p)] for p in probabilities])

        # print("response vector:")
        # print(response)
        return response


    def get_design_matrix(self):
        return self._design_matrix

    def get_params(self):
        return self._params

    def get_response(self):
        return self._response

    def get_single_reg_counts(self):
        pass

    def get_analytical_model(self):
        return self._analytical_model

