import random
import pandas as pd
import AnalyticalSolution
import numpy as np
from scipy.stats import bernoulli
from scipy.special import expit
import copy
from timeit import default_timer as timer


class Generator:
    """
    Randomly Generates the design matrix first and then generates the parameters. The number of regressors in the
    1-hot encoded data will depend on the number of levels for each initial regressor chosen during the random
    generator.
    """

    def __init__(self, num_observations, num_regressors, num_levels, nudge, beta_range, drop_under_count, mixing_percentage=0, large_samples_size=30, ):
        """
        :param num_observations:
        :param num_regressors:
        :param num_levels_size:  determines the number of "levels" each regressor
        :mixing_percentage: a number between 0 and 1 where our samples will have mixing_percentage percent of total samples from a much smaller pool
        than the rest of the samples are randomly generated pool. This will allow us to get much larger counts for certain rows that others. 
        If mixing_percentage=0 then all samples are drawn from all rows with equal probability. This paramter should be set to 0 if running an 
        experiment with regressors which have more than 2 levels.
        """
        self._num_observations = num_observations
        self._num_regressors = num_regressors
        self._num_levels = num_levels
        self._mixing_percentage = mixing_percentage
        self._large_samples_size = large_samples_size
        self._large_sample_rows = None
        self._batch_integers = []
        self._individual_integers = []
        self._individual_samples = 0
        self._batch_samples = 0
        self._drop_under_count = drop_under_count

        start = timer()
        self._design_matrix = self.gen_design_matrix()
        end = timer()

        print("design mat gen time")
        print(end - start)

        self._nudge = nudge
        self._beta_range = beta_range
        self._params = self.gen_model_params()
        # x_boxplot*len(gamma_errors)(self._design_matrix)
        self._analytical_model = None

        start = timer()
        self._encoded_x = self.transform_design()
        end = timer()

        print("encode X  time")
        print(end - start)

        start = timer()
        self._response = self.gen_response()
        end = timer()

        print("response gen time")
        print(end - start)

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
            params[index] = round(random.uniform(
                self._beta_range[0], self._beta_range[1]), 2)
        return params

    def gen_design_matrix(self):
        """
        :return: a randomly generated design matrix
        """
        n = self._num_regressors
        N = int(self._num_observations * (1-self._mixing_percentage))
        N_prime = int(self._num_observations * self._mixing_percentage)
        K = self._num_levels
        design = []
        levels = list(range(K))

        col_names = ["col" + str(x) for x in range(n)]

        # Generate a random design matrix

        if self._mixing_percentage > 0:

            N_prime = int(self._num_observations *
                          self._mixing_percentage / self._large_samples_size)

            a = 0
            b = 2**self._num_regressors

            if b - a + 1 < N_prime:

                unique_integers = np.array(random.sample(range(a, b), b-a))
            else:
                unique_integers = np.array(
                    random.sample(range(a, b + 1), N_prime))

            self._batch_integers = unique_integers

            # Use random.sample to generate n unique integers

            num_bits = self._num_regressors

            if N_prime > 0:
                large_sample_rows = pd.DataFrame(((unique_integers[:, None] & (1 << np.arange(num_bits)[::-1]))
                                                  > 0).astype(int))

            samples_withreplacement = np.array(
                random.choices(range(a, b), k=N))
            self._individual_integers = set(samples_withreplacement)

            if N > 0:
                samples_withreplacement = pd.DataFrame(((samples_withreplacement[:, None] & (1 << np.arange(num_bits)[::-1]))
                                                        > 0).astype(int))

            if N_prime > 0 and N > 0:
                for _, row in large_sample_rows.iterrows():
                    copies = pd.DataFrame([row] * 30)
                    samples_withreplacement = pd.concat(
                        [samples_withreplacement, copies], ignore_index=True)
            elif N == 0:
                samples_withreplacement = large_sample_rows

            design_matrix = samples_withreplacement

        else:
            design_matrix = np.random.randint(0, K, size=(N, n))

        self._individual_samples = N
        self._batch_samples = N_prime
        X = pd.DataFrame(design_matrix)

        # print("Design Matrix:")
        # print(X)

        return X

    def transform_design(self):

        model = AnalyticalSolution.AnalyticalSolution(
            self.get_design_matrix(), self.get_sample_size_info(), nudge=self._nudge, drop_under_count=self._drop_under_count)

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

        # response = np.array([[bernoulli.rvs(p)] for p in probabilities])

        response2 = bernoulli.rvs(probabilities).reshape(-1, 1)

        return response2

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

    def get_sample_size_info(self):
        return self._batch_integers, self._individual_integers, self._individual_samples, self._batch_samples
