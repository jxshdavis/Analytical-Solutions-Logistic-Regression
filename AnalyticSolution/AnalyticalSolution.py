import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from scipy.stats import bernoulli
from scipy.special import expit
import math
import copy
import pandas as pd
from numpy.linalg import inv
from collections import defaultdict
from timeit import default_timer as timer


class AnalyticalSolution:
    """
    Generates an analytical estimate of beta when there are multiple regressors.
    """

    def encode_df(self):
        """
        Performes 1-hot encoding on the original dataframe.
        returns: 1-hot encoded dataframe wit the first level of each regressor dropped.
        """
        df = copy.deepcopy(self._x)
        df_encoded = pd.get_dummies(df, drop_first=True)
        df_encoded = df_encoded.astype(int)

        return df_encoded

    def __init__(
        self,
        x,
        sample_size_info,
        drop_under_count,
        lamb=0,
        nudge=10 ** (-5),
        y=np.array([0]),
        large_sample_rows=None,
    ):
        """
        @param x: Your data matrix with each regressor having at least 1 of all of its levels in its collumn. Do not
                    pass in a 1-hot encoded matrix - that will be done within the class.

        @param lamb:
        @param nudge:
        @param y:
        """

        self._x = x
        self._y = y
        self._lambda = lamb
        self._x_tilde = None
        self._Hz_hat = None
        self._invert_this = None
        self._z = None
        # self._Xz = None
        self._level_counts = None
        self._large_sample_rows = large_sample_rows
        self._sample_size_info = sample_size_info
        self._w_combo_counts = None
        self._gammas = []
        self._nudge = nudge
        self._success_counts = None
        self._row_counts = None
        self._remaining_num_batch_samples = None
        self._drop_under_count = drop_under_count
        self._weights = None
        self._count_per_row = None
        self._all_unique_rows = None
        self._final_design = None

        # count the number of levels of each predictor

        start = timer()
        self.count_levels()
        end = timer()

        # print("count levels  time")
        # print(end - start)

        # if X is not encoded with dummy varibles, we do so here
        self._encoded_x = self.encode_df()

        # self.transform_design(self)

    def get_count_per_row(self):
        return self._count_per_row

    def get_final_design(self):
        return self._final_design

    def get_weights(self):
        return self._weights

    def set_lambda(self, lamb):
        self._lambda = lamb

    def set_penalty(self, penalty):
        self._penalty = penalty

    def get_transformed(self):
        return self._z, self._x_tilde

    @staticmethod
    def add_combos(current_combos, num_categories):
        """
        :param current_combos: the current list of combinations you are extending
        :param num_categories: The number of new categories in the predictor you are encoding
        :return: an updated list of all possible combinations of predictors in a 1-hot encoded fashion
        """
        new_combos = []
        for i in range(num_categories - 1):
            combo = [0] * (num_categories - 1)
            combo[i] = 1
            for old_combo in current_combos:
                new_combos.append(old_combo + combo)

        combo = [0] * (num_categories - 1)
        for old_combo in current_combos:
            new_combos.append(old_combo + combo)

        return new_combos

    @staticmethod
    def transform_design(self):
        num_rows, num_cols = self._x.shape
        K = self.count_levels()
        df_encoded = self._encoded_x
        # get an ordered list of all possible combinations of categories
        combinations = [[]]

        for num_cat in K:
            combinations = self.add_combos(combinations, num_cat)
        xt = pd.DataFrame(combinations, columns=df_encoded.columns)

        xt.insert(0, 'intercept', 1)
        # print("transformed Design:")
        # print(xt)

        self._x_tilde = xt

        prod = (xt.T).dot(xt)

    def transform_response(self):

        (
            batch_integers,
            individual_integers,
            num_individual_samples,
            num_batch_samples,
        ) = self._sample_size_info

        individual_integers = np.array(list(individual_integers))

        start_time = timer()

        df_encoded = self._encoded_x

        ones_column = np.ones((df_encoded.shape[0], 1), dtype=int)
        # df_encoded = np.hstack((ones_column, df_encoded))

        y = self._y

        decimal_numbers = np.dot(df_encoded, 2 ** np.arange(df_encoded.shape[1])[::-1])

        unq, indices, inverse_indices, cnt = np.unique(
            decimal_numbers,
            axis=0,
            return_index=True,
            return_counts=True,
            return_inverse=True,
        )

        num_bits = df_encoded.shape[1]

        end_time = timer()
        transform_time = end_time - start_time

        response = [x[0] for x in self._y.tolist()]

        # Compute success_counts
        success_counts = np.bincount(inverse_indices, weights=response)

        # Compute success_frequencies
        count_per_row = np.bincount(inverse_indices)

        self._count_per_row = count_per_row

        success_frequencies = np.where(
            count_per_row > 0, success_counts / count_per_row, 0
        )

        # get a list of the gammas for each drop_under_count value needed. The batch size info and fittign time info will only
        # reflect the last
        # value in self._drop_under_count

        stored_all_unique_rows = (
            (unq[:, None] & (1 << np.arange(num_bits)[::-1])) > 0
        ).astype(int)

        ones_column = np.ones((stored_all_unique_rows.shape[0], 1), dtype=int)

        self._all_unique_rows = np.hstack((ones_column, stored_all_unique_rows))

        for drop_under_count in self._drop_under_count:

            # To remove rows where freq == 0 or freq == 1 or the count is <= drop_under_count

            # if we want to remove rows with success frequecy of 0 or 1 use this line
            to_delete = np.where(
                (success_frequencies == 0)
                | (success_frequencies == 1)
                | (count_per_row <= drop_under_count)
            )[0]

            # alternate method where we estimate 1 and 0 success frequencies

            alternate = True

            if alternate:
                zero_success_frequencies = np.where((success_frequencies == 0))[0]

                unity_success_frequencies = np.where((success_frequencies == 1))[0]

                zero_mu = 1 / 2 * (1 / cnt[zero_success_frequencies])
                one_mu = 1 - 1 / 2 * (1 / cnt[unity_success_frequencies])

                new_success_frequencies = np.copy(success_frequencies)

                new_success_frequencies[zero_success_frequencies] = zero_mu
                new_success_frequencies[unity_success_frequencies] = one_mu

                new_unq = unq

                new_cnt = cnt

                z = np.log(
                    (new_success_frequencies) / (1 - new_success_frequencies)
                ) + (success_frequencies - new_success_frequencies) / (
                    new_success_frequencies * (1 - new_success_frequencies)
                )

            else:
                nums_to_delete = unq[to_delete]

                remaining_batch_integers = np.setdiff1d(batch_integers, nums_to_delete)

                remaining_num_batch_samples = remaining_batch_integers.shape[0]

                self._remaining_num_batch_samples = remaining_num_batch_samples

                new_success_frequencies = np.delete(
                    np.array(success_frequencies), to_delete, axis=0
                )

                z = np.log((new_success_frequencies) / (1 - new_success_frequencies))

                # print(f"Min suc freq: {min(success_frequencies)}")
                # print(f"Max suc freq: {max(success_frequencies)}")

                new_unq = np.delete(unq, to_delete, axis=0)

                new_cnt = np.delete(cnt, to_delete, axis=0)

            self._row_counts = new_cnt

            sample_weights = (
                new_cnt * new_success_frequencies * (1 - new_success_frequencies)
            )

            W = np.diag(sample_weights)

            self._weights = W

            new_unq = (
                (new_unq[:, None] & (1 << np.arange(num_bits)[::-1])) > 0
            ).astype(int)

            # add intercept column to unq
            ones_column = np.ones((new_unq.shape[0], 1), dtype=int)
            new_unq = np.hstack((ones_column, new_unq))

            start_time = timer()

            # save the final version of tilde X for later use
            self._final_design = new_unq

            if self._lambda != 0:
                linear_model = Ridge(alpha=self._lambda).fit(
                    new_unq, z, sample_weight=sample_weights
                )
            else:
                linear_model = LinearRegression(fit_intercept=False).fit(
                    new_unq, z, sample_weight=sample_weights
                )
                # linear_model = LinearRegression(fit_intercept=True).fit(
                #     new_unq, z, sample_weight=sample_weights)

            g_new = linear_model.coef_

            # print(f"Determinant: {np.linalg.det(Xw@unq)}")

            self._gammas.append(g_new)
            end_time = timer()

        fit_time = end_time - start_time
        # print()
        # print(f"fit time: {fit_time}")
        # print(f"transform time: {transform_time}")

        return fit_time, transform_time

    def count_levels(self):
        data = np.array(copy.deepcopy(self._x))
        num_rows, num_cols = data.shape

        # K = []
        # obtain the K_js

        K = [np.unique(data[:, 0]).shape[0]] * num_cols

        # for col in range(0, num_cols):
        #     K.append(len(set(data[:, col].flatten())))

        self._level_counts = K
        return K

    def get_encoded_x(self):
        return self._encoded_x

    def fit(self, center_design=False):

        n = self._invert_this.shape[0]

        invert_this = self._invert_this

        Xz = self._Xz

        g = inv(invert_this) @ Xz

        self._gamma = g

    def get_gamma(self):
        return self._gammas

    def get_simple_counts(self):
        """
        :return: List of successes for each combination, and number of each combination total
        """
        return self._success_counts, self._w_combo_counts

    def set_y(self, y):
        self._y = y

    def get_mixing_info(self):
        return self._row_counts, self._remaining_num_batch_samples

    def get_all_unique_rows(self):
        return self._all_unique_rows

    def predict_proba(self, X):
        success = 1 / (1 + np.exp(-1 * X @ self._gammas))
        return np.hstack(1 - success, success)
