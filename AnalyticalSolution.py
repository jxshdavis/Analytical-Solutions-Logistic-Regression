import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
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

    def __init__(self, x, lamb=[0], nudge=10 ** (-5), y=np.array([0])):
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
        self._penalty = None
        self._x_tilde = None
        self._invert_this = None
        self._z = None
        # self._Xz = None
        self._level_counts = None

        # count the number of levels of each predictor
        self.count_levels()

        # if X is not encoded with dummy varibles, we do so here
        self._encoded_x = self.encode_df()

        # self.transform_design(self)

        self._w_combo_counts = None
        self._gamma = None
        self._nudge = nudge
        self._success_counts = None

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

        print(prod)

    def transform_response(self):

        start_time = timer()

        df_encoded = self._encoded_x

        ones_column = np.ones((df_encoded.shape[0], 1), dtype=int)
        # df_encoded = np.hstack((ones_column, df_encoded))

        y = self._y

        decimal_numbers = np.dot(
            df_encoded, 2 ** np.arange(df_encoded.shape[1])[::-1])

        # unq, inverse_indices, cnt = np.unique(
        #     df_encoded, axis=0, return_counts=True, return_inverse=True)

        unq, indices,  inverse_indices, cnt = np.unique(
            decimal_numbers, axis=0, return_index=True, return_counts=True, return_inverse=True)

        # sorted_indices = sorted(indices)
        # unq = decimal_numbers[sorted_indices]

        # sorted_inverse_indices = np.argsort(sorted_indices)[inverse_indices]

        num_bits = df_encoded.shape[1]
        unq = ((unq[:, None] & (
            1 << np.arange(num_bits)[::-1])) > 0).astype(int)

        end_time = timer()
        transform_time = end_time - start_time

        start_time = timer()

        response = [x[0] for x in self._y.tolist()]

        # Compute success_counts
        success_counts = np.bincount(inverse_indices, weights=response)

        # Compute success_frequencies
        count_per_row = np.bincount(inverse_indices)
        success_frequencies = np.where(
            count_per_row > 0, success_counts / count_per_row, 0)

        # To remove rows where freq == 0 or freq == 1
        to_delete = np.where((success_frequencies == 0) |
                             (success_frequencies == 1))[0]

        success_frequencies = np.delete(
            np.array(success_frequencies), to_delete, axis=0)
        unq = np.delete(unq, to_delete, axis=0)

        cnt = np.delete(cnt, to_delete, axis=0)

        W = np.diag(cnt * success_frequencies*(1-success_frequencies))

        z = np.log((success_frequencies) / (1-success_frequencies))

        # add intercept column to unq
        ones_column = np.ones((unq.shape[0], 1), dtype=int)
        unq = np.hstack((ones_column, unq))

        Xw = unq.T @ W
        g = inv(Xw@unq) @ Xw @ z
        self._gamma = g
        end_time = timer()

        fit_time = end_time - start_time

        return fit_time, transform_time

    def count_levels(self):
        data = np.array(copy.deepcopy(self._x))
        num_rows, num_cols = data.shape

        K = []
        # obtain the K_js
        for col in range(0, num_cols):
            K.append(len(set(data[:, col].flatten())))
        self._level_counts = K
        return K

    def get_encoded_x(self):
        return self._encoded_x

    def fit(self, center_design=False):
        # compute gamma
        # cc = self._w_combo_counts
        # W = np.diag(cc)
        # X = self._x_tilde
        # z = self._z

        n = self._invert_this.shape[0]

        # D = np.identity(n) * self._lambda

        # if center_design:
        #     X = self._x_tilde - np.mean(self._x_tilde)

        # recompute this product since the computational shortcut we used before does not work with the centered X

        # self._invert_this = np.transpose(X) @ W @ X

        # If we are using a penalty we want to center the deisgn matrix for the penalized regression so that
        # we do not end up having an intercept (we do not want to penalize an intercept so we just remove it)
        # if self._penalty != None:
        #     Xp = self._invert_this - np.mean(self._invert_this)
        # else:
        #     Xp = self._invert_this

        invert_this = self._invert_this

        # if self._penalty == None:
        #     H = inv(invert_this) @ np.transpose(X) @ W
        #     self._gamma = H @ z
        # else:
        #     if self._penalty == "l2":
        #         clf = Ridge()
        #     elif self.set_penalty == 'l1':
        #         clf = Lasso()

        Xz = self._Xz

        g = inv(invert_this) @ Xz

        self._gamma = g

        # param_grid = [{"alpha": self._lambda}]
        # grid_search = GridSearchCV(
        # clf, param_grid, cv=3, scoring="accuracy", verbose=10)
        # If we are running regression we want to center z so that there is no intercept, to match our removale
        # of the intercept for the penalized analytical model
        # z = z - np.mean(z)

        # grid_search.fit(X, z, sample_weight=[1 / x for x in cc])

        # final_clf = grid_search.best_estimator_
        # params = final_clf.coef_

        # H = inv(invert_this) @ np.transpose(X) @ W
        # g = H @ z
        # print("done")

    def get_gamma(self):
        return self._gamma

    def get_simple_counts(self):
        """
        :return: List of successes for each combination, and number of each combination total
        """
        return self._success_counts, self._w_combo_counts

    def set_y(self, y):
        self._y = y
