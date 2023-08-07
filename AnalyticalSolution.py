import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import bernoulli
from scipy.special import expit
import math
import copy
import pandas as pd
from numpy.linalg import inv


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

    def __init__(self, x, lamb=0, nudge=10 ** (-5), y=np.array([0])):
        """
        @param x:
        @param lamb:
        @param nudge:
        @param y:
        """

        self._x = x
        self._y = y
        self._lambda = lamb
        self._x_tilde = None
        self._invert_this = None
        self._z = None
        self._level_counts = None
        self._encoded_x = self.encode_df()
        self.transform_design(self)
        self._w_combo_counts = None
        self._gamma = None
        self._nudge = nudge
        self._success_counts = None

    def set_lambda(self, lamb):
        self._lambda = lamb

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

    def transform_response(self):

        # perform 1-hot encoding

        df_encoded = self._encoded_x

        combinations = self._x_tilde.values.tolist()
        self._combinations = combinations

        response = [x[0] for x in self._y.tolist()]

        # start with row in df
        # method 1: faster with high number of regressors and levels

        num_reg = self._x.shape[1]

        num_combo = len(combinations)
        combo_count = [0] * num_combo
        num_reg = self._x.shape[1]

        num_combo = len(combinations)
        combo_count = [0] * num_combo
        success_counts = [0] * num_combo

        levels = self._level_counts

        # We can also consturct the (X_tilde^T W X_tilde) matrix here
        mat_size = 1 - len(levels) + sum(levels)
        print()

        invert_this = np.zeros((mat_size, mat_size))

        for index, row in df_encoded.iterrows():
            one_indicies = []
            sub_row_start_idx = 0
            row = row.tolist()
            tilde_idx = 0

            for i in range(num_reg):
                Ki = levels[i]
                sub_row = row[sub_row_start_idx:sub_row_start_idx + Ki - 1]

                one_index = sub_row.index(1) if 1 in sub_row else -1

                if one_index != -1:
                    one_indicies.append(one_index + sub_row_start_idx)

                if one_index >= 1:
                    tilde_idx += (Ki - one_index - 1) * Ki ** (i)
                elif one_index == -1:
                    tilde_idx += (Ki - 1) * Ki ** (i)

                # update the information to get next subrow
                sub_row_start_idx = sub_row_start_idx + Ki - 1

            # update the invert_this matrix: \tilde X^T W \tilde X
            invert_this[0, 0] += 1
            for row in one_indicies:
                invert_this[0, row + 1] += 1
                invert_this[row + 1, 0] += 1
                invert_this[row + 1, row + 1] += 1
                for col in one_indicies:
                    # col0 and row0
                    if row != col:
                        invert_this[row + 1, col + 1] += 1

            self._invert_this = invert_this

            # update the counts of each row of the data
            combo_count[tilde_idx] += 1

            # update the success counts of the row of data
            success_counts[tilde_idx] += response[index]

        self._w_combo_counts = combo_count

        self._success_counts = success_counts
        success_frequencies = []
        for index in range(len(success_counts)):
            if combo_count[index] != 0:
                success_frequencies.append(success_counts[index] / combo_count[index])
            else:
                success_frequencies.append(0)
        # print("success Freq")
        E = self._nudge

        nudged = []
        for x in success_frequencies:
            if x == 0:
                nudged.append(E)
            elif x == 1:
                nudged.append(1 - E)
            else:
                nudged.append(x)

        nudged = np.array(nudged)
        nudged = np.log(nudged / (1 - nudged))

        # print("z:")
        # print(pd.DataFrame(nudged))

        self._z = pd.DataFrame(nudged)

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

    def fit(self):
        # compute gamma
        cc = self._w_combo_counts
        W = np.diag(cc)
        X = self._x_tilde
        z = self._z
        n = self._invert_this.shape[0]
        # D = np.identity(n) * self._lambda
        invert_this = self._invert_this + np.identity(n) * self._lambda
        H = inv(invert_this) @ np.transpose(X) @ W
        self._gamma = H @ z

    def get_gamma(self):
        return self._gamma

    def get_simple_counts(self):
        """
        :return: List of successes for each combination, and number of each combination total
        """
        return self._success_counts, self._w_combo_counts

    def set_y(self, y):
        self._y = y
