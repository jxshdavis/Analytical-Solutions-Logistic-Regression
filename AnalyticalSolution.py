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
    Generates the analytical solution when there are multiple regressors.
    """

    def encode_df(self):
        df = copy.deepcopy(self._x)
        df_encoded = pd.get_dummies(df, drop_first=True)
        df_encoded = df_encoded.astype(int)
        return df_encoded

    def __init__(self, x, nudge = 10**(-5), y=np.array([0])):
        self._x = x
        self._y = y
        self._x_tilde = None
        self._z = None
        self._level_counts = None
        self._encoded_x = self.encode_df()
        self.transform_design(self)
        self._w_combo_counts = None
        self._gamma = None
        self._nudge = nudge
        self._success_counts = None


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

        num_reg = self._x.shape[1]

        # num_combo = len(combinations)
        # combo_count = [0] * num_combo
        # success_counts = [0] * num_combo
        #
        # levels = self._level_counts
        # for index, row in df_encoded.iterrows():
        #     sub_row_start_idx = 0
        #     row = row.tolist()
        #     tilde_idx = 0
        #
        #     for i in range(num_reg):
        #         Ki = levels[i]
        #         sub_row = row[sub_row_start_idx:sub_row_start_idx + Ki-1]
        #
        #         one_index = sub_row.index(1) if 1 in sub_row else -1
        #
        #         if one_index >= 1:
        #             tilde_idx += (Ki-one_index-1)*Ki**(i)
        #         elif one_index ==-1:
        #             tilde_idx += (Ki-1)*Ki**(i)
        #
        #         # update the information to get next subrow
        #         sub_row_start_idx = sub_row_start_idx + Ki - 1
        #
        #     combo_count[tilde_idx] += 1
        #     success_counts[tilde_idx] += response[index]

        combo_count = []
        success_counts = []
        for combo in combinations:
            # Count the occurrences of the row

            target_row = combo[1:]
            inner = df_encoded.apply(lambda row: row.tolist() == target_row, axis=1)
            indices = df_encoded[inner].index
            success_count = 0
            combo_count.append(len(indices))
            for index in indices:
                if response[index] == 1:
                    success_count += 1

            success_counts.append(success_count)

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
        cc = self._w_combo_counts

        W = np.diag(cc)
        X = self._x_tilde
        z = self._z
        invert_this = np.transpose(X) @ W @ X
        H = inv(invert_this) @ np.transpose(X) @ W
        self._gamma = H @ z

        # print("W")
        # print(W)

    def get_gamma(self):
        return self._gamma

    def get_simple_counts(self):
        """
        :return: List of successes for each combination, and number of each combination total
        """
        return self._success_counts, self._w_combo_counts

    def set_y(self, y):
        self._y = y

