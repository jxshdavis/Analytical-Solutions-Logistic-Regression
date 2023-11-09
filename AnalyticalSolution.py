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
        self._Xz = None
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

        # perform 1-hot encoding
        df_encoded = self._encoded_x

        # combinations = self._x_tilde.values.tolist()
        # self._combinations = combinations

        response = [x[0] for x in self._y.tolist()]

        # start with row in df
        # method 1: faster with high number of regressors and levels

        num_reg = self._x.shape[1]

        combo_count = defaultdict(int)

        num_reg = self._x.shape[1]

        success_counts = defaultdict(int)

        levels = self._level_counts

        # We can also consturct the (X_tilde^T W X_tilde) matrix here
        mat_size = 1 - len(levels) + sum(levels)
        # print()

        invert_this = np.zeros((mat_size, mat_size))

        list_of_tilde_indicies = []
        list_of_one_indicies = []

        start_time = timer()
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

            # update the counts of each row of the data
            list_of_one_indicies.append(one_indicies)
            list_of_tilde_indicies.append(tilde_idx)
            combo_count[tilde_idx] += 1

            # update the success counts of the row of data
            success_counts[tilde_idx] += response[index]

        # time to compute our success frequencies
        end_time = timer()
        print("first time chunk: compute success coutnts and lists of indicies")
        print(end_time-start_time)

        start_time = timer()
        success_frequencies = defaultdict(int)
        for tilde_idx in list(success_counts.keys()):
            if combo_count[tilde_idx] != 0:
                success_frequencies[tilde_idx] = success_counts[tilde_idx] / \
                    combo_count[tilde_idx]

        for design_idx in range(len(list_of_one_indicies)):
            one_indicies = list_of_one_indicies[design_idx]
            tilde_idx = list_of_tilde_indicies[design_idx]

            f = success_frequencies[tilde_idx]
            alpha = f*(1-f)
            invert_this[0, 0] += alpha

            for row in one_indicies:
                invert_this[0, row + 1] += alpha
                invert_this[row + 1, 0] += alpha
                invert_this[row + 1, row + 1] += alpha
                for col in one_indicies:
                    # col0 and row0
                    if row != col:
                        invert_this[row + 1, col + 1] += alpha
        end_time = timer()
        print("second time chunk: compute invert this")
        print(end_time-start_time)

        # self._w_combo_counts = combo_count
        self._invert_this = invert_this
        self._success_counts = success_counts

        # keys = sorted(success_frequencies.keys())
        # sf = [success_frequencies[idx] for idx in keys]

        # find X^T z directly here

        start_time = timer()
        set_of_tilde_indicies = set(list_of_tilde_indicies)
        Xz = np.array([0] * mat_size)

        unique_rows = df_encoded.drop_duplicates()
        # Optionally reset the index
        unique_rows = unique_rows.reset_index(drop=True)

        # print(unique_rows)

        unique_indicies = []
        unique_set = set()

        for idx in list_of_tilde_indicies:
            if idx not in unique_set:
                unique_set.add(idx)
                unique_indicies.append(idx)

        for index, row in unique_rows.iterrows():
            tilde_idx = unique_indicies[index]
            row = row.values.tolist()
            f = success_frequencies[tilde_idx]
            Xz = Xz + combo_count[tilde_idx]*f * \
                (1-f)*np.array([1]+row) * math.log(f/(1-f))

        end_time = timer()
        print("third time chunk: compute Xz")
        print(end_time-start_time)

        self._Xz = pd.DataFrame(Xz)

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
