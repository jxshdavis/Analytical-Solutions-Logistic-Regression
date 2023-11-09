

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import bernoulli
from scipy.special import expit
import math
import copy
import pandas as pd
from numpy.linalg import inv
import AnalyticalSolution
import ExperimentSimulations

class RealDataExperiment():

    def __init__(self, response_col_name, num_cols, data_pathname, num_levels, drop_these):
        self._iterative_model = None
        self._analytic_model = None
        self._num_cols = num_cols
        self._data = pd.read_csv(data_pathname)
        self._y = pd.get_dummies(self._data[response_col_name], drop_first=True).astype(int)

        self.clean_df(drop_these)

        self._tf = copy.deepcopy(self._data)

        numerical_columns = self._tf.select_dtypes(include=['int', 'float']).columns.tolist()
        for col in numerical_columns:
            self._tf[col] = pd.cut(self._tf [col], bins=num_levels).astype(str)


    def clean_df(self, drop_these):
        self._data = self._data.drop(columns=drop_these).iloc[:, :self._num_cols]

    def fit_models(self, center_design=False):
        print()
        anal_sol = AnalyticalSolution.AnalyticalSolution(x=self._tf, y=np.array(self._y))
        anal_sol.transform_response()
        anal_sol.fit()

        iterative_model = LogisticRegression(solver='saga')

        centered_y = np.array(self._y) - np.mean(np.array(self._y))

        iterative_model.fit(self._data, np.array(self._y))

        self._iterative_model = iterative_model
        self._analytic_model = anal_sol

    def get_params(self):
        lib_lin_sol = list(self._iterative_model.intercept_) + list(self._iterative_model.coef_[0])
        lib_lin_sol = [round(x, 3) for x in lib_lin_sol]

        print(lib_lin_sol)
        print(self._analytic_model.get_gamma())

# convert y to 0 and 1's








