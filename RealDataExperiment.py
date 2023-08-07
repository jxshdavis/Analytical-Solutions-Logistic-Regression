

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

    def __init__(self, data_pathname = '/Users/joshdavis/Desktop/Meng Research/Analytical-Solutions-Logistic-Regression/datsets/cancer-data.csv', response_col_name, num_cols):
        response_col_name = 'diagnosis'
        self._iterative_model = None
        self._analytic_model = None
        self._data = pd.read_csv(data_pathname)
        self._y = pd.get_dummies(self._data[response_col_name], drop_first=True).astype(int)
        self._num_cols = num_cols
        self.clean_df()
        self._tf = copy.deepcopy(self._data)
        num_levels = 7
        numerical_columns = self._tf.select_dtypes(include=['int', 'float']).columns.tolist()
        for col in numerical_columns:
            self._tf [col] = pd.cut(self._tf [col], bins=num_levels).astype(str)


    def clean_df(self):
        self._data = self._data.drop(columns=['Unnamed: 32', 'id', 'diagnosis']).iloc[:, :number_cols]

    def fit_models(self):
        anal_sol = AnalyticalSolution.AnalyticalSolution(x=self._tf, y=np.array(self._y))
        anal_sol.transform_response()
        anal_sol.fit()

        iterative_model = LogisticRegression(solver='liblinear')

        iterative_model.fit(self._data, y)


    def get_params(self):
        lib_lin_sol = list(iterative_model.intercept_) + list(iterative_model.coef_[0])
        lib_lin_sol = [round(x, 3) for x in lib_lin_sol]

        print(lib_lin_sol)
        print(anal_sol.get_gamma())

# convert y to 0 and 1's








