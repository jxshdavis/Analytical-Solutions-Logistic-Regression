
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


sim = ExperimentSimulations.ExperimentSimulations(num_trials = 3, num_regressors=3, num_levels=3, min_number_obs=2,
                                                  max_number_obs=5, number_observations_sizes=5, beta_range=[-10,10])
sim.plot_errors()
sim.plot_times()






# For real data, first we want to discretize the numeric covariates

# # Load the dataset
# df = pd.read_csv('/Users/joshdavis/Desktop/Meng Research/Analytical-Solutions-Logistic-Regression/datasets/cancer-data.csv')
# y = df['diagnosis']
#
# number_cols = 6
#
# # Drop Unnamed: 32, patient id and response
# df = df.drop(columns=['Unnamed: 32', 'id', 'diagnosis']).iloc[:, :number_cols]
#
#
# tf = copy.deepcopy(df)
# num_levels = 2
# numerical_columns = tf.select_dtypes(include=['int', 'float']).columns.tolist()
#
# # discretize
# for col in numerical_columns:
#     tf[col] = pd.cut(tf[col], bins=num_levels).astype(str)
#
# # df.insert(0, 'intercept', 1)
# # tf.insert(0, 'intercept', 1)
#
#
# # convert y to 0 and 1's
# y = pd.get_dummies(y, drop_first=True).astype(int)
#
#
# print(tf.shape)
# anal_sol = AnalyticalSolution.AnalyticalSolution(x = tf, y = np.array(y))
# anal_sol.transform_response()
# anal_sol.fit()
#
# iterative_model = LogisticRegression(solver='liblinear')
# df_encoded = pd.get_dummies(tf, drop_first=True)
# df_encoded_1 = df_encoded.astype(int)
# iterative_model.fit(df_encoded_1, y)
#
#
# lib_lin_sol = list(iterative_model.intercept_) + list(iterative_model.coef_[0])
# lib_lin_sol = [round(x, 3) for x in lib_lin_sol]
#
# print(anal_sol.get_gamma())
# print(lib_lin_sol)


