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
import seaborn as sns
import RealDataExperiment

#
# min_x = np.log(2)/np.log(10)
# max_x = np.log(8)/np.log(10)

min_x = 3
max_x = 4

for num_reg in [3]:
    sim = ExperimentSimulations.ExperimentSimulations(num_trials=50, num_regressors=num_reg, num_levels=2, num_obs=10**3, log_min_x=min_x,
                                                      log_max_x=max_x, number_x_ticks=5, beta_range=[-3, 3], penalty=None)
    sim.run_observation_sim()


#
# cancer_experiment = RealDataExperiment.RealDataExperiment(response_col_name='diagnosis',
#                                                           num_cols=2,
#                                                           data_pathname='/Users/joshdavis/Desktop/Meng Research/Analytical-Solutions-Logistic-Regression/datsets/cancer-data.csv',
#                                                           num_levels=2, drop_these=['Unnamed: 32', 'id', 'diagnosis'])
#
#
# cancer_experiment.fit_models()
# cancer_experiment.get_params()


# real data stuff below


# For real data, first we want to discretize the numeric covariates
#
# # Load the dataset
# df = pd.read_csv('/Users/joshdavis/Desktop/Meng Research/Analytical-Solutions-Logistic-Regression/datsets/cancer-data.csv')
# y = df['diagnosis']
#
# number_cols = 6
#
# # Drop Unnamed: 32, patient id and response
# df = df.drop(columns=['Unnamed: 32', 'id', 'diagnosis']).iloc[:, :number_cols]
#
#
# tf = copy.deepcopy(df)
# num_levels = 7
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
