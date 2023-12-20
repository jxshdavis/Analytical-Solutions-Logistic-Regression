

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

    def __init__(self, response_col_name, num_cols, data_pathname, num_levels, drop_these, penalty):

        drop_these = drop_these + [response_col_name]
        self._penalty = penalty
        self._iterative_model = None
        self._analytic_model = None
        self._num_cols = num_cols
        self._data = pd.read_csv(data_pathname)
        self._y = pd.get_dummies(
            self._data[response_col_name], drop_first=True).astype(int)

        # make sure to drop the response collumn from the design matrix
        self.clean_df(drop_these)

        self._tf = copy.deepcopy(self._data)

        numerical_columns = self._tf.select_dtypes(
            include=['int', 'float']).columns.tolist()
        for col in numerical_columns:
            self._tf[col] = pd.cut(self._tf[col], bins=num_levels).astype(str)

    def clean_df(self, drop_these):
        self._data = self._data.drop(
            columns=drop_these).iloc[:, :self._num_cols]
        last = self._data.columns[-1]
        print(f"Last Col: {last}")
        if last == "Female":
            print("stop")
            pass

    def fit_models(self, center_design=False):

        # need to add the
        # self._batch_integers, self._individual_integers, self._individual_samples, self._batch_samples
        # parameter
        num_samples = self._tf.shape[0]
        sample_size_info = [None], [None], num_samples,   0

        anal_sol = AnalyticalSolution.AnalyticalSolution(
            x=self._tf, y=np.array(self._y), sample_size_info=sample_size_info, lamb=self._penalty)

        anal_sol.transform_response()
        # anal_sol.fit()

        if penalty == 0:
            iterative_model = LogisticRegression(solver='lbfgs', penalty=None)
        else:
            iterative_model = LogisticRegression(
                solver='lbfgs', penalty='l2', C=1/penalty)

        # centered_y = np.array(self._y) - np.mean(np.array(self._y))

        iterative_model.fit(self._data, np.array(self._y).ravel())

        self._iterative_model = iterative_model
        self._analytic_model = anal_sol

    def get_params(self):
        lib_lin_sol = list(self._iterative_model.intercept_) + \
            list(self._iterative_model.coef_[0])

        lib_lin_sol = [x for x in lib_lin_sol]

        return lib_lin_sol, self._analytic_model.get_gamma()
        print("iterative sol and gamma:")
        print(lib_lin_sol)
        print(self._analytic_model.get_gamma())


def plot_real_data_experiment(PATHNAME, response_col_nam, dataset_name, penalty, drop_these=[]):

    data = pd.read_csv(PATHNAME)
    max_num_cols = data.shape[1]
    # max_num_cols = min(max_num_cols, 10)
    mses = []
    gamma_avg_magnitudes = []
    iterative_beta_avg_magnitudes = []
    for num_cols in range(1, max_num_cols):
        print(str(num_cols)+" columns.")
        real_data_experimenrt = RealDataExperiment(response_col_name=response_col_name,
                                                   num_cols=num_cols,
                                                   data_pathname=PATHNAME,
                                                   num_levels=2, drop_these=drop_these,
                                                   penalty=penalty)
        real_data_experimenrt.fit_models()
        iterative_beta, gamma = real_data_experimenrt.get_params()

        iterative_beta = np.array(iterative_beta)
        gamma = np.array(gamma)
        mse = ((iterative_beta-gamma)**2).mean()

        gamma_avg_magnitude = (gamma**2).mean()
        iterative_beta_avg_magnitude = (iterative_beta**2).mean()
        gamma_avg_magnitudes.append(gamma_avg_magnitude)
        iterative_beta_avg_magnitudes.append(iterative_beta_avg_magnitude)

        mses.append(mse)
        print("MSE: "+str(mse))
        print()
    x_vals = range(1, max_num_cols)

    plt.plot(x_vals, mses, label='MSE', color="darkred", linestyle=':')
    plt.plot(x_vals, gamma_avg_magnitudes,
             label='Avg Magnitude of Analytic Est.', linewidth=4, alpha=0.6)
    plt.plot(x_vals, iterative_beta_avg_magnitudes,
             label='Avg Magnitude of iterative MLE', color="green", linewidth=4,
             alpha=0.6)

    plt.yscale('log')
    # Adding labels and title
    plt.xlabel('Number of Regressors Used')
    plt.ylabel('MSE')
    plt.suptitle('MSE of MLE and Analytic Estimator: '+dataset_name)
    plt.title("Ridge Penalty. Lambda = " + str(penalty), fontsize=10)
    plt.legend()
    plt.savefig(dataset_name+" Real Data. RidgePenalty=" +
                str(penalty)+"_.png", dpi=300)


# convert y to 0 and 1's


# adult dataset
PATHNAME = '/Users/joshdavis/Desktop/STAT RESEARCH/Analytical-Solutions-Logistic-Regression/datsets/adult_data.csv'
response_col_name = 'Over50K'
dataset_name = "Adult Dataset"

# drop_these = ["JobAgriculture",
#                 "NeverMarried",
#                 "WorkHrsPerWeek_geq_50",
#                 "Female",
#                 "OtherRace",
#                 "NativeUSorCanada",
#                 "AnyCapitalGains",
#                 "NativeImmigrant"]

drop_these = []


# bank dataset
PATHNAME = '/Users/joshdavis/Desktop/STAT RESEARCH/Analytical-Solutions-Logistic-Regression/datsets/bank_data.csv'
response_col_name = 'sign_up'
dataset_name = "Bank Dataset"

# Breastcancer Dataset
PATHNAME = '/Users/joshdavis/Desktop/STAT RESEARCH/Analytical-Solutions-Logistic-Regression/datsets/breastcancer_data.csv'
response_col_name = 'Benign'
dataset_name = "Breast Cancer Dataset"


# mammo Dataset
PATHNAME = '/Users/joshdavis/Desktop/STAT RESEARCH/Analytical-Solutions-Logistic-Regression/datsets/mammo_data.csv'
response_col_name = 'Malignant'
dataset_name = "Mammo Dataset"


# mushroom dataset
PATHNAME = '/Users/joshdavis/Desktop/STAT RESEARCH/Analytical-Solutions-Logistic-Regression/datsets/mushroom_data.csv'
response_col_name = 'poisonous'
dataset_name = "Muschroom Dataset"


penalty = 0
plot_real_data_experiment(
    PATHNAME=PATHNAME, response_col_nam=response_col_name, dataset_name=dataset_name, drop_these=drop_these, penalty=penalty)


# real_data_experimenrt = RealDataExperiment(response_col_name=response_col_name,
#                                            num_cols=10,
#                                            data_pathname=PATHNAME,
#                                            num_levels=2, drop_these=[])


# real_data_experimenrt.fit_models()
# iterative_beta, gamma = real_data_experimenrt.get_params()

# iterative_beta = np.array(iterative_beta)
# gamma = np.array(gamma)


# print(((iterative_beta-gamma)**2).mean())
# print(((iterative_beta-gamma)))


# real data stuff below


# For real data, first we want to discretize the numeric covariates

# Load the dataset


# df = pd.read_csv(PATHNAME)
# y = df['Over50K']


# number_cols = 6

# # Drop Unnamed: 32, patient id and response
# df = df.drop(columns=['Unnamed: 32', 'id', 'diagnosis']).iloc[:, :number_cols]


# tf = copy.deepcopy(df)
# num_levels = 7
# numerical_columns = tf.select_dtypes(include=['int', 'float']).columns.tolist()

# # discretize
# for col in numerical_columns:
#     tf[col] = pd.cut(tf[col], bins=num_levels).astype(str)

# # df.insert(0, 'intercept', 1)
# # tf.insert(0, 'intercept', 1)


# # convert y to 0 and 1's
# y = pd.get_dummies(y, drop_first=True).astype(int)


# print(tf.shape)
# anal_sol = AnalyticalSolution.AnalyticalSolution(x=tf, y=np.array(y))
# anal_sol.transform_response()
# anal_sol.fit()

# iterative_model = LogisticRegression(solver='liblinear')
# df_encoded = pd.get_dummies(tf, drop_first=True)
# df_encoded_1 = df_encoded.astype(int)
# iterative_model.fit(df_encoded_1, y)


# lib_lin_sol = list(iterative_model.intercept_) + list(iterative_model.coef_[0])
# lib_lin_sol = [round(x, 3) for x in lib_lin_sol]

# print(anal_sol.get_gamma())
# print(lib_lin_sol)
