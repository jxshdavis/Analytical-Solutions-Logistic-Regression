

import matplotlib.pyplot as plt
from matplotlib.table import Table
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression


class RealDataExperiment():

    def __init__(self, response_col_name, num_cols, data_pathname, drop_these, num_levels=2, penalty=0, train_split_proportion=1):

        self._penalty = penalty
        self._iterative_model = None
        self._analytic_model = None
        self._num_cols = num_cols
        self._tilde_X = None
        self._all_unique_rows = None
        self._weights = None
        self._count_per_row = None
        self._final_design = None

        self._data = pd.read_csv(data_pathname)

        # get response
        self._y = pd.get_dummies(
            self._data[response_col_name], drop_first=True).astype(int)

        # make sure to drop the response collumn from the design matrix
        drop_these = drop_these + [response_col_name]
        self.clean_df(drop_these)

        # turn into 0s and 1s

        self._tf = copy.deepcopy(self._data)

        numerical_columns = self._tf.select_dtypes(
            include=['int', 'float']).columns.tolist()

        for col in numerical_columns:
            self._tf[col] = pd.cut(self._tf[col], bins=num_levels).astype(str)

        self._col_names = self._tf.columns

        self._tf = pd.get_dummies(self._tf, drop_first=True).astype(int)

        if train_split_proportion < 1:
            self._tf, self._X_test, self._y, self._y_test = train_test_split(
                self._tf, self._y, test_size=1-train_split_proportion)

    def get_weights(self):
        return self._weights

    def get_final_design(self):
        return self._final_design

    def get_count_per_row(self):
        return self._count_per_row

    def get_column_names(self):
        return self._col_names

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
            x=self._tf, y=np.array(self._y), sample_size_info=sample_size_info, lamb=self._penalty, drop_under_count=[0])

        anal_sol.transform_response()
        # anal_sol.fit()
        self._final_design = anal_sol.get_final_design()

        self._count_per_row = anal_sol.get_count_per_row()

        self._all_unique_rows = anal_sol.get_all_unique_rows()
        self._weights = anal_sol.get_weights()

        if penalty == 0:
            iterative_model = LogisticRegression(solver='lbfgs', penalty=None)
        else:
            iterative_model = LogisticRegression(
                solver='lbfgs', penalty='l2', C=1/penalty)

        # centered_y = np.array(self._y) - np.mean(np.array(self._y))

        iterative_model.fit(self._tf, np.array(self._y).ravel())

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

    def get_test_set_prediction_errors(self):

        test_encoded = pd.get_dummies(self._X_test, drop_first=True)

        df_encoded = df_encoded.astype(int)

    def get_all_unique_rows(self):
        return self._all_unique_rows

    def get_iterative_df(self):
        return self._tf


def plot_mse_v_num_predictors(PATHNAME, response_col_nam, dataset_name, penalty, drop_these=[]):

    data = pd.read_csv(PATHNAME)
    max_num_cols = data.shape[1]
    # max_num_cols = min(max_num_cols, 10)
    mses = []
    gamma_avg_magnitudes = []
    iterative_beta_avg_magnitudes = []
    for num_cols in range(1, max_num_cols):
        print(str(num_cols)+" columns.")
        real_data_experiment = RealDataExperiment(response_col_name=response_col_name,
                                                  num_cols=num_cols,
                                                  data_pathname=PATHNAME,
                                                  num_levels=2, drop_these=drop_these,
                                                  penalty=penalty)
        real_data_experiment.fit_models()
        iterative_beta, gamma = real_data_experiment.get_params()

        iterative_beta = np.array(iterative_beta)
        gamma = np.array(gamma)

        print("gamma")
        print(gamma)
        print("iterative beta")
        print(iterative_beta)

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


def prediction_accuracy_compariosn(num_trials, train_split_proportion, PATHNAME, response_col_nam, dataset_name, penalty, drop_these=[]):
    data = pd.read_csv(PATHNAME)
    max_num_cols = data.shape[1]

    # set up the real data experiment object to get estimates
    real_data_experimenrt = RealDataExperiment(response_col_name=response_col_name,
                                               num_cols=max_num_cols,
                                               data_pathname=PATHNAME,
                                               num_levels=2, drop_these=drop_these,
                                               penalty=penalty, train_split_proportion=train_split_proportion)
    real_data_experimenrt.fit_models()

    prediction_auc = real_data_experimenrt.get_test_set_prediction_errors()


def beta_standard_error_comparison(PATHNAME, response_col_nam, dataset_name, penalty, num_beta_samples=1000, drop_these=[]):
    data = pd.read_csv(PATHNAME)
    max_num_cols = data.shape[1]

    # set up the real data experiment object to get estimates
    real_data_experiment = RealDataExperiment(response_col_name=response_col_name,
                                              num_cols=max_num_cols,
                                              data_pathname=PATHNAME,
                                              num_levels=2, drop_these=drop_these,
                                              penalty=penalty)

    real_data_experiment.fit_models()

    iterative_beta, gamma = real_data_experiment.get_params()
    final_design = real_data_experiment.get_final_design()

    gamma = np.array(gamma[0])

    iterative_beta = np.array(iterative_beta)
    # gamma = np.array(gamma)

    all_unique_rows = real_data_experiment.get_all_unique_rows()

    z_hat = (all_unique_rows @ gamma)

    # p_hat = iterative_beta @

    gamma_predicted_success_prob = 1/(1+np.exp(-1*z_hat))

    tf = real_data_experiment.get_iterative_df()
    tf.insert(0, 'Intercept', 1)

    beta_MLE_predicted_success_prob = 1 / \
        (1+np.exp(-1*all_unique_rows @ iterative_beta))

    analytic_betas = []
    iterative_monte_Caro_betas = []

    count_per_row = real_data_experiment.get_count_per_row()

    # sample betas for both the analytic and iterative estimated success probabilities
    for i in range(num_beta_samples):
        sampled_gamma, sampled_iterative_beta = sample_betas(
            gamma_predicted_success_prob, beta_MLE_predicted_success_prob, all_unique_rows, count_per_row, penalty=0)

        analytic_betas.append(sampled_gamma[0])

        iterative_monte_Caro_betas.append(sampled_iterative_beta)

        print(round((i+1)/num_beta_samples, 4))

    # set back to arrays
    analytic_betas = np.array(analytic_betas)
    iterative_monte_Caro_betas = np.array(iterative_monte_Caro_betas)

    analytic_monte_carlo_means = np.mean(analytic_betas, axis=0)

    iterative_monte_carlo_means = np.mean(iterative_monte_Caro_betas, axis=0)

    analytic_beta_monte_carlo_standard_error = np.var(
        analytic_betas, axis=0, ddof=1)**(0.5)

    iterative_beta_monte_carlo_standard_error = np.var(
        iterative_monte_Caro_betas, axis=0, ddof=1)**(0.5)

    standard_error_multiplier = 1.96

    W = real_data_experiment.get_weights()

    # asymptotic SE

    XWX = np.linalg.inv(final_design.T @ W @ final_design)
    asymptotic_standard_error = np.array(
        [XWX[i][i] for i in range(len(XWX))])**(.5)

    # asymptotic_standard_error = analytic_beta_monte_carlo_standard_error

    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(15, 15))

    # MC analytic error bars
    ax1.errorbar(y=[.2 + i for i in range(len(analytic_monte_carlo_means))], x=gamma,
                 xerr=standard_error_multiplier*analytic_beta_monte_carlo_standard_error, fmt='o', capsize=6, color="blue", label='Analytic MC SE')

    # iterative error bars
    ax1.errorbar(y=[.3 + i for i in range(len(iterative_monte_carlo_means))], x=iterative_beta,
                 xerr=standard_error_multiplier*iterative_beta_monte_carlo_standard_error, fmt='o', linewidth=2, capsize=5, color="green", label='Iterative MC SE')

    # asymptotic error bars
    ax1.errorbar(y=range(len(analytic_monte_carlo_means)), x=gamma,
                 xerr=standard_error_multiplier*asymptotic_standard_error, fmt='o', capsize=5, color="red", linewidth=2, alpha=.5, label='Asymptotic SE')

    names = ["Intercept"]+list(real_data_experiment.get_column_names())

    title = "Beta Confidence Intervals: Mean pm ", str(
        standard_error_multiplier), " SE."
    ax1.set_title(title)

    ax1.set_yticks(range(len(analytic_monte_carlo_means)))
    ax1.set_yticklabels(names)
    # ax1.tick_params(axis='y')

    ax1.axvline(x=0, color='black', linestyle='--')

    ax1.legend()

    table_array = np.empty((len(names), 0))

    # Generate a new column
    table_array = np.column_stack((gamma, analytic_beta_monte_carlo_standard_error,
                                   iterative_beta, iterative_beta_monte_carlo_standard_error))
    table_array = np.round(table_array, decimals=4)
    names.reverse()
    # table_array = np.column_stack((np.array(names), table_array))

    # Create table
    table = ax2.table(cellText=table_array[::-1], loc='center', cellLoc='center', colLabels=[
                      "Analytic Value", "Analytic SE", "Iterative Value", "Iterative SE"], rowLabels=names
                      )
    bbox = [.13, .03, 1, 1]
    table.scale(.9, 3)  # Increase size by scaling

    ax2.set_title(
        "Estimated Regression Coefficients with MC Standad Error", y=1.03)
    # Adjusting font size
    table.auto_set_font_size(False)
    table.set_fontsize(11)  # Set font size

    # Hide axes
    ax2.axis('off')

    # Add table to plot
    ax2.add_table(table)

    plt.savefig("BreastCancerModel Standard Errors.png", dpi=300)

    print("done")

    # iterative_predicted_success_prob = 1/(1+np.exp(z_hat))


def sample_betas(gamma_predicted_success_prob, beta_MLE_predicted_success_prob, all_unique_rows, count_per_row, penalty=0):

    y_gamma = []
    y_iterative = []
    X = []
    # remove the intercept column since the solver will automaticall acount for the need
    design = all_unique_rows[:, 1:]
    for row, gamma_prob, MLE_prob, num_samples in zip(design, gamma_predicted_success_prob, beta_MLE_predicted_success_prob, count_per_row):
        y_gamma += np.random.binomial(n=1,
                                      p=gamma_prob, size=num_samples).tolist()
        y_iterative += np.random.binomial(n=1,
                                          p=MLE_prob, size=num_samples).tolist()
        X += [row]*num_samples

    sample_size_info = [None], [None], num_samples,   0

    X = pd.DataFrame(X)
    y_gamma = np.array(y_gamma).reshape(-1, 1)

    anal_sol = AnalyticalSolution.AnalyticalSolution(
        x=X, y=y_gamma, sample_size_info=sample_size_info, drop_under_count=[0])

    anal_sol.transform_response()
    # anal_sol.fit()

    if penalty == 0:
        iterative_model = LogisticRegression(solver='lbfgs', penalty=None)
    else:
        iterative_model = LogisticRegression(
            solver='lbfgs', penalty='l2', C=1/penalty)

        # centered_y = np.array(self._y) - np.mean(np.array(self._y))

    iterative_model.fit(X, y_iterative)

    iterative_beta = list(iterative_model.intercept_) + \
        list(iterative_model.coef_[0])

    return anal_sol.get_gamma(), np.array(iterative_beta)

    # get predicted success probabilities


    # draw 1000 samples from the binomial distributution to ~simulate samples of beta
    # compute the variance for each beta
    # compute the varinace for beta based on the asympotitic distribution
    # variance of the standard estimator
    # create plot of compiled information
# # adult dataset
# PATHNAME = '/Users/joshdavis/Desktop/STAT RESEARCH/Analytical-Solutions-Logistic-Regression/datsets/adult_data.csv'
# response_col_name = 'Over50K'
# dataset_name = "Adult Dataset"
# drop_these = ["JobAgriculture",
#                 "NeverMarried",
#                 "WorkHrsPerWeek_geq_50",
#                 "Female",
#                 "OtherRace",
#                 "NativeUSorCanada",
#                 "AnyCapitalGains",
#                 "NativeImmigrant"]
drop_these = []


# # bank dataset
# PATHNAME = '/Users/joshdavis/Desktop/STAT RESEARCH/Analytical-Solutions-Logistic-Regression/datsets/bank_data.csv'
# response_col_name = 'sign_up'
# dataset_name = "Bank Dataset"

# Breastcancer Dataset
PATHNAME = '/Users/joshdavis/Desktop/STAT RESEARCH/Analytical-Solutions-Logistic-Regression/datsets/breastcancer_data.csv'
response_col_name = 'Benign'
dataset_name = "Breast Cancer Dataset"
drop_these = ["Mitoses"]

# # mammo Dataset
# PATHNAME = '/Users/joshdavis/Desktop/STAT RESEARCH/Analytical-Solutions-Logistic-Regression/datsets/mammo_data.csv'
# response_col_name = 'Malignant'
# dataset_name = "Mammo Dataset"


# # mushroom dataset
# PATHNAME = '/Users/joshdavis/Desktop/STAT RESEARCH/Analytical-Solutions-Logistic-Regression/datsets/mushroom_data.csv'
# response_col_name = 'poisonous'
# dataset_name = "Muschroom Dataset"


penalty = 0
# plot_mse_v_num_predictors(
#     PATHNAME=PATHNAME, response_col_nam=response_col_name, dataset_name=dataset_name, drop_these=drop_these, penalty=penalty)


# This function calculuates the SE for each element of the regression vector and compares both methods

beta_standard_error_comparison(PATHNAME=PATHNAME, response_col_nam=response_col_name,
                               dataset_name=dataset_name, drop_these=drop_these, penalty=penalty)


# prediction_accuracy_compariosn(num_trials=10, train_split_proportion=.8, PATHNAME=PATHNAME,
#                                response_col_nam=response_col_name, dataset_name=dataset_name, drop_these=drop_these, penalty=penalty)
