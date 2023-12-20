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


import matplotlib.pyplot as plt
import numpy as np

# Given list of proportions


min_x = 3
max_x = 7


for num_reg in [2, 4, 6, 8, 10, 14]:
    sim = ExperimentSimulations.ExperimentSimulations(num_trials=20, num_regressors=num_reg, num_levels=2, num_obs=10**3, log_min_x=min_x,
                                                      log_max_x=max_x, number_x_ticks=5, beta_range=[-2, 2], penalty=None, use_mean=False,
                                                      mixing_percentage=0, large_samples_size=6)
    sim.run_observation_sim()

    # sim.run_regressors_sim()
