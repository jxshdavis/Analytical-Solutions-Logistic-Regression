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

# import RealDataExperiment

#
# min_x = np.log(2)/np.log(10)
# max_x = np.log(8)/np.log(10)


import matplotlib.pyplot as plt
import numpy as np

# Given list of proportions


min_x = 1
max_x = 2

# leaving blank will use sklearn default value of 100
max_iter = 100


for num_reg in [4, 8]:
    sim = ExperimentSimulations.ExperimentSimulations(
        num_trials=100,
        num_regressors=num_reg,
        num_levels=2,
        num_obs=10**2,
        log_min_x=min_x,
        log_max_x=max_x,
        number_x_ticks=4,
        beta_range=[-2, 2],
        penalty=None,
        use_mean=False,
        mixing_percentage=0,
        large_samples_size=6,
        drop_under_count=[0],
        max_iter=max_iter,
    )
    sim.run_observation_sim()

    # sim.run_regressors_sim()
