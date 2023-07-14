
import ExperimentSimulations


sim = ExperimentSimulations.ExperimentSimulations(num_trials = 10, num_regressors=2, num_levels=2, min_number_obs=1, max_number_obs=4.5, number_observations_sizes=5)
sim.plot_errors()