import numpy as np


def calcChisqr(obs_arr, theory_arr, obs_err_arr):
    # For arbitrary arrays of theory, experiment, errors, returns the chisqr statistic.
    # Returns it for each point I guess.
    N = np.nansum([obs_err_arr < 1e9])
    obs_arr = np.array(obs_arr)
    theory_arr = np.array(theory_arr)
    obs_err_arr = np.array(obs_err_arr)
    diffsqr = (obs_arr - theory_arr) ** 2
    chisqr_arr = diffsqr / obs_err_arr ** 2
    chisqr = np.nansum(chisqr_arr) / N
    return chisqr