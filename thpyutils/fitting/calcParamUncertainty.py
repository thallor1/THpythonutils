import os
import lmfit
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, Model
from thpyutils.fitting.calcChisqr import calcChisqr
from thpyutils.scripting import ProgressBar

# noinspection PyArgumentList
def calcParamUncertainty(obs_vals, obs_errs, model, result, independent_vars=False, fast_calc=False,
                         extrapolate=True, show_plots=True, fname='uncertainties.txt', overwrite_prev=False,
                         num_test_points=30, debug=False, fit_method='powell'):
    """
    This is a function to calculate the uncertainties in the free parameters of any lmfit model.
    It assumes a parabolic form, and takes the following arguments:

    :param obs_vals: 1D array of observed values
    :param obs_errs: 1D array of error bars
    :param model: lmfit model object used to fit
    :param result: lmfit results object from initial fit
    :param independent_vars: independent variables used in fitting, a dictionary.
    :param fast_calc: Simply assumes a 30% error bar.
    :param extrapolate: Performs a parabolic fit to chisqr curve rather than waiting for a threshold
    :param show_plots: Flag to display plots of chisqr curves at every point.
    :param fname: File to save results to and load from.
    :param overwrite_prev: Flag to load or overwrite previous uncertainty values.
    :param num_test_points: Number of points required for parabolic extrapolation
    :param debug: Debugging flag.
    :param fit_method: Method to perform fitting at each step, usually leave as "powell".
    :return err_out: A dictionary of error bars for each parameter.
    """

    # First step is to check if the parameters already exist.
    if os.path.isfile(fname) and overwrite_prev is True:
        os.remove(fname)
    if os.path.isfile(fname) and overwrite_prev is not True:
        errors = np.load(fname, allow_pickle=True).item()
        return errors
    if fast_calc is True:
        errs = {}
        for param in result.params:
            errs[param] = result.params[param].value * 0.2
        return errs
        # Get an initial value of chisqr
    obs_errs[obs_errs == 0] = np.nan
    obs_errs[np.isnan(obs_errs)] = np.inf
    obs_vals[obs_vals == 0] = 0
    obs_vals[np.isnan(obs_vals)] = 0
    chisqr0 = calcChisqr(obs_vals, result.best_fit, obs_errs)
    # Get the number of free parameters in the fit:
    num_free_params = 0
    for param in result.params:
        if result.params[param].vary is True:
            num_free_params += 1
    # Get the number of points being fit
    num_points = np.nansum([obs_errs < 1e10])
    # Calculate the statistical min and max allowed values of chisqr
    chisqrmax = chisqr0 / (1.0 - 1.0 / (num_points - num_free_params))
    if debug is True:
        print('Chisqr0=' + str(chisqr0))
        print('Chisqrmax=' + str(chisqrmax))
    err_out = {}

    progress = ProgressBar(num_free_params, fmt=ProgressBar.FULL)  # I like a progress bar
    '''
    General algorithm for which points to test is the following:
    1. Set a min/max param value based on a percent error, noting that near zero a pure percent error will fail. 
    2. Test each of these points. If they are above the max value, then reduce the percent error by a factor of two. 
            If they are below the max value, then increase the percent error by a factor of two. 
            This should result in wide adaptability for ranges of fit parameters. 
    3. Once the two border points have been found (the points directly above and below the max chisqr)
        see if the number of evaluated points is acceptable. If not, fill in the parameter space between the points 
        uniformly to find an acceptable number of points. 
    4. Once a suitable number of points have been evaluated, perform a parabolic fit to determine 
    the error bar if desired.
    5. If the parabolic fit is not performed, then the error bar is taken as half
     the distance between the min/max points.

    '''
    # Iterate through each parameter to do this
    weights = 1.0 / obs_errs
    prev_slope = False
    for param in result.params:
        affect_chisqr = True
        try:
            if result.params[param].vary is True:
                print('Evaluating Uncertainty for ' + str(param))
                found_min = False
                found_max = False
                min_i = 0.0
                max_i = 0.0
                # Evaluated points will be kept track of in a param val list and a chisqr list
                param_list = []
                chisqr_list = []
                init_param_val = result.params[param].value
                if debug is True:
                    print('Init param val')
                    print(init_param_val)
                # param_list.append(init_param_val)
                # chisqr_list.append(chisqr0)
                opt_val = init_param_val
                new_min_chisqr_prev = 0.0
                while found_min is False and min_i < 1e2:
                    new_params_min = result.params.copy()
                    min_param_val = init_param_val - np.abs(
                        init_param_val * (2.0 ** min_i) * 0.005)  # Start with a small 0.5% error
                    new_params_min.add(param, vary=False, value=min_param_val)
                    if type(independent_vars) is bool:
                        new_result_min = model.fit(obs_vals, params=new_params_min, nan_policy='omit',
                                                   method=fit_method,
                                                   weights=weights)
                    elif type(independent_vars) is dict:
                        new_result_min = model.fit(obs_vals, params=new_params_min, nan_policy='omit',
                                                   method=fit_method,
                                                   weights=weights, **independent_vars)
                    # Get the chisqr after fitting the new param
                    new_min_chisqr = calcChisqr(obs_vals, new_result_min.best_fit, obs_errs)
                    if new_min_chisqr > chisqrmax:
                        found_min = True  # We are free
                    if new_min_chisqr < new_min_chisqr_prev:
                        # Jumped out of a local minima, need to break. Do not append these points to the array
                        found_min = True
                    else:
                        param_list.append(min_param_val)
                        chisqr_list.append(new_min_chisqr)

                        min_i = min_i + 1.0
                        if debug is True:
                            print(min_i)
                            print('Curr chisqr: ' + str(new_min_chisqr))
                            print('Max chisqr: ' + str(chisqrmax))
                        if new_min_chisqr == new_min_chisqr_prev and min_i > 4:
                            print('Param ' + str(param) + ' does not affect chisqr.')
                            found_min = True
                            affect_chisqr = False
                        new_min_chisqr_prev = new_min_chisqr
                new_max_chisqr_prev = 0.0
                while found_max is False and max_i < 1e2:
                    max_param_val = init_param_val + np.abs(init_param_val * (2.0 ** max_i) * 0.005)
                    new_params_max = result.params.copy()
                    new_params_max.add(param, vary=False, value=max_param_val)
                    if type(independent_vars) == bool:
                        new_result_max = model.fit(obs_vals, params=new_params_max, nan_policy='omit',
                                                   method=fit_method,
                                                   weights=weights)
                    else:
                        new_result_max = model.fit(obs_vals, params=new_params_max, nan_policy='omit',
                                                   method=fit_method,
                                                   weights=weights, **independent_vars)
                        # Get the chisqr after fitting the new param
                    new_max_chisqr = calcChisqr(obs_vals, new_result_max.best_fit, obs_errs)
                    if new_max_chisqr > chisqrmax:
                        found_max = True
                    if new_max_chisqr < new_max_chisqr_prev:
                        # Jumped out of local minima, function is not well-behaved.
                        found_max = True
                    else:
                        param_list.append(max_param_val)
                        chisqr_list.append(new_max_chisqr)

                        if new_max_chisqr == new_max_chisqr_prev and max_i > 4:
                            print('Param ' + str(param) + ' does not affect chisqr.')
                            affect_chisqr = False
                            found_max = True
                        max_i = max_i + 1
                        if debug is True:
                            print(max_i)
                            print('Curr chisqr: ' + str(new_max_chisqr))
                            print('Max chisqr: ' + str(chisqrmax))
                        new_max_chisqr_prev = new_max_chisqr
                if found_min is False or found_max is False:
                    print(
                        'WARNING- strange behavior in uncertainty calculation,\
                         enbable show_plots==True to assure correctness')
                # Supposedly they have both been found. Check the number of points.
                # if the estimate was initially way too large, then we need to fill them in .
                num_eval_points = np.sum([np.array(chisqr_list) < chisqrmax])
                max_point = np.max(param_list)
                min_point = np.min(param_list)
                if debug is True:
                    print('Max Point: ')
                    print(max_point)
                    print('Min point: ')
                    print(min_point)
                    print('Num eval points:')
                    print(num_eval_points)
                while num_eval_points < num_test_points:
                    # Evaluate the remainder of points in an evenly spaced fashion
                    num_new_points = int(1.5 * (num_test_points - num_eval_points))
                    # fill_points = np.linspace(min_point,max_point,num_new_points)
                    fill_points = np.random.uniform(low=min_point, high=max_point, size=num_new_points)
                    for param_val in fill_points:
                        new_params = result.params.copy()
                        new_params.add(param, vary=False, value=param_val)
                        if type(independent_vars) == bool:
                            new_result = model.fit(obs_vals, params=new_params, nan_policy='omit', method=fit_method,
                                                   weights=weights)
                        else:
                            new_result = model.fit(obs_vals, params=new_params, nan_policy='omit', method=fit_method,
                                                   weights=weights, **independent_vars)
                            # Get the chisqr after fitting the new param
                        new_chisqr = calcChisqr(obs_vals, new_result.best_fit, obs_errs)
                        param_list.append(param_val)
                        chisqr_list.append(new_chisqr)
                    # number of ponts below chisqrmax is the num_eval points
                    num_eval_points = np.sum([np.array(chisqr_list) < chisqrmax])
                    # num_eval_points == 0 or 1 is a special case.
                    good_param_vals_i = [np.array(chisqr_list) < chisqrmax]
                    good_param_vals = np.array(param_list)[good_param_vals_i]
                    if num_eval_points > 1:
                        max_point = np.max(good_param_vals)
                        min_point = np.min(good_param_vals)
                    else:
                        minus_points_i = tuple([param_list < opt_val])
                        minus_points = np.array(param_list)[minus_points_i]
                        min_point = np.max(minus_points)
                        plus_points = np.array(param_list)[param_list > opt_val]
                        max_point = np.max(plus_points)
                        if debug is True:
                            print('new min point')
                            print(min_point)
                            print('new max point')
                            print(max_point)
                    if num_eval_points < num_test_points:
                        print('Insufficient number of points under max chisqr. Recursively iterating.')
                        print('Good points: ' + str(num_eval_points) + '/' + str(num_test_points))
                # In theory should have all points needed to get uncertainties now.
                chisqr_list = np.array(chisqr_list)
                param_list = np.array(param_list)

                # If there are new points that have a lower chisqr than the initial fit,
                #   adjust the maxchisqr now or the error will be artificially large.
                opt_val = param_list[np.argmin(chisqr_list)]
                opt_chisqr = np.nanmin(chisqr_list)
                temp_chisqrmax = opt_chisqr / (1.0 - 1.0 / (num_points - num_free_params))
                if extrapolate is True:
                    def parabola(x, a, b, c):
                        return a * ((x - b) ** 2) + c

                    para_model = Model(parabola)
                    para_params = para_model.make_params()
                    # very roughly assume it's linear between 0 and 1 points
                    if prev_slope is False:
                        guess_slope = np.abs(temp_chisqrmax - chisqr0) / (
                            ((np.nanmax(np.array(param_list)) - opt_val) ** 2))
                    else:
                        guess_slope = prev_slope
                    para_params.add('a', value=guess_slope, min=0, max=1e8)
                    para_params.add('b', vary=True, value=opt_val)  # Should just be hte optimum value
                    para_params.add('c', vary=True, value=chisqr0, min=chisqr0 - 0.1, max=chisqrmax)
                    # weight by distance from optimum value?
                    para_weights = np.exp(-1.0 * (np.abs(param_list - opt_val)) / np.abs(opt_val))
                    para_fit = para_model.fit(chisqr_list, x=param_list, params=para_params, method=fit_method,
                                              weights=para_weights)
                    a_fit = para_fit.params['a'].value
                    b_fit = para_fit.params['b'].value
                    c_fit = para_fit.params['c'].value
                    prev_slope = a_fit
                    max_param_val = np.sqrt(np.abs((temp_chisqrmax - c_fit) / a_fit)) + b_fit
                    # print('Max param_val='+str(max_param_val))
                    error = np.abs(max_param_val - b_fit)
                    opt_val = b_fit
                    eval_range = np.linspace(opt_val - error * 1.2, opt_val + error * 1.2, 3000)
                    fit_eval = para_model.eval(x=eval_range, params=para_fit.params)
                    if affect_chisqr is True:
                        err_out[param] = error
                    else:
                        err_out[param] = np.nanmean(param_list) / np.sqrt(len(param_list))
                else:
                    good_param_val_i = tuple([np.array(chisqr_list) < temp_chisqrmax])
                    good_param_vals = param_list[good_param_val_i]
                    good_chisqrs = chisqr_list[good_param_val_i]
                    max_i = np.argmax(good_chisqrs)
                    init_i = np.argmin(np.abs(np.max(good_chisqrs) - chisqr0))
                    error = np.abs(good_param_vals[max_i] - good_param_vals[init_i])
                    if affect_chisqr is True:
                        err_out[param] = error
                    else:
                        err_out[param] = np.nanmean(param_list) / np.sqrt(len(param_list))
                if show_plots is True:
                    try:
                        plt.figure()
                        plt.plot(param_list, chisqr_list, color='k', marker='o', ls='')
                        plt.xlabel('Param val')
                        plt.ylabel('Chisqr')
                        plt.title('Uncertainty ' + str(param) + ' Error =' + str(round(error, 3)) + ' or ' + str(
                            round(100 * error / opt_val, 3)) + '%')
                        # plt.xlim(np.min(test_value_arr)-np.abs(np.min(test_value_arr))/10.0,np.max(test_value_arr)*1.1)
                        plt.plot(np.linspace(np.min(param_list), np.max(param_list) + 1e-9, 10),
                                 np.ones(10) * np.abs(temp_chisqrmax), color='r', marker=' ', ls='-')
                        if extrapolate is True:
                            plt.plot(eval_range, fit_eval, 'b--')
                        plt.plot(opt_val + error, temp_chisqrmax, 'g^')
                        plt.plot(opt_val - error, temp_chisqrmax, 'g^')
                        plt.ylim(np.min(chisqr_list) - np.abs(np.abs(temp_chisqrmax) - np.min(chisqr_list)) / 3.0,
                                 np.abs(temp_chisqrmax) + np.abs(np.abs(temp_chisqrmax) - np.min(chisqr_list)) / 3.0)
                        # plt.xlim(0.9*np.min(test_value_arr),np.max(test_value_arr)*1.1)
                        plt.show()
                    except Exception as e:
                        print('Some error while plotting.')
                        print(e)
                    # plt.ylim(0.0,1.3*np.abs(chisqrmax))
                progress.current += 1
                progress()
            else:
                err_out[param] = 0.0
        except Exception as e:
            err_out[param] = 0
            print('Warning: Error when evaluating uncertainty. ')
            print(e)
    # Save to a file
    np.save(fname, err_out)
    return err_out
