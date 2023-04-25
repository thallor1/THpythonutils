import numpy as np
from thpyutils.scripting.progressbar import ProgressBar
from thpyutils.neutron.magAnalysis import getMagFF
import thpyutils.neutron.methods.mdutils as mdu
from thpyutils.fitting.calcParamUncertainty import calcParamUncertainty

import os
import lmfit
from lmfit import Model, Parameters
import matplotlib.pyplot as plt


def factorization_f(n, m, delE, **vals):
    """

    :param int n: number of points along the X / Q axis
    :param int m: number of points along the Y / Energy axis
    :param np.ndarray delE: Array of energy transfers of length m
    :param np.ndarray vals: Factorization parameters of length n+m
    :return np.ndarray calcI: Matrix of calculated intensities from factorization of shape n x m.
    """
    vals_arr = []
    for i in range(len(vals)):
        vals_arr.append(vals['param_' + str(i)])
    vals = np.array(vals_arr)
    Q_vals = vals[0:n].reshape(n, 1)  # Reconstruct y coordinates
    # Update to this function means that these are delta vals in e^(-delta_i)/Z
    # Second update: This does not behave well in unceratinty calculations. Go back to
    # linearized form.
    deltas = vals[n:]
    # Z = np.nansum(np.exp(-1.0 * deltas))
    Z = np.nansum(deltas)
    # E_vals = np.exp(-1.0 * deltas) / (delE * Z)
    E_vals = deltas / (delE * Z)
    E_vals = E_vals.reshape(1, m)  # '' x coord
    slice2D = Q_vals * E_vals
    calcI = slice2D.flatten()

    return calcI


def factorization(q_values, energies, intensities, errors, mag_ion=None, qe_limits=False,
                  method='powell', fname='uncertainties.txt',
                  fast_mode=False, overwrite_prev=False, g_factor=2.0,
                  fix_Qcut=False, fix_Ecut=False):
    """

    :param q_values: Input x-axis (or momentum transfer) values.
    :param energies: Input y-axis (or energy transfer) values.
    :param intensities: Input matrix of intensities
    :param errors: Input matrix of error bars
    :param mag_ion: Magnetic ion of sample, default is None.
    :param qe_limits: Limits of the factorization region in format of [Qmin, Qmax, Emin, Emax]
    :param method: Least-squares fitting method, default is Powell.
    :param fname: Filename to save result to.
    :param fast_mode: Flag to just assume a 30 percent error bar.
    :param overwrite_prev: Flag to overwrite previously calculated errorbars
    :param g_factor: G-factor of magnetic ion, default 2.
    :param fix_Qcut: Flag to fix the factorization in Q to be the same as an averaged cut.
    :param fix_Ecut: Flag to fix the factorization in E to be the same as an averaged cut.
    :return result: Array of [q, sq, sqerr, e, ge, geerr].
    """
    if overwrite_prev is True and os.path.exists(fname):
        # Delete the file
        os.remove(fname)
    if qe_limits is not False:
        qmin, qmax, emin, emax = qe_limits
    else:
        qmin, qmax, emin, emax = np.min(q_values), np.max(q_values), np.min(energies), np.max(energies)

    intensities = intensities[np.intersect1d(np.where(q_values >= qmin), np.where(q_values <= qmax))]
    intensities = intensities[:, np.intersect1d(np.where(energies >= emin), np.where(energies <= emax))]
    errors = errors[np.intersect1d(np.where(q_values >= qmin), np.where(q_values <= qmax))]
    errors = errors[:, np.intersect1d(np.where(energies >= emin), np.where(energies <= emax))]
    energies = energies[np.intersect1d(np.where(energies >= emin), np.where(energies <= emax))]
    q_values = q_values[np.intersect1d(np.where(q_values >= qmin), np.where(q_values <= qmax))]

    x = q_values
    y = energies
    z = intensities
    intensities[np.isnan(z)] = 0
    errors[np.isnan(z)] = 1e10
    errors[errors == 0] = 1e10
    errors[np.isnan(errors)] = 1e10

    # Take big cuts of the dataset to get a guess of g(E) and s(q)
    q_cut_guess = np.zeros(len(x))
    q_cut_guess_errs = np.zeros(len(x))
    e_cut_guess = np.zeros(len(y))
    e_cut_guess_errs = np.zeros(len(y))
    for i in range(len(q_cut_guess)):
        q_i = intensities[i, :]
        qerr_i = errors[i, :]
        qpt = np.average(q_i, weights=1.0 / qerr_i)
        q_cut_guess[i] = qpt
        q_cut_guess_errs[i] = qpt * np.mean(q_i / qerr_i)
    for i in range(len(e_cut_guess)):
        e_i = intensities[:, i]
        eerr_i = errors[:, i]
        ept = np.average(e_i, weights=1.0 / eerr_i)
        e_cut_guess[i] = ept
        e_cut_guess_errs[i] = ept * np.mean(e_i / eerr_i)

    # Normalize in energy
    e_cut_guess[np.where(e_cut_guess <= 0)[0]] = 0.0
    e_cut_integral = np.trapz(x=y, y=e_cut_guess)
    e_cut_guess /= e_cut_integral
    q_cut_guess *= e_cut_integral
    # Convert the actual cut into deltas used in the expoential definition of G(omega)
    # We do this by defining the delta at the first value in the cut to be zero.
    g_omega_0 = e_cut_guess[0]
    # delta_0 = 0.0
    #Z = 1.0 / g_omega_0  # This is used to solve for delta now.
    # delta_arr = np.zeros(len(e_cut_guess))
    # delta_arr = np.log(1.0 / e_cut_guess * Z) #Revert to linearized form
    # calc_ecut_guess = np.exp(-1.0 * delta_arr) / Z
    delta_arr = e_cut_guess
    m = len(y)  # number of E-values
    n = len(x)  # number of q_values

    arr_guess = np.append(q_cut_guess, delta_arr)
    arr_guess[np.isnan(arr_guess)] = 0
    params = Parameters()
    for i in range(len(arr_guess)):
        val = arr_guess[i]
        if i >= n:
            # Spectral weight can't be negative physically
            # Need to fix the first energy value
            if i == n:
                vary_val = True
                param_guess = arr_guess[i]
            else:
                if fix_Ecut:
                    vary_val = False
                else:
                    vary_val = True
                param_guess = arr_guess[i]
            # From of G(w) is e^(-delta), e^-8 is 3e-4
            params.add('param_' + str(i), vary=vary_val, value=param_guess, min=0, max=1.0)
        else:
            if fix_Qcut:
                vary_val = False
            else:
                vary_val = True
            params.add('param_' + str(i), value=val, vary=vary_val)

    weights = 1.0 / (np.abs(errors))

    # Make note of q, e indices at which there exist no intensities. They will be masked.
    bad_q_i = []
    bad_e_i = []
    for i in range(np.shape(intensities)[0]):
        # Check Q-cuts
        q_cut = intensities[i]
        num_nan = np.sum(np.isnan(q_cut))
        num_zero = np.sum([q_cut == 0])
        num_bad = num_nan + num_zero
        if num_bad == len(q_cut):
            bad_q_i.append(i)
        else:
            # Do nothing
            pass
    # Some high energies will also have no counts
    for i in range(np.shape(intensities)[1]):
        e_cut = intensities[:, i]
        num_nan = np.sum(np.isnan(e_cut))
        num_zero = np.sum([e_cut == 0])
        num_bad = num_nan + num_zero
        if num_bad == len(e_cut):
            bad_e_i.append(i)
        else:
            # Do nothing
            pass
    weights = np.ravel(weights)
    meas_errs = 1.0 / weights

    z_fit = np.copy(intensities)
    z_fit = z_fit.flatten()
    meas_errs[np.isnan(z_fit)] = np.inf
    z_fit[np.isnan(z_fit)] = np.nan

    vals = []
    for i in range(len(params)):
        vals.append(params['param_' + str(i)].value)
    weights = 1.0 / meas_errs
    data = z_fit
    data[np.isnan(data)] = 0
    weights[data == 0] = 0
    meas_errs = 1.0 / weights
    meas_errs[weights == 0] = np.nan
    model = Model(factorization_f, independent_vars=['n', 'm', 'delE'])
    eRes = np.abs(energies[1] - energies[0])
    # minimize this chisqr function.
    result = model.fit(data, n=n, m=m, delE=eRes, params=params, method=method, weights=weights, nan_policy='omit')
    f_array = []
    for i in range(len(result.params)):
        f_array.append(result.params['param_' + str(i)].value)

    # Normalize s.t. energy spectra integrates to one
    x_q = np.array(f_array[0:n])
    x_q[bad_q_i] = 0
    deltas = np.array(f_array[n:])

    # Z = np.nansum(np.exp(-1.0 * deltas))
    # g_e = np.exp(-1.0 * deltas) / (eRes * Z)
    # Revert to older definition of g_e
    Z = np.nansum(deltas)
    g_e = deltas / (eRes * Z)

    err_dict = {}
    if os.path.isfile(fname) or fast_mode is True:
        if fast_mode:
            err_array = 0.3 * np.array(f_array)
        else:
            err_dict = np.load(fname, allow_pickle=True).item()
            err_array = []
            for i in range(len(err_dict.keys())):
                key = 'param_' + str(i)
                err_array.append(err_dict[key])
        # Normalize s.t. energy spectra integrates to one
        q = q_values
        e = energies
        x_q = np.array(f_array[0:n])
        x_q[bad_q_i] = 0
        deltas = np.array(f_array[n:])
        # Z = np.nansum(np.exp(-1.0 * deltas))
        Z = np.nansum(deltas)
        #g_e = np.exp(-1.0 * deltas) / (eRes * Z)
        g_e = deltas / (eRes*Z)
        xq_err = err_array[0:n]
        ge_err = err_array[n:]

        x_q, g_e, xq_err, ge_err = np.array(x_q), np.array(g_e), np.array(xq_err), np.array(ge_err)

        # Now convert X(Q) into S(Q)
        r0 = 0.5391
        g = g_factor
        if mag_ion is not None:
            magFFsqr = getMagFF(q, mag_ion)
        s_q = (2.0 * x_q) / (r0 ** 2 * g ** 2 * magFFsqr)
        s_q_err = (2.0 * xq_err) / (r0 ** 2 * g ** 2 * magFFsqr)
        return q, s_q, s_q_err, e, g_e, ge_err
    # Get error bars using random sampling method
    # Create a parameter object mased on linearized form of the factorization for uncertainty
    if len(err_dict.keys()) == 0:
        # If the dictionary is populated, the error bars have been previously calculated.
        err_dict = calcParamUncertainty(data, meas_errs, model, result, fast_calc=fast_mode,
                                               independent_vars={'n': n, 'm': m, 'delE': eRes},
                                               extrapolate=True, show_plots=True, fname=fname,
                                               overwrite_prev=overwrite_prev, num_test_points=10, debug=False,
                                               fit_method=method, buffer_val=0.1)
    err_array = []
    for i in range(len(result.params)):
        f_array.append(result.params['param_' + str(i)].value)
        err_array.append(err_dict['param_' + str(i)])
    err_array = np.array(err_array)
    q = q_values
    e = energies
    x_q = np.array(f_array[0:n])
    x_q[bad_q_i] = 0
    g_e[bad_e_i] = 0
    g_e[np.isnan(g_e)] = 0
    x_q[np.isnan(x_q)] = 0
    xq_err = err_array[0:n]
    delta_err = np.array(err_array[n:])
    #ge_err = g_e * deltas * delta_err  # propogate the error
    ge_err = delta_err  # propogate the error

    x_q = np.array(x_q)
    xq_err = np.array(xq_err)
    g_e = np.array(g_e)
    ge_err = np.array(ge_err)
    # Now convert X(Q) into S(Q)
    r0 = 0.5391
    g = g_factor
    magFFsqr = getMagFF(q, mag_ion)
    s_q = (2.0 * x_q) / (r0 ** 2 * g ** 2 * magFFsqr)
    s_q_err = (2.0 * xq_err) / (r0 ** 2 * g ** 2 * magFFsqr)
    # Finally, return results

    return q, s_q, s_q_err, e, g_e, ge_err
