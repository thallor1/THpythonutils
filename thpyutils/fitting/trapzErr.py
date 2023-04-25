import numpy as np


def trapzErr(x, y, errs, xlim=False):
    """
    Function for the purpose of getting an error bar for a trapzoidal numpy integral.
    :param x: Independent variable
    :param y: Measured values.
    :param errs: Errors assosciated with the values.
    :param xlim: Window in which to evaluate the integral error
    :return:
    """
    if not xlim:
        xlim = [np.nanmin(x) - 0.1, np.nanmax(x) + 0.1]
    good_i = np.intersect1d(np.where(x >= xlim[0])[0], np.where(x <= xlim[1])[0])
    x = x[good_i]
    errs = errs[good_i]
    int_err = 0
    for i in range(len(errs) - 1):
        yterm = 0.5 * (y[i + 1] + y[i])
        xterm = x[i + 1] - x[i]
        yerr = np.sqrt(0.5 * (errs[i] ** 2 + errs[i + 1] ** 2))
        z = xterm * yterm
        zerr = np.sqrt(z ** 2 * (yerr / yterm))
        int_err = np.sqrt(int_err ** 2 + zerr ** 2)
    return int_err
