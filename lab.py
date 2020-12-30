import csv
from math import floor, ceil

import matplotlib.pyplot as plt
import numpy as np
import scipy.odr
from scipy.stats import chi2
from tabulate import tabulate


def line(B, x):
    """ax+b

    Parameters
    ----------
    B : list
        List with two items. First is the line's slope, second is free parameter
    x : float
        x value to calculate

    Returns
    -------
        Returns the line's value at x
    """
    return B[0]*x + B[1]


def gaussian(B, x):
    return B[0] * np.exp(-np.divide((x - B[1]) ** 2,
                                    2 * (B[2] ** 2)))


def statistical_error(errors, data):
    """
    Adds statistical error to an existing (systematic)
    errors list.
    Returns updated errors in a new list.
    """
    # ddof=1 is used to calculate the sample Standard Error rather than
    # that of the population
    statistical_err = np.std(data, ddof=1)
    return [np.sqrt(err**2 + statistical_err**2) for err in errors]


def average(data, systematic_errors):
    """
    Calculates the normal (not weighted) average of
    a sample and the error on it.
    The statistical uncertainty is added internally.
    systematic_errors should include only the systematic error on
    each measurement.
    """
    errors = statistical_error(systematic_errors, data)
    squared_errors = [err**2 for err in errors]
    avg_err = (1/len(data)) * np.sqrt(sum(squared_errors))
    return np.average(data), avg_err


def residuals(func, x_data, y_data, *args): 
    return [(yi - func(args, xi)) for xi, yi in zip(x_data, y_data)]


def parse_data(filename, skip_header=True, delimiter=","):
    """
    Parses a csv file, skips header row if specified.
    Returns list of rows (lists of items)
    """
    data = []
    with open(filename, "r", newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=delimiter)
        if skip_header:
            next(datareader)
        for row in datareader:
            data.append(row)

    return data


def calc_chi_reduced_and_p_value(chi_squared, dof):
    reduced_chi_squared = chi_squared / dof
    p_value = chi2.sf(chi_squared, dof)
    
    return reduced_chi_squared, p_value


def save_table(func, name, header, rows):
    filename = '{}_{}.csv'.format(func.__name__, name)
    filename = "".join(x for x in filename if x.isalnum() or x in "._-")
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
    except PermissionError:
        print(f"Can't save fit results for {name} to file")


def print_fit(func, params, param_errs, reduced_chi_squared, p_value, do_print=False):
    header = ['Parameters', 'Values', 'Errors']
    rows = [["a[{}]".format(i+1), params[i], param_errs[i]] for i in range(len(params))]
    table = tabulate(rows, headers=header)
    if do_print:
        print("Fit values for {}".format(func.__name__))
        print(table)
        print("Chi sqaured reduced:", reduced_chi_squared)
        print("P-value:", p_value)
        # Export to file
        save_table(func, "fit_params", header, rows)


def save_goodness_of_fit(func, reduced_chi_squared, p_value):
    save_table(func, "goodness_of_fit", ["Chi-squared reduced", "p-value"], [[reduced_chi_squared, p_value]])


def odr(func, x_data, y_data, y_errs, params_guess=None, delta_dof=0, x_errs=None):
    model = scipy.odr.Model(func)
    data = scipy.odr.RealData(x_data, y_data, sx=x_errs, sy=y_errs)
    myodr = scipy.odr.ODR(data, model, beta0=params_guess, maxit=1000)
    fit = myodr.run()
    return fit


def fit(func, x_data, y_data, y_errs, params_guess=None, delta_dof=0, x_errs=None, do_print=False):
    """
    Find best fit for func with measured y_data for x_data values, with errors on y axis.
    """
    # if y_errs and x_errs:
    fit_result = odr(func, x_data, y_data, y_errs, params_guess=params_guess, delta_dof=0, x_errs=x_errs)
    params = fit_result.beta
    param_errs = fit_result.sd_beta
    reduced_chi_squared = fit_result.res_var
    dof = len(x_data) - len(params) - delta_dof
    chi_squared = reduced_chi_squared * dof
    p_value = chi2.sf(chi_squared, len(x_data) - len(params) - delta_dof)
    if do_print:
        print_fit(func, params, param_errs, reduced_chi_squared, p_value)
        save_goodness_of_fit(func, reduced_chi_squared, p_value)
    return params, param_errs, reduced_chi_squared, p_value


def plot(func, params_list, x_data, y_data, y_errs, x_errs=None, is_discrete=False, xlabel=None, ylabel=None, title="", **plot_args):
    fig, axes = plt.subplots(nrows=2) 
    
    # Plot data with errors and the fit
    axes[0].errorbar(x_data, y_data, y_errs, xerr=x_errs,
                     label="data", **plot_args)
    if is_discrete:
        fit_range = range(floor(min(x_data)), ceil(max(x_data)), 1)
    else:
        fit_range = np.linspace(min(x_data), max(x_data), 100)
        
    if type(params_list) is not list:
        params_list = params_list.tolist()
    fitted_data = np.asarray([func(params_list, i) for i in fit_range])

    axes[0].plot(fit_range, fitted_data, "-", label="fit")
    
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].legend()
    
    # Plot residulas
    res = residuals(func, x_data, y_data, *params_list)
    axes[1].errorbar(x_data, res, fmt=".", xerr=x_errs, yerr=y_errs,
                     markersize=2)
    axes[1].plot([min(x_data), max(x_data)], [0,0], ":", color="grey")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title("Residuals")
    
    plt.subplots_adjust(hspace=0.4)



