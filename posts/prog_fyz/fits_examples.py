import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

# linear function
def lin_func(x, m, c):
    return m * x + c

# x to the power of -1 function
def neg_pow_func(x, m):
    return m / x

# taylor expansion
def taylor(x, a, b, c):
    return a / x / x + b / x + c

def plot(x_graph, y_graph, y_err, x_fit, y_fit, y_fit_lower, y_fit_upper, xlabel, filename, ylabel=r"C [$\mu$F]"):

    # draw x_graph, y_graph with error bars
    plt.errorbar(x_graph, y_graph, y_err, fmt='o', label='Data', color='black')

    # draw the fit function and its uncertainty band
    plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='red', alpha=0.3)

    # create a legend entry for the fit function and its uncertainty band
    line_with_band = mpl.lines.Line2D([], [], color='red', label='Fit', linestyle='-', linewidth=2)
    band = mpl.patches.Patch(color='red', alpha=0.3, label='Fit uncertainty')

    # get the current legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles + [(line_with_band, band)], labels=labels + ['Fit'])

    # finally, plot
    plt.plot(x_fit, y_fit, 'r-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

    return

C = np.array([275.4, 182.1, 111.0, 83.2, 68.0, 57.9, 51.2, 45.9, 41.9, 39.0, 37.9])
d = np.array([2.00, 3.00, 5.00, 7.00, 9.00, 11.0, 13.0, 15.0, 17.0, 19.0, 20.0])
d_inv = 1 / d
C_err = [10]*len(C)

# Now, the example code that fits the data!
# Use scipy.optimize.curve_fit to fit the curve.
# Note that the uncertainty of the C measurements is 10.
popt, pcov = scipy.optimize.curve_fit(neg_pow_func, d, C, sigma=C_err, absolute_sigma=True)
x_fit = np.linspace(min(d), max(d), 100)
y_fit = neg_pow_func(x_fit, popt[0])
y_fit_upper = neg_pow_func(x_fit, popt[0] + np.sqrt(pcov[0,0]))
y_fit_lower = neg_pow_func(x_fit, popt[0] - np.sqrt(pcov[0,0]))
plot(d, C, C_err, x_fit, y_fit, y_fit_lower, y_fit_upper, "d [cm]", "fits/neg_pow_fit.png")

# Fit with the taylor function. Don't consider the uncertainty of the fit parameters.
popt, pcov = scipy.optimize.curve_fit(taylor, d, C, sigma=C_err, absolute_sigma=True)
x_fit = np.linspace(min(d), max(d), 100)
y_fit = taylor(x_fit, *popt)
plot(d, C, C_err, x_fit, y_fit, y_fit, y_fit, "d [cm]", "fits/taylor.png")

# Fit with the linear function.
popt, pcov = scipy.optimize.curve_fit(lin_func, d_inv, C, sigma=C_err, absolute_sigma=True)
x_fit = np.linspace(min(d_inv), max(d_inv), 100)
y_fit = lin_func(x_fit, popt[0], popt[1])
# error propagation; take into account the uncertainty of the two fit parameters and their correlation
dyda = x_fit
dydb = 1
y_unc = np.sqrt(dyda**2 * pcov[0,0] + dydb**2 * pcov[1,1] + 2*dyda*dydb*pcov[0,1])
y_fit_upper = y_fit + y_unc
y_fit_lower = y_fit - y_unc
plot(d_inv, C, C_err, x_fit, y_fit, y_fit_lower, y_fit_upper, "1/d [1/cm]", "fits/linear.png")
