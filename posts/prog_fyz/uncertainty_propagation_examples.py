import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib as mpl

from uncertainties import ufloat

# path length as a function of time in a free fall
g = ufloat(10, 5)
t = ufloat(3, 0.5)
s = 0.5 * g * t**2

# spring stiffness, frequency
from uncertainties import ufloat, unumpy
y = unumpy.uarray([41.2, 39.6, 36.3, 33, 26.3, 16.6], 0.1) / 100
m = unumpy.uarray([150, 200, 300, 400, 600, 900], 0) / 1000
y_m0 = ufloat(46.4, 0.5) / 100
y0 = y_m0 - y

# linear function
def lin_func(x, a):
    return a * x

# fit y0 vs m, resulting in g/k
popt, pcov = scipy.optimize.curve_fit(lin_func, 
                                      unumpy.nominal_values(m), 
                                      unumpy.nominal_values(y0), 
                                      sigma=unumpy.std_devs(y0), 
                                      absolute_sigma=True)

# omega = sqrt(k/m)
g = ufloat(9.81, 0)
g_over_k = ufloat(popt[0], np.sqrt(pcov[0,0]))
omega = unumpy.sqrt(g / g_over_k / m)


# plot the data and the fit
from fits.fits_examples import plot
xfit = np.linspace(min(unumpy.nominal_values(m)), max(unumpy.nominal_values(m)), 100)
yfit = lin_func(xfit, popt[0])
yfit_lower = lin_func(xfit, popt[0] - np.sqrt(pcov[0,0]))
yfit_upper = lin_func(xfit, popt[0] + np.sqrt(pcov[0,0]))
plot(unumpy.nominal_values(m), unumpy.nominal_values(y0), unumpy.std_devs(y0), 
    xfit, yfit, yfit_lower, yfit_upper, "m [g]", "uncertainty_propagation/spring.png", ylabel="y0 [m]")


y = unumpy.uarray([41.2, 39.6, 36.3, 33, 26.3, 16.6], 0.1)
y = unumpy.uarray([41.2, 39.6, 36.3, 33, 26.3, 16.6], [0.5, 1.0, 1.1, 1.5, 2.0, 2.5])
y /= 100
print(y)
nvs = unumpy.nominal_values(y)
sds = unumpy.std_devs(y)
