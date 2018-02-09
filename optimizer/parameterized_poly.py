import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def cal_error(poly_parameters, data):
    # poly_paramaters: parameters of polynomial
    # data: 2d array where each row is a point (x, y)
    error=np.sum((data[:, 1]-np.polyval(poly_parameters, data[:, 0]))**2)
    return error

def fit_poly(date, error_fun, degree=3):
    # find polynomial that minimizes the error function
    # generate initial guess for poly model (all coefficients=1)
    p=np.poly1d(np.ones(degree+1, dtype=np.float32))

    # plot initial guess
    x_end=np.linspace(-5, 5, 21)
    plt.plot(x_end, np.polyval(p, x_end), 'm--', linewidth=2., label='Initial Guess')

    # call optimizer to minimize error function
    result=spo.minimize(error_fun, p, args=(date,), method='SLSQP', options={'disp':True})
    return np.poly1d(result.x)  # convert optimal result into a poly1d object and return

def run():
    # define original line parameters
    poly_para_origin=np.float32([1.5, 2.0, -0.2, 0.1])
    X_origin=np.linspace(0, 10, 21)
    Y_origin=np.polyval(poly_para_origin, X_origin)
    plt.plot(X_origin, Y_origin, 'b--', linewidth=2., label='Original poly')

    # generate noisy data points
    noise_sigma=1.
    noise=np.random.normal(0, noise_sigma, Y_origin.shape)
    data=np.asarray([X_origin, Y_origin+noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label='Data Points')

    # fit a line to this data
    poly_fit=fit_poly(data, cal_error)
    print 'Fitted poly parameters: ', poly_fit
    plt.plot(data[:, 0], np.polyval(poly_fit, data[:, 0]), 'r--', linewidth=2., label='Fitted poly')
    plt.legend(loc='upper left')
    plt.show()

if __name__=='__main__':
    run()