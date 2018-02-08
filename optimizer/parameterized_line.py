import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def cal_error(line_parameters, data):
    # line_paramaters: slope and y-intercept
    # data: 2d array where each row is a point (x, y)
    error=np.sum((data[:, 1]-(line_parameters[0]*data[:, 0]+line_parameters[1]))**2)
    return error

def fit_line(date, error_fun):
    # find line that minimizes the error function
    # generate initial guess for line model
    l=np.float32([0, np.mean(date[:, 1])])   # slope=0, intercept=np.mean(date[:, 1])

    # plot initial guess
    x_end=np.float32([-5, 5])
    plt.plot(x_end, l[0]*x_end+l[1], 'm--', linewidth=2., label='Initial Guess')

    # call optimizer to minimize error function
    result=spo.minimize(error_fun, l, args=(date,), method='SLSQP', options={'disp':True})
    return result.x

def run():
    # define original line parameters
    line_para_origin=np.float32([4, 2])
    X_origin=np.linspace(0, 10, 21)
    Y_origin=line_para_origin[0]*X_origin+line_para_origin[1]
    plt.plot(X_origin, Y_origin, 'b--', linewidth=2., label='Original line')

    # generate noisy data points
    noise_sigma=3.
    noise=np.random.normal(0, noise_sigma, Y_origin.shape)
    data=np.asarray([X_origin, Y_origin+noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label='Data Points')

    # fit a line to this data
    line_fit=fit_line(data, cal_error)
    print 'Fitted line slope: ', line_fit[0], 'Fitted line intercept: ', line_fit[1]
    plt.plot(data[:, 0], line_fit[0]*data[:, 0]+line_fit[1], 'r--', linewidth=2., label='Fitted line')
    plt.legend(loc='upper left')
    plt.show()

if __name__=='__main__':
    run()