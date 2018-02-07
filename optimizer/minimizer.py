import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def f(X):
    Y=(X-2.5)**2+1.5
    return Y

def run():
    initial_guess=5.0
    minimum=spo.minimize(f, initial_guess, method='SLSQP', options={'disp':True})
    print 'Minimum location: ', 'X={}, Y={}'.format(minimum.x, minimum.fun)

    # plot and mark minimum location
    X_plot=np.linspace(0., 5., 21)
    Y_plot=f(X_plot)
    plt.plot(X_plot, Y_plot)
    plt.plot(minimum.x, minimum.fun, 'ro')
    plt.show()


if __name__=='__main__':
    run()

