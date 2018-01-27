import pandas as pd
import numpy as np
from time import time

dfSPY=pd.read_csv('data/SPY.csv', index_col='Date', parse_dates=True,
                  usecols=['Date', 'Volumn', 'Adj Close'], na_values=['nan'])
print dfSPY

dfSPY_value=dfSPY.values
print dfSPY_value.shape
print dfSPY_value.size
print dfSPY_value.dtype
print dfSPY_value[0:3:2, :]

arr=np.array([(2, 3, 4), (5, 6, 7)])
print arr
print np.empty(5)
print np.empty((5, 4, 3))
print np.ones((5, 4))
print np.ones((5, 4), dtype=np.int)

# random numbers generation (uniformly sampled from [0.0, 1.0))
print np.random.random((5, 4))   # pass in a size tuple
print np.random.rand(5, 4)   # function arguments (not a tuple)

# random number (Gaussian)
print np.random.normal(0, 1, size=(2, 3))  # mean=0, std=1

# random integers
integer=np.random.randint(0, 10, size=(2, 3))   # low=0, high=10
print integer
print integer.sum()
print integer.sum(axis=0)
print integer.sum(axis=1)
print integer[0].argmax()
print integer[1].argmax()

nd=np.random.random((10000, 10000))
time1=time()
nd_mean=nd.mean()
time2=time()
t=time2-time1
print t
print nd_mean

arr2=np.array([1, 2, 3, 4, 5])
index=np.array([1, 1, 2, 3])
print arr2[index]

arr3=np.array([[1, 2, 34, 5], [34, 5, 32, 9]])
arr3_mean=arr3.mean()
print arr3
print arr3_mean
arr3[arr3<arr3_mean]=arr3_mean
print arr3