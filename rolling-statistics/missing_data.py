import pandas as pd
import matplotlib.pyplot as plt

dfXOM=pd.read_csv('data/XOM_missing.csv', index_col='Date', parse_dates=True,
                  usecols=['Date', 'Adj Close'], na_values=['nan'])
ax1=dfXOM.plot(title='Stock Price', fontsize=12)

dfXOM.fillna(method='ffill', inplace='True')  # forward fill
dfXOM.fillna(method='bfill', inplace='True')  # backward fill
ax2=dfXOM.plot(title='Stock Price Filled', fontsize=12)

# compute daily returns
daily_return=(dfXOM/dfXOM.shift(1))-1
daily_return.ix[0, :]=0
print daily_return

# plot histogram
daily_return.hist(bins=20)

# compute mean and std
mean=daily_return['Adj Close'].mean()
print 'mean:', mean
std=daily_return['Adj Close'].std()
print 'std:', std

# plot mean line
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2)
plt.axvline(std, color='g', linestyle='dashed', linewidth=1.5)
plt.axvline(-std, color='g', linestyle='dashed', linewidth=1.5)

plt.show()
