import pandas as pd

start_date='2012-09-06'
end_date='2012-09-10'
dates=pd.date_range(start_date, end_date)
print dates
print dates[0]
df1=pd.DataFrame(index=dates)
print df1

dfSPY=pd.read_csv('data/SPY.csv', index_col='Date', parse_dates=True,
                  usecols=['Date', 'Adj Close'], na_values=['nan'])
print dfSPY

df2=df1.join(dfSPY)
print df2

print df2.dropna()

df3=df1.join(dfSPY, how='inner')
print df3