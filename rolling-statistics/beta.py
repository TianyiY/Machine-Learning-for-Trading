import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def symbol_to_path(symbol, base_dir="data"):
    # Return CSV file path given ticker symbol.
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    # Read stock data (adjusted close) for given symbols from CSV files.
    df = pd.DataFrame(index=dates)
    if 'SP500' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SP500')

    for symbol in symbols:
        df_temp=pd.read_csv(symbol_to_path(symbol), index_col='Date', parse_dates=True,
                            usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp=df_temp.rename(columns={'Adj Close': symbol})
        df=df.join(df_temp)
        if symbol=='SP500':
            # Drop dates which did not trade
            df=df.dropna(subset=['SP500'])

    return df


def plot_data(df, title='Stock Price', xlabel='Date', ylabel='Price'):
    # Plot stock prices
    ax=df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def compute_daily_return(df):
    # Compute and return the daily return values
    daily_return=df.copy()
    # the second row - the first row, then assign the result to the second row
    # expand this process to the whole df
    daily_return[1:]=(df[1:]/df[:-1].values)-1
    # OR:
    #daily_return=(df/df.shift(1))-1
    # set the first row of daily return to 0
    daily_return.ix[0, :]=0
    return daily_return


def run():
    # read data
    dates=pd.date_range('2017-01-16', '2018-01-16')
    symbols=['XOM', 'CEO', 'CVX', 'RDS-A']
    df=get_data(symbols, dates)

    # compute daily return
    daily_return=compute_daily_return(df)
    print daily_return

    # scatter plot SP500 vs XOM
    daily_return.plot(kind='scatter', x='SP500', y='XOM')
    # compute beta
    beta_XOM, alpha_XOM=np.polyfit(daily_return['SP500'], daily_return['XOM'], 1)
    print beta_XOM, alpha_XOM
    plt.plot(daily_return['SP500'], beta_XOM*daily_return['SP500']+alpha_XOM, '-', color='r')
    plt.show()

    # compute correlation coefficient
    print daily_return.corr(method='pearson')


if __name__ == "__main__":
    run()