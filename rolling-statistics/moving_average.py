import os
import pandas as pd
import matplotlib.pyplot as plt


def symbol_to_path(symbol, base_dir="data"):
    # Return CSV file path given ticker symbol.
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    # Read stock data (adjusted close) for given symbols from CSV files.
    df = pd.DataFrame(index=dates)
    if 'XOM' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'XOM')

    for symbol in symbols:
        df_temp=pd.read_csv(symbol_to_path(symbol), index_col='Date', parse_dates=True,
                            usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp=df_temp.rename(columns={'Adj Close': symbol})
        df=df.join(df_temp)
        if symbol=='XOM':
            # Drop dates which did not trade
            df=df.dropna(subset=['XOM'])

    return df


def normalize_data(df):
    # Normalize stock prices using the first row of the dataframe
    return df/df.ix[0, :]


def get_rolling_mean(values, window):
    # return rolling mean of given values, using sliding window
    return pd.rolling_mean(values, window=window)


def get_rolling_std(values, window):
    # return rolling std of given values, using sliding window
    return pd.rolling_std(values, window=window)


def get_bollinger_bands(rolling_mean, rolling_std):
    # return upper bound and lower bound of Bollinger bands
    upper_bound=rolling_mean+2.*rolling_std
    lower_bound=rolling_mean-2.*rolling_std
    return upper_bound, lower_bound


def plot_data(df, title='Stock Price', xlabel='Date', ylabel='Price'):
    # Plot stock prices
    ax=df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def bollinger_run():
    # Define a date range
    dates = pd.date_range('2017-01-16', '2018-01-16')

    # Choose stock symbols to read
    symbols = ['CVX', 'CEO', 'RDS-A']

    # Get stock data
    df = get_data(symbols, dates)
    #df=normalize_data(df)
    #plot_data(df)

    rolling_mean_XOM=get_rolling_mean(df['XOM'], window=10)
    rolling_std_XOM=get_rolling_std(df['XOM'], window=10)
    upper_bound_XOM, lower_bound_XOM=get_bollinger_bands(rolling_mean_XOM, rolling_std_XOM)

    ax=df['XOM'].plot(title='Bollinger Bands', label='XOM')
    rolling_mean_XOM.plot(label='Rolling Mean', ax=ax)
    upper_bound_XOM.plot(label='Upper Band', ax=ax)
    lower_bound_XOM.plot(label='Lower Band', ax=ax)

    # compute rolling mean using 10 days moving window
    #RM_XOM=pd.rolling_mean(df['XOM'], window=10)

    # add rolling mean to the same plot
    #RM_XOM.plot(label='Rolling mean', ax=ax)

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
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


def daily_return_run():
    # Define a date range
    dates = pd.date_range('2017-01-16', '2018-01-16')

    # Choose stock symbols to read
    symbols = ['CVX', 'CEO', 'RDS-A']

    # Get stock data
    df = get_data(symbols, dates)
    # df=normalize_data(df)
    plot_data(df)

    # compute daily returns
    daily_return=compute_daily_return(df)
    plot_data(daily_return, title='Daily Return', ylabel='Daily Return')



if __name__ == "__main__":
    bollinger_run()
    daily_return_run()
