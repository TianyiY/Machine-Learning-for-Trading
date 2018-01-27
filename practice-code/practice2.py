import os
import pandas as pd
import matplotlib.pyplot as plt


def symbol_to_path(symbol, base_dir="data"):
    # Return CSV file path given ticker symbol.
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    # Read stock data (adjusted close) for given symbols from CSV files.
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp=pd.read_csv(symbol_to_path(symbol), index_col='Date', parse_dates=True,
                            usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp=df_temp.rename(columns={'Adj Close': symbol})
        df=df.join(df_temp)
        if symbol=='SPY':
            # Drop dates which did not trade
            df=df.dropna(subset=['SPY'])

    return df


def normalize_data(df):
    # Normalize stock prices using the first row of the dataframe
    return df/df.ix[0, :]


def plot_data(df, title='Stock prices'):
    # Plot stock prices
    ax=df.plot(title=title, fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.show()


def run():
    # Define a date range
    dates = pd.date_range('2012-09-06', '2012-09-10')

    # Choose stock symbols to read
    symbols = ['SPX']

    # Get stock data
    df = get_data(symbols, dates)
    print 'demo1:', df[1:]
    print 'demo2:', df[:-1].values
    print 'demo3:', df.ix[0, :]
    #df=normalize_data(df)

    plot_data(df)

    print df
    print df.ix['2012-09-06':'2012-09-07']
    print df['SPY']
    print df.ix['2012-09-06':'2012-09-07', ['SPY']]


if __name__ == "__main__":
    run()
