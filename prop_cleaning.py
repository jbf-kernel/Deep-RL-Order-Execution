# This code is meant for cleaning trades and quotes data. Included are methods to clean data individually
# and merge trades and quotes and aggregate data so as to be used in the models.


import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import glob


def intra_day(data):
    return data[(data.index.strftime("%H:%M") >= "09:30") & (data.index.strftime("%H:%M") < "16:00")]


def clean_quotes(quotes):
    """ Cleans quotes data based on the following steps:

    1. Transform data into time series.
    2. Subset data into intraday data (9:30 am - 4:00 pm)
    3. Delete entries with a bid, ask and volumes of 0
    4. Subset to quotes only from NYSE
    5.When multiple quotes have the same timestamp, we replace all these with a single entry
    with the median bid and median ask price and we sum the volumes. 
    6. Delete entries for which the spread is negative
    7. Delete entries for which the spread is 50 times the daily median. (TO DO!) """

    quotes["DATE"] = quotes["DATE"].astype(str)
    quotes["TIME"] = quotes["DATE"] + quotes["TIME_M"]
    quotes["TIME"] = quotes["TIME"].apply(lambda x: datetime.strptime(x, "%Y%m%d%H:%M:%S.%f"))
    quotes = quotes.drop(columns=["DATE", "TIME_M", "SYM_SUFFIX", "SYM_ROOT"])
    quotes = quotes.set_index("TIME")
    quotes = quotes.sort_index()

    quotes = intra_day(quotes)

    quotes = quotes[(quotes["BID"] > 0) & (quotes["BIDSIZ"] > 0) & (quotes["ASK"] > 0) & (quotes["ASKSIZ"] > 0)]
    quotes = quotes.drop(columns=["EX"])

    quotes = quotes.groupby(quotes.index).agg({
        "BID": np.median,
        "BIDSIZ": np.sum,
        "ASK": np.median,
        "ASKSIZ": np.sum,
        "QU_SEQNUM": "first",
        "NATBBO_IND": "first"})
    quotes["SPREAD"] = quotes["ASK"] - quotes["BID"]
    quotes = quotes[quotes["SPREAD"] > 0]
    return quotes


def clean_trades(trades):
    trades["DATE"] = trades["DATE"].astype(str)
    trades["TIME"] = trades["DATE"] + trades["TIME_M"]
    trades["TIME"] = trades["TIME"].apply(lambda x: datetime.strptime(x, "%Y%m%d%H:%M:%S.%f"))
    trades = trades.drop(columns=["DATE", "TIME_M", "SYM_SUFFIX"])
    trades = trades.set_index("TIME")
    trades = trades.sort_index()
    trades = intra_day(trades)

    trades = trades[(trades["SIZE"] > 0) & (trades["PRICE"] > 0)]

    trades = trades[trades["TR_CORR"] == 0]

    trades = trades.drop(columns=["TR_RF", "TR_CORR", "TR_SCOND", "TR_STOPIND", "EX", "SYM_ROOT"])

    trades = trades.groupby(trades.index).agg({"SIZE": np.sum, "PRICE": np.median,
                                               "TR_SEQNUM": "first"})

    return trades


def match_tq(qdata, tdata, adjustment=2):
    qdata = qdata.set_index(qdata.index + timedelta(seconds=adjustment))
    cols_to_use = qdata.columns.difference(tdata.columns)
    dfNew = pd.merge(tdata, qdata[cols_to_use], left_index=True, right_index=True, how='outer')

    # Forward fill the NANs on the qdata side

    dfNew[cols_to_use] = dfNew[cols_to_use].ffill()

    dfNew = dfNew[dfNew.index.isin(tdata.index)]
    # Back fill so that we get rid of Nans at the beginning.
    dfNew[cols_to_use] = dfNew[cols_to_use].bfill()

    return dfNew


def trade_classify(data, method="FTT"):
    # Classifies trades based on the forward tick test. Also computes Ordersign imbalance

    #Can also add the rule, trades above mid price = +1 and likewise. Discard trades at mid-price or
    #inside the spread. See Nonlinear price impact from linear models. 
    #Merge based on transaction sign and millisecond time-stamp. 
    if method == "FTT":
        FTT_full(data)
        data["OSIFTT"] = np.cumsum(data["SIGNFTT"]) / np.arange(1, data.shape[0] + 1)


def FTT(data):  # Implements a forward tick test to determine for one day.

    signs = np.zeros(data.shape[0])
    signs[0] = 1  # Customary to initialize trading with an uptick
    signs[1:] = np.sign(data["PRICE"][1:].values - data["PRICE"][0:data.shape[0] - 1].values)

    return signs


def FTT_full(data):  # Implements a forward tick test for the full data set

    days = data.index.strftime("%Y-%m-%d").unique()

    data["SIGNFTT"] = 0
    for day in days:
        data["SIGNFTT"][data.index.strftime("%Y-%m-%d") == day] = FTT(data[data.index.strftime("%Y-%m-%d") == day])

    data["SIGNFTT"] = data["SIGNFTT"].replace(to_replace=0, method='ffill')


def VWAP(df):
    return (df["PRICE"] * df["SIZE"]).sum() / (df["SIZE"].sum())


def aggregate(df):
    # To do: aggregate order sign imbalance in a better way?

    # Could speed this up by taking unique seconds and not days and then just looping through that.
    days = df.index.strftime("%Y-%m-%d").unique()  # Get all the days in the series

    start = days[0] + " " + df.index[df.index.strftime("%Y-%m-%d") == days[0]][0].strftime("%H:%M:%S")

    end = days[0] + " " + "16:00:00"

    dates = pd.date_range(start=start, end=end, freq="S")[1:]  # Equivalent to adding a second.

    for days in days[1:]:
        start = days + " " + df.index[df.index.strftime("%Y-%m-%d") == days][0].strftime("%H:%M:%S")

        end = days + " " + "16:00:00"

        intra_ = pd.date_range(start=start, end=end, freq="S")[1:]

        dates = dates.append(intra_)

    bid = np.zeros(len(dates))
    bidsiz = np.zeros(len(dates))
    ask = np.zeros(len(dates))
    asksiz = np.zeros(len(dates))
    prices = np.zeros(len(dates))
    volume = np.zeros(len(dates))
    osiftt = np.zeros(len(dates))

    bid[0] = df["BID"][df.index < dates[0]][-1]
    bidsiz[0] = df["BIDSIZ"][df.index < dates[0]][-1]
    ask[0] = df["ASK"][df.index < dates[0]][-1]
    asksiz[0] = df["ASKSIZ"][df.index < dates[0]][-1]
    prices[0] = VWAP(df[df.index < dates[0]])
    volume[0] = df["SIZE"][df.index < dates[0]].sum()
    osiftt[0] = df["OSIFTT"][df.index < dates[0]][-1]  # Take the last order sign imbalance value

    for i in range(1, len(dates)):

        if dates[i].strftime("%H:%M:%S") == "16:00:00":

            if df[(df.index < dates[i]) & (df.index >= dates[i - 1])].shape[0] < 1:
                # This is to take care of the case when there is no activity within these periods
                bid[i] = None
                bidsiz[i] = None
                ask[i] = None
                asksiz[i] = None
                prices[i] = None
                volume[i] = None
                osiftt[i] = None

            else:
                bid[i] = df["BID"][(df.index <= dates[i]) & (df.index >= dates[i - 1])][-1]
                bidsiz[i] = df["BIDSIZ"][(df.index <= dates[i]) & (df.index >= dates[i - 1])][-1]
                ask[i] = df["ASK"][(df.index <= dates[i]) & (df.index >= dates[i - 1])][-1]
                asksiz[i] = df["ASKSIZ"][(df.index <= dates[i]) & (df.index >= dates[i - 1])][-1]
                prices[i] = VWAP(df[(df.index <= dates[i]) & (df.index >= dates[i - 1])])
                volume[i] = df["SIZE"][(df.index <= dates[i]) & (df.index >= dates[i - 1])].sum()
                osiftt[i] = df["OSIFTT"][(df.index <= dates[i]) & (df.index >= dates[i - 1])][-1]


        else:
            if df[(df.index < dates[i]) & (df.index >= dates[i - 1])].shape[0] < 1:
                bid[i] = None
                bidsiz[i] = None
                ask[i] = None
                asksiz[i] = None
                prices[i] = None
                volume[i] = None
                osiftt[i] = None

            else:
                bid[i] = df["BID"][(df.index < dates[i]) & (df.index >= dates[i - 1])][-1]
                bidsiz[i] = df["BIDSIZ"][(df.index < dates[i]) & (df.index >= dates[i - 1])][-1]
                ask[i] = df["ASK"][(df.index < dates[i]) & (df.index >= dates[i - 1])][-1]
                asksiz[i] = df["ASKSIZ"][(df.index < dates[i]) & (df.index >= dates[i - 1])][-1]
                prices[i] = VWAP(df[(df.index < dates[i]) & (df.index >= dates[i - 1])])
                volume[i] = df["SIZE"][(df.index < dates[i]) & (df.index >= dates[i - 1])].sum()
                osiftt[i] = df["OSIFTT"][(df.index < dates[i]) & (df.index >= dates[i - 1])][-1]

    data = pd.DataFrame(np.column_stack((prices, volume, osiftt, bid, bidsiz, ask, asksiz)), index=dates,
                        columns=["PRICE", "VOLUME", "OSIFTT", "BID", "BIDSIZ",
                                 "ASK", "ASKSIZ"])
    data = data.ffill()
    return data


def create_episodes(df, minutes):
    # Takes how many minutes an episode is and aggregates the data into episodes of number of minutes long.

    # To do: Need to be careful because it is possible that we do not choose a length of time that leaves out the last period. I.e., if you
    # choose minutes = 60 then we do not have 16:00 as a date time. For now best to choose minutes as a divisor of 30.
    assert 30 % minutes == 0, "Minutes must be a divisor of 30"
    days = df.index.strftime("%Y-%m-%d").unique()

    dates = pd.date_range(start=days[0] + " " + "9:30:00", end=days[0] + " " + "16:00:00", freq=str(minutes) + "min")[
            1:]
    for days in days[1:]:
        time = pd.date_range(start=days + " " + "9:30:00", end=days + " " + "16:00:00", freq=str(minutes) + "min")[1:]

        dates = dates.append(time)

    episodes = []
    episodes.append(df[df.index < dates[0]].values)
    for i in range(1, len(dates) - 1):
        if dates[i].strftime("%H:%M:%S") == "16:00:00":
            episodes.append(df[(df.index >= dates[i - 1]) & (df.index <= dates[i])].values)
        elif dates[i].strftime("%H:%M:%S") == dates[0].strftime(
                "%H:%M:%S"):  # This is so we don't double count the ending last points of the previous day.
            episodes.append(df[(df.index > dates[i - 1]) & (df.index < dates[i])].values)
        else:
            episodes.append(df[(df.index >= dates[i - 1]) & (df.index < dates[i])].values)
    episodes = np.array(episodes)

    return episodes


def Paper_subset(df, minutes):
    # Subsets the data as done in the paper. We only subset for trading hours between 11-12, 12-13, and 13-14.
    days = df.index.strftime("%Y-%m-%d").unique()

    dates = pd.date_range(start=days[0] + " " + "11:00:00", end=days[0] + " " + "14:00:00", freq=str(minutes) + "min")
    for days in days[1:]:
        time = pd.date_range(start=days + " " + "11:00:00", end=days + " " + "14:00:00", freq=str(minutes) + "min")

        dates = dates.append(time)

    episodes = []
    episodes.append(df[(df.index >= dates[0]) & (df.index < dates[1])].values)
    for i in range(1, len(dates) - 1):
        if dates[i].strftime("%H:%M:%S") == "13:00:00":  # If all equalities we can very much clean this up.
            episodes.append(df[(df.index >= dates[i]) & (df.index <= dates[
                i + 1])].values)  # Can change second inequality to an equality if we want to include the last time stamp for trading.
        elif dates[i].strftime("%H:%M:%S") in ["11:00:00",
                                               "12:00:00"]:  # This is so we don't double count the ending last points of the previous day.
            episodes.append(df[(df.index >= dates[i]) & (df.index <= dates[
                i + 1])].values)  # Might want an equality in the second term so that we get the end point to fully liquidate.

    episodes = np.array(episodes)

    return episodes


def load_data(symbol):
    # Automatically loads the trades and quotes data for stock symbol.

    quotes_path = r"/Users/Joseph/Desktop/Data/" + symbol + "/Quotes"
    trades_path = r"/Users/Joseph/Desktop/Data/" + symbol + "/Trades"

    quote_files = glob.glob(quotes_path + "/*.csv")
    trade_files = glob.glob(trades_path + "/*.csv")

    quotes = pd.concat((pd.read_csv(f) for f in quote_files))

    trades = pd.concat((pd.read_csv(f) for f in trade_files))

    return quotes, trades


def gen_prop_data(trade, quote):
    """
    :param trade: trade dataset
    :param quote: quote dataset
    :return: gives dataset for price propagator model


    n refers to trades that do not change mid price and c refers to those that do. 
    """
    master = trade.merge(quote, on='TIME', how='outer')
    master.index = master.TIME
    master = master.sort_index()
    master['mid_price'] = (master.BID + master.ASK) / 2
    master.mid_price = master.mid_price.fillna(method='ffill')
    master = master.drop(['TIME', 'TR_SEQNUM', 'BID', 'BIDSIZ', 'ASK', 'ASKSIZ', 'QU_SEQNUM', 'NATBBO_IND', 'SPREAD'],
                         axis=1)
    master['Sign'] = np.where(master.PRICE == master.mid_price, None, 1)
    master.dropna(inplace=True)

    temp = master[['SIZE', 'PRICE']]
    temp['W_price'] = temp.SIZE * temp.PRICE
    d_temp = temp.resample('s').sum()
    d_temp['VWAP'] = d_temp.W_price / d_temp.SIZE
    d_temp.dropna(inplace=True)
    d_temp.drop(['SIZE', 'PRICE', 'W_price'], axis=1, inplace=True)

    d_master = master.resample('s').mean()
    d_master.dropna(inplace=True)
    d_master.drop(['SIZE', 'PRICE'], axis=1, inplace=True)

    df = pd.concat([d_temp, d_master], axis=1)

    df = df.round(4)
    df['Sign'] = np.where(df.VWAP == df.mid_price, None, 1)
    df.dropna(inplace=True)
    df['Sign'] = np.where(df.VWAP > df.mid_price, 1, -1)
    df['lag_mid'] = df.mid_price.shift(-1)
    df['Event'] = np.where(df['mid_price'] == df['lag_mid'], 'n', 'c')
    df['Return'] = np.log(df.lag_mid) - np.log(df.mid_price)
    df.dropna(inplace=True)

    return df
