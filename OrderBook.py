import seaborn as sns
import math
from datetime import datetime
from datetime import timedelta
from helper import *

sns.set()


class LimitOrder:

    def __init__(self, lam, nu, size, **params):
        """
        :param params: Order size & order intensity
        """
        self.lam = lam
        self.nu = nu
        self.size = size
        self.params = params

    def update(self, v):
        '''
        Updates the volume via a proisson process for limit orders.
        '''
        lam = self.lam
        nu = self.nu
        size = self.size
        v += (np.random.poisson(lam) - np.random.poisson(nu * v)) * size #Orders come and are cancelled at a uniform order size.
        return np.maximum(v, 0)

    def gen_vol(self, t):
        '''
        Generates the volume up until some time t.
        '''
        lam = self.lam
        size = self.size
        v = np.random.poisson(lam) * size #Initial volume. 
        upb = min(t, 10000)
        for i in range(upb):
            v = self.update(v)
        return v

    def is_valid(self, vol, t):
        '''
        Once the volume at the best bid/ask is depleted, we call gen_vol to generate the volume at the
        next price level. Count keeps track of the levels we need to traverse to get to the next best price
        since it possible that there is no volume at the adjacent price level. 
        '''
        count = int(0)
        while vol <= 0:
            vol = self.gen_vol(t)
            count += 1
        return vol, count


class MarketOrder:

    def __init__(self, mu, rho, G, size, **params):
        """
        :param params: Order size & order intensity
        """
        self.mu = mu
        self.rho = rho
        self.G = G
        self.size = size

        self.params = params

    def update_bid_ask_price(self, m, x):
        '''
        Updates the bid and ask based on the Madhavan-Richardson-Roomans model. 
        '''
        G = self.G
        rho = self.rho
        mu_t_1 = m + G * rho * x
        ask = mu_t_1 + G * (1 - rho * x) # MRR 
        bid = mu_t_1 - G * (1 + rho * x)
        return bid, ask

    def gen_vol(self):
        '''
        Generates volume of a market order. 
        '''
        mu = self.mu
        size = self.size
        mkt = np.random.poisson(mu) * size
        return np.max(mkt, 0)


class Propagator:

    def __init__(self, K, data, **params):
        """
        :param params: propagator kernel K_nc, K_cc
        """
        self.K = K
        self.data = data
        self.params = params

    def propagate(self, m):
        """
        :param m: previous midprice
        :return: Price Propagator model for the next update midprice
        """
        K = self.K
        data = self.data
        prop_re = propagator(K, data)
        new_m = m * (1 + prop_re)
        return new_m


class OrderBook():

    def __init__(self, LimitOrder, MarketOrder, mid_price, start_time, end_time, frequency, tic, p, **params):
        """
        :param params:
        best bid, best ask, mid-price
        price min, price max
        resolution parameters: lot, tic
        time start, time end, frequency
        """
        self.LimitOrder = LimitOrder
        self.MarketOrder = MarketOrder

        self.mid_price = mid_price
        self.start_time = start_time
        self.end_time = end_time
        self.frequency = frequency

        self.tic = tic
        self.p = p

        self.params = params

    @property
    def time_range(self):
        return pd.date_range(start=self.start_time, end=self.end_time, freq=self.frequency)

    def lob_init_val(self):
        lam = self.LimitOrder.lam
        ask_vol = bid_vol = self.LimitOrder.size

        bid_pr = self.mid_price - lam * self.tic
        ask_pr = self.mid_price + lam * self.tic

        return bid_vol, bid_pr, ask_vol, ask_pr

    def update_limit(self, bid_vol, bid_pr, ask_vol, ask_pr, t):
        bid_vol = self.LimitOrder.update(bid_vol)
        ask_vol = self.LimitOrder.update(ask_vol)

        bid_vol, count_bid = self.LimitOrder.is_valid(bid_vol, t)
        ask_vol, count_ask = self.LimitOrder.is_valid(ask_vol, t)

        bid_pr -= count_bid * self.tic
        ask_pr += count_ask * self.tic

        return bid_vol, bid_pr, ask_vol, ask_pr

    def update_mkt_buy(self, ask_vol, ask_pr, t):
        '''
        Updates the LOB based on a market buy.
        '''
        mkt_buy_vol = self.MarketOrder.gen_vol()
        temp = mkt_buy_vol
        count_ask = int(0)

        if mkt_buy_vol == 0:
            x = 0
            next_ask_vol = ask_vol
            next_ask_pr = ask_pr

        else:
            x = 1
            while mkt_buy_vol > 0:
                temp = mkt_buy_vol
                mkt_buy_vol -= ask_vol
                ask_vol -= temp

                ask_vol, ask_k = self.LimitOrder.is_valid(ask_vol, t)
                count_ask += ask_k

            next_ask_vol = ask_vol
            next_ask_pr = ask_pr + count_ask * self.tic

        return next_ask_vol, next_ask_pr, temp, x

    def update_mkt_sell(self, bid_vol, bid_pr, t):
        "Updates the LOB based on a market sell."
        mkt_sell_vol = self.MarketOrder.gen_vol()
        temp = mkt_sell_vol
        count_bid = int(0)

        if mkt_sell_vol == 0:
            x = 0
            next_bid_vol = bid_vol
            next_bid_pr = bid_pr

        else:
            x = -1
            while mkt_sell_vol > 0:
                temp = mkt_sell_vol
                mkt_sell_vol -= bid_vol
                bid_vol -= temp

                bid_vol, bid_k = self.LimitOrder.is_valid(bid_vol, t)
                count_bid += bid_k

            next_bid_vol = bid_vol
            next_bid_pr = bid_pr - count_bid * self.tic

        return next_bid_vol, next_bid_pr, temp, x

    def update_lob(self, bid_vol, bid_pr, ask_vol, ask_pr, t):
        '''
        Updates the LOB with both a limit order and market buy or sell based on the probability p. 
        '''
        bid_vol, bid_pr, ask_vol, ask_pr = self.update_limit(bid_vol, bid_pr, ask_vol, ask_pr, t)
        if np.random.binomial(1, self.p) == 1:
            ask_vol, ask_pr, mkt_vol, x = self.update_mkt_buy(ask_vol, ask_pr, t)
            mkt_pr = ask_pr
        else:
            bid_vol, bid_pr, mkt_vol, x = self.update_mkt_sell(bid_vol, bid_pr, t)
            mkt_pr = bid_pr
        self.mid_price = (ask_pr + bid_pr) / 2
        return bid_vol, bid_pr, ask_vol, ask_pr, mkt_vol, mkt_pr, x

    def AR_update(self, rho):
        p_old = self.p
        eps = np.random.normal(loc=0, scale=1) / 50
        c = 0.5 * (1 - rho)
        p = c + p_old * rho + eps
        if p > 1:
            p = 1
        elif p < 0:
            p = 0
        self.p = p

    def simulate_one_day(self, rho):
        '''
        Simulates a day of the LOB. 
        '''
        # make sure has only one trading day of time
        assert (self.end_time - self.start_time).days < 1

        bid_vol, bid_pr, ask_vol, ask_pr = self.lob_init_val()
        record = {}
        T = len(self.time_range)

        for idx, t in enumerate(self.time_range):
            bid_vol, bid_pr, ask_vol, ask_pr, mkt_vol, mkt_pr, x = self.update_lob(bid_vol, bid_pr, ask_vol, ask_pr,
                                                                                idx + 1)
            record[t] = [bid_vol, bid_pr, ask_vol, ask_pr, mkt_vol, mkt_pr, x]
            bid_pr, ask_pr = self.MarketOrder.update_bid_ask_price(self.mid_price, x)
            self.AR_update(rho)

        dta = pd.DataFrame(record).T
        dta.columns = ['Bid Size', 'Bid Price', 'Ask Size', 'Ask Price', 'Vol', 'Price', 'Sign']

        return dta

    def simulate(self, rho):
        N_days = (self.end_time - self.start_time).days + 1
        total_record = []
        self.start_time = self.start_time - timedelta(days=1)
        for d in range(N_days):
            START = self.start_time + timedelta(days=1)
            END = datetime(START.year, START.month, START.day, 16, 0)
            self.start_time = START
            self.end_time = END
            d_record = self.simulate_one_day(rho)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(self.mid_price)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            total_record.append(d_record)

        return total_record
