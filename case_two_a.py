"""
This file contains the primary strategy for trading options on UChicago's proprietary xChange platform
"""
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import asyncio
import json

from time import sleep
from math import log, sqrt, pi, exp
from scipy.stats import norm
from scipy.special import ndtr
from datetime import datetime, date
import numpy as np
import pandas as pd
from pandas import DataFrame
from collections import defaultdict

from enum import Enum

PARAM_FILE = "params.json"
CONTRACTS = ["SPY"] + [f"SPY{i}{j}" for i in range(65, 140, 5) for j in ["C", "P"]]
print(CONTRACTS)
UTY = 0.05

RISK_LIMITS = [2000, 5000, 10000, 5000]


class Greeks(Enum):
    DELTA = 0
    GAMMA = 1
    THETA = 2
    VEGA = 3


def d1(S, K, T, r, sigma):
    return (log(S / K) + (r + sigma**2 / 2.0) * T) / (sigma * sqrt(T))


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * sqrt(T)


def calc_bs(S, K, T, r, sigma, is_call):
    if is_call:
        return S * norm.cdf(d1(S, K, T, r, sigma)) - K * exp(-r * T) * norm.cdf(
            d2(S, K, T, r, sigma)
        )
    else:
        return (
            norm.cdf(-d2(S, K, T, r, sigma)) * K * exp(-r * T)
            - norm.cdf(-d1(S, K, T, r, sigma)) * S
        )


def nPrime(d1v):
    return 1 / (sqrt(2 * pi)) * exp(-(d1v**2) / 2)


def calc_delta(S, K, T, r, sigma, is_call):
    if is_call:
        return norm.cdf(d1(S, K, T, r, sigma))
    else:
        return norm.cdf(-d1(S, K, T, r, sigma))


def calc_gamma(S, K, T, r, sigma):
    return 1 / (S * sigma * sqrt(T)) * nPrime(d1(S, K, T, r, sigma))


def calc_theta(S, K, T, r, sigma, is_call):
    if is_call:
        return 1 / 252 * (
            -((S * sigma) / (2 * sqrt(T)) * nPrime(d1(S, K, T, r, sigma)))
        ) - r * K * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))
    else:
        return 1 / 252 * (
            -((S * sigma) / (2 * sqrt(T)) * nPrime(d1(S, K, T, r, sigma)))
        ) - r * K * exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma))


def calc_vega(S, K, T, r, sigma):
    return (1 / 100) * S * sqrt(T) * nPrime(d1(S, K, T, r, sigma))


PARAM_FILE = "params.json"


class OptionBot(UTCBot):
    def __init__(self, *args, **kwargs):
        super(OptionBot, self).__init__(*args, **kwargs)
        self.tick = 0
        self.history = pd.DataFrame(columns=CONTRACTS)
        self.returns = pd.DataFrame(columns=CONTRACTS)
        self.volatility = pd.Series()
        self.position = {asset: 0 for asset in CONTRACTS}
        self.greeks = {asset: [] for asset in CONTRACTS}
        self.best_bid = defaultdict(int)
        self.best_ask = defaultdict(int)
        self.itm = defaultdict(bool)
        self.underlying = 100
        self.placed = False

    def update_greeks(self):
        for asset in CONTRACTS[1:]:
            lcp = self.history.iloc[-1]["SPY"]
            parse = asset.split("SPY", 1)[1]
            strike = np.float64(parse[:-1])
            is_call = parse[-1] == "C"
            time_to_maturity = ((900 - self.tick) / 10) / 252
            u_volatility = self.volatility["SPY"]
            self.greeks[asset] = [
                calc_delta(lcp, strike, time_to_maturity, UTY, u_volatility, is_call),
                calc_gamma(lcp, strike, time_to_maturity, UTY, u_volatility),
                calc_theta(lcp, strike, time_to_maturity, UTY, u_volatility, is_call),
                calc_vega(lcp, strike, time_to_maturity, UTY, u_volatility),
            ]

    async def handle_round_started(self):
        await asyncio.sleep(0.1)
        asyncio.create_task(self.handle_read_params())

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "market_snapshot_msg":
            msg = update.market_snapshot_msg
            book = msg.books
            midprices = {}

            for asset in CONTRACTS:
                if len(book[asset].bids) > 0 and len(book[asset].asks) > 0:
                    self.best_bid[asset] = book[asset].bids[0].px
                    self.best_ask[asset] = book[asset].asks[0].px
                    midprices[asset] = (
                        float(book[asset].asks[0].px) + float(book[asset].bids[0].px)
                    ) / 2
                elif bool(len(book[asset].bids) > 0) != bool(len(book[asset].asks) > 0):
                    if len(book[asset].bids) > 0:
                        self.best_bid[asset] = book[asset].bids[0].px
                        midprices[asset] = float(book[asset].bids[0].px)
                    else:
                        self.best_ask[asset] = book[asset].asks[0].px
                        midprices[asset] = float(book[asset].asks[0].px)
                else:
                    midprices[asset] = np.NaN
                if asset != "SPY":
                    parse = asset.split("SPY", 1)[1]
                    strike = np.float64(parse[:-1])
                    is_call = parse[-1] == "C"
                    if not np.isnan(midprices[asset]):
                        self.itm[asset] = (
                            strike > self.underlying
                            if not is_call
                            else strike < self.underlying
                        )

                if asset == "SPY" and not np.isnan(midprices["SPY"]):
                    self.underlying = midprices[asset]

            if not all(np.isnan(list(midprices.values()))):
                for asset in midprices:
                    if np.isnan(midprices[asset]) and self.history.shape[0] > 0:
                        midprices[asset] = self.history.iloc[-1][asset]
                self.history.loc[self.history.shape[0]] = midprices
                if self.history.shape[0] > 1:
                    self.returns.loc[self.returns.shape[0]] = (
                        self.history.iloc[-1] - self.history.iloc[-2]
                    ) / self.history.iloc[-2]
                    self.volatility = self.returns.std() * sqrt(252)
                    self.update_greeks()
                self.tick += 1
                if self.tick % 5 == 0:
                    await self.handle_order()

                print(await self.get_positions())

        elif kind == "fill_msg":
            fill_msg = update.fill_msg
            order_side = fill_msg.order_side
            filled_qty = fill_msg.filled_qty
            filled_asset = fill_msg.asset
            if order_side == 0:
                self.position[filled_asset] += filled_qty
            else:
                self.position[filled_asset] -= filled_qty
        elif kind == "generic_msg":
            msg = update.generic_msg.message
            print(msg)

    async def handle_read_params(self):
        while True:
            try:
                self.params = json.load(open(PARAM_FILE, "r"))
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(1)

    def total_greeks(self):
        greeks = np.zeros(4)
        for greek in range(4):
            for asset in CONTRACTS[1:]:
                greeks[greek] += self.position[asset] * self.greeks[asset][greek]
        return greeks

    def get_strike(self, asset):
        parse = asset.split("SPY", 1)[1]
        strike = np.float64(parse[:-1])
        return strike

    async def handle_order(self):
        if len(self.volatility) > 0 and not all(np.isnan(self.volatility)):
            epsilon = self.params["epsilon"]

            shares = self.params["shares"]

            otms = [asset for asset in CONTRACTS[1:] if not self.itm[asset]]

            otm_puts = np.array([asset for asset in otms if asset[-1] == "P"])
            otm_calls = np.array([asset for asset in otms if asset[-1] == "C"])

            puts_dist = np.argsort(
                [abs(self.get_strike(asset) - self.underlying) for asset in otm_puts]
            )
            calls_dist = np.argsort(
                [abs(self.get_strike(asset) - self.underlying) for asset in otm_calls]
            )

            order_puts = otm_puts[puts_dist]
            order_calls = otm_calls[calls_dist]

            i = 0

            while i < epsilon and i < len(puts_dist) and i < len(calls_dist):
                next_put = order_puts[i]
                next_call = order_calls[i]
                if self.position[next_put] < shares:
                    print(
                        f"Placing BID for {next_put} for {shares - self.position[next_put]} shares"
                    )
                    ok = await self.place_order(
                        next_put,
                        pb.OrderSpecType.MARKET,
                        pb.OrderSpecSide.BID,
                        shares - self.position[next_put],
                    )
                    print(ok)

                if self.position[next_call] < shares:
                    print(
                        f"Placing BID for {next_call} for {shares - self.position[next_call]} shares"
                    )

                    ok = await self.place_order(
                        next_call,
                        pb.OrderSpecType.MARKET,
                        pb.OrderSpecSide.BID,
                        shares - self.position[next_call],
                    )
                    print(ok)

                i += 1
            while i < len(puts_dist) and i < len(calls_dist):
                next_put = order_puts[i]
                next_call = order_calls[i]
                if self.position[next_put] > 0:
                    print(
                        f"Placing ASK for {next_put} for {self.position[next_put]} shares"
                    )
                    ok = await self.place_order(
                        next_put,
                        pb.OrderSpecType.MARKET,
                        pb.OrderSpecSide.ASK,
                        self.position[next_put],
                    )

                if self.position[next_call] > 0:
                    print(
                        f"Placing ASK for {next_call} for {self.position[next_put]} shares"
                    )
                    ok = await self.place_order(
                        next_call,
                        pb.OrderSpecType.MARKET,
                        pb.OrderSpecSide.ASK,
                        self.position[next_call],
                    )

                i += 1
            if self.tick % 20 == 0:
                print(self.total_greeks())


if __name__ == "__main__":
    start_bot(OptionBot)
