"""
This file contains the primary strategy for market-making in the soybeans market.
"""
from collections import defaultdict
from typing import DefaultDict, Dict, Tuple
from utc_bot import UTCBot, start_bot
import math
import proto.utc_bot as pb
import betterproto
import asyncio
import re
import itertools
import numpy as np

DAYS_IN_MONTH = 21
DAYS_IN_YEAR = 252
INTEREST_RATE = 0.02
NUM_FUTURES = 14
TICK_SIZE = 0.01
MULTIPLIER = 10000
ACTIVE_THRESHOLD = 5
FUTURE_CODES = [chr(ord("A") + i) for i in range(NUM_FUTURES)]
CONTRACTS = ["SBL"] + ["LBS" + c for c in FUTURE_CODES] + ["LLL"]


class Case1Bot(UTCBot):
    etf_suffix = ""

    async def create_etf(self, qty: int):
        if len(self.etf_suffix) == 0:
            return pb.SwapResponse(False, "Unsure of swap")
        return await self.swap("create_etf_" + self.etf_suffix, qty)

    async def redeem_etf(self, qty: int):
        if len(self.etf_suffix) == 0:
            return pb.SwapResponse(False, "Unsure of swap")
        return await self.swap("redeem_etf_" + self.etf_suffix, qty)

    async def days_to_expiry(self, asset):
        future = ord(asset[-1]) - ord("A")
        expiry = 21 * (future + 1)
        return self._day - expiry

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "generic_msg":
            msg = update.generic_msg.message

            if "trade_etf" in msg:
                self.etf_suffix = msg.split(" ")[1]

            if "Weather" in update.generic_msg.message:
                msg = update.generic_msg.message
                weather = float(re.findall("\d+\.\d+", msg)[0])
                self._weather_log.append(weather)

            if "Day" in update.generic_msg.message:
                self._day = int(re.findall("\d+", msg)[0])

            else:
                resp = await self.get_positions()
                if resp.ok:
                    self.positions = resp.positions

        elif kind == "market_snapshot_msg":
            for asset in CONTRACTS:
                book = update.market_snapshot_msg.books[asset]
                if book.bids:
                    self._best_bid[asset] = float(book.bids[0].px)
                if book.asks:
                    self._best_ask[asset] = float(book.asks[0].px)

        elif kind == "fill_msg":
            fill_msg = update.fill_msg

            order_side = fill_msg.order_side
            filled_qty = fill_msg.filled_qty
            month = int(self._day / 21)
            await self.make_hedge_trades(filled_qty, month, order_side)

    async def handle_round_started(self):
        self._day = 0

        self._best_bid: Dict[str, float] = defaultdict(lambda: 0)

        self._best_ask: Dict[str, float] = defaultdict(lambda: 0)

        self.__orders: DefaultDict[str, Tuple(str, float)] = defaultdict(
            lambda: ("", 0)
        )

        self.ask_id = self.bid_id = self.ask_price = self.bid_price = 0
        self.order_id_iter = itertools.count(1)
        self.enter = True

        self._weather_log = []

        await asyncio.sleep(0.01)

        asyncio.create_task(self.make_market_asset("LLL"))

    async def make_hedge_trades(self, quantity, month, side):
        future1, future2, future3 = (
            "LBS" + FUTURE_CODES[month],
            "LBS" + FUTURE_CODES[month + 1],
            "LBS" + FUTURE_CODES[month + 2],
        )
        if side == 0:
            await self.place_order(
                asset_code=future1,
                order_type=pb.OrderSpecType.MARKET,
                order_side=1,
                qty=80,
            )
            await self.place_order(
                asset_code=future2,
                order_type=pb.OrderSpecType.MARKET,
                order_side=1,
                qty=80,
            )
            await self.place_order(
                asset_code=future3,
                order_type=pb.OrderSpecType.MARKET,
                order_side=1,
                qty=80,
            )
        if side == 1:
            await self.place_order(
                asset_code=future1,
                order_type=pb.OrderSpecType.MARKET,
                order_side=0,
                qty=80,
            )
            await self.place_order(
                asset_code=future2,
                order_type=pb.OrderSpecType.MARKET,
                order_side=0,
                qty=80,
            )
            await self.place_order(
                asset_code=future3,
                order_type=pb.OrderSpecType.MARKET,
                order_side=0,
                qty=80,
            )
        await asyncio.sleep(0.1)

    async def make_market_asset(self, asset: str):
        while self._day <= DAYS_IN_YEAR:
            new_ask_price, new_bid_price = self._best_ask[asset], self._best_bid[asset]

            if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
                await self.cancel_order(str(self.bid_id))
                self.bid_id = 0
            if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
                await self.cancel_order(str(self.ask_id))
                self.ask_id = 0

            if self.enter:
                old_bid_price, old_ask_price = (
                    self.__orders[f"yesterdays_bid_{asset}"][1],
                    self.__orders[f"yesterdays_ask_{asset}"][1],
                )

                order = (
                    ["ask", "bid"]
                    if (new_ask_price + new_bid_price) > (old_ask_price + old_bid_price)
                    else ["bid", "ask"]
                )
                for side in order:
                    if side == "bid":
                        self.bid_id = next(self.order_id_iter)
                        order_id = self.bid_id
                        order_side = pb.OrderSpecSide.BID
                        order_price = new_bid_price + TICK_SIZE
                    else:
                        self.ask_id = next(self.order_id_iter)
                        order_id = self.ask_id
                        order_side = pb.OrderSpecSide.ASK
                        order_price = new_ask_price - TICK_SIZE
                    await self.place_order(
                        asset_code=asset,
                        order_type=pb.OrderSpecType.LIMIT,
                        order_side=order_side,
                        qty=80,
                        px=round_nearest(order_price, TICK_SIZE),
                    )
                    self.__orders[f"yesterdays_{side}_{asset}"] = (
                        str(order_id),
                        order_price,
                    )

            curr_midpoint = (self._best_ask[asset] + self._best_bid[asset]) / 2
            old_midpoint = (
                self._best_ask[f"yesterdays_best_ask_{asset}"]
                + self._best_bid[f"yesterdays_best_bid_{asset}"]
            ) / 2

            self._best_ask[f"yesterdays_best_ask_{asset}"] = self._best_ask[asset]
            self._best_bid[f"yesterdays_best_bid_{asset}"] = self._best_bid[asset]

            if curr_midpoint:
                self.enter = (
                    True
                    if np.abs(
                        (curr_midpoint - old_midpoint) / curr_midpoint * MULTIPLIER
                    )
                    > 5
                    else False
                )
            await asyncio.sleep(0.1)


def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))


if __name__ == "__main__":
    start_bot(Case1Bot)
