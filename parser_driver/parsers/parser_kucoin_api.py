import pandas as pd
from datetime import datetime
from kucoin.client import User, Trade, Market

from parser_driver.api import ParserApi
from core import settings 
from core.models.dataset import Dataset

import logging

class KuCoinAPI(ParserApi):
        
    api_key = settings.api_key
    api_secret = settings.api_secret
    api_passphrase = settings.api_passphrase

    args_entry = (settings.api_key, settings.api_secret, settings.api_passphrase)
    user = User(*args_entry)
    trade = Trade(*args_entry)
    market = Market(*args_entry)

    logger = logging.getLogger("parser_logger.KuCoinAPI")

    def get_account_summary_info(self):
        return self.user.get_account_summary_info()

    @classmethod
    def get_stat(cls, symbol: str, currency: str = "USDT") -> dict:
        """
        {
            "time": 1602832092060, // time
            "symbol": "BTC-USDT", // symbol
            "buy": "11328.9", // bestAsk
            "sell": "11329", // bestBid
            "changeRate": "-0.0055", // 24h change rate
            "changePrice": "-63.6", // 24h change price
            "high": "11610", // 24h highest price
            "low": "11200", // 24h lowest price
            "vol": "2282.70993217", // 24h volume the aggregated trading volume in BTC
            "volValue": "25984946.157790431", // 24h total, the trading volume in quote currency of last 24 hours
            "last": "11328.9", // last price
            "averagePrice": "11360.66065903", // 24h average transaction price yesterday
            "takerFeeRate": "0.001", // Basic Taker Fee
            "makerFeeRate": "0.001", // Basic Maker Fee
            "takerCoefficient": "1", // Taker Fee Coefficient
            "makerCoefficient": "1" // Maker Fee Coefficient
        }
        """
        return cls.market.get_24hr_stats(f"{symbol}-{currency}")
    
    @classmethod
    def get_orders_market(cls, symbol: str, currency: str = "USDT"):
        return cls.market.get_order_book(f"{symbol}-{currency}")

    @classmethod
    def get_kline(cls, symbol: str, 
                  currency: str = "USDT",
                  time: str = "5m") -> Dataset: 
        """
        "1545904980", //Start time of the candle cycle "0.058", 
        //opening price "0.049", 
        //closing price "0.058", 
        //highest price "0.049", 
        //lowest price "0.018", 
        //Transaction amount "0.000945" 
        //Transaction volume 143676
        """

        cls.logger.info(f"Get coin: {symbol} time: {time=}")
        
        if time[-1] == "m":
            time = time.replace("m", "min")
        elif time[-1] == "H":
            time = time.replace("H", "hour")
        elif time[-1] == "D":
            time = time.replace("D", "day")
        elif time[-1] == "W":
            time = time.replace("W", "week")

        try:
            data = cls.market.get_kline(f"{symbol}-{currency}", time)
        except Exception as e:
            cls.logger.error(f"Error get kline {symbol}-{currency} - {e}")
            return None

        colums = ["datetime", "open", "close", "max", "min", "_", "volume"]

        df = pd.DataFrame(data, columns=colums).drop("_", axis=1)

        if len(df) == 0:
            cls.logger.error(f"Error get kline {symbol}-{currency} - {len(df)=}")
            return None
        
        df = df.drop(df.index[0])

        df["datetime"] = df["datetime"].apply(lambda x: datetime.fromtimestamp(int(x)))
        
        df["datetime"] = pd.to_datetime(df['datetime'])

        if "day" in time or "week" in time:
            df["datetime"] = df["datetime"].dt.strftime('%Y-%m-%d')
        else:
            df["datetime"] = df["datetime"].dt.strftime('%Y-%m-%d %H:%M:%S')

        df["volume"] = df["volume"].apply(float)

        return Dataset(df)