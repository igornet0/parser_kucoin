import pandas as pd
from datetime import datetime
from kucoin.client import User, Trade, Market

from parser_driver.api import ParserApi
from core import settings 
from core.models.dataset import Dataset
from core.utils import setup_logging

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
        
        df = df.drop(df.index[-1])

        df["datetime"] = df["datetime"].apply(lambda x: datetime.fromtimestamp(int(x)))
        
        df["datetime"] = pd.to_datetime(df['datetime'])

        if "day" in time or "week" in time:
            df["datetime"] = df["datetime"].dt.strftime('%Y-%m-%d')
        else:
            df["datetime"] = df["datetime"].dt.strftime('%Y-%m-%d %H:%M:%S')

        df["volume"] = df["volume"].apply(float)

        return Dataset(df)