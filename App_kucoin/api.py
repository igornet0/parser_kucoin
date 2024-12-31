import pandas as pd
from datetime import datetime
from kucoin.client import User, Trade, Market

from .Log import Loger
from .NN_dataset import Dataset
from .models import Coin

class KuCoinAPI:

    def __init__(self, api_key, api_secret, api_passphrase, save: bool = False, 
                 path_save="datasets", logger: Loger=None):
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.user = User(api_key, api_secret, api_passphrase)
        self.trade = Trade(api_key, api_secret, api_passphrase)
        self.market = Market(api_key, api_secret, api_passphrase)

        self.logger = logger if logger else Loger().off

        if save:
            self.logger["INFO"]("Save data: True")
            self.save = save
            self.path_save = path_save

    def get_account_summary_info(self):
        return self.user.get_account_summary_info()
    
    def get_kline(self, symbol, time: str = "5m") -> pd.DataFrame: 
        """
        "1545904980", //Start time of the candle cycle "0.058", 
        //opening price "0.049", //closing price "0.058", //highest price "0.049", //lowest price "0.018", 
        //Transaction amount "0.000945" //Transaction volume 143676
        """

        self.logger["INFO"](f"Get coin: {symbol} time: {time=}")
        
        if time[-1] == "m":
            time = time.replace("m", "min")
        elif time[-1] == "H":
            time = time.replace("H", "hour")
        elif time[-1] == "D":
            time = time.replace("D", "day")
        elif time[-1] == "W":
            time = time.replace("W", "week")

        try:
            data = self.market.get_kline(f"{symbol}-USDT", time)
        except Exception as e:
            self.logger["ERROR"](f"Error get kline {e}")
            return None

        colums = ["datetime", "open", "high", "low", "close", "_", "volume"]

        df = pd.DataFrame(data, columns=colums).drop("_", axis=1)
        if len(df) == 0:
            self.logger["ERROR"](f"Error get kline {symbol}-USDT - {len(df)=}")
            return None
        
        df = df.drop(df.index[-1])

        df["datetime"] = df["datetime"].apply(lambda x: datetime.fromtimestamp(int(x)))
        
        df["datetime"] = pd.to_datetime(df['datetime'])
        if "day" in time or "week" in time:
            df["datetime"] = df["datetime"].dt.strftime('%Y-%m-%d')
        else:
            df["datetime"] = df["datetime"].dt.strftime('%Y-%m-%d %H:%M:%S')

        return Dataset(df, 
                       save=self.save, 
                       path_save=self.path_save, 
                       file_name=f"{symbol}_{time}.csv").get_dataset()