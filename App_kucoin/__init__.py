from .Log import Loger
from .NN_dataset import Dataset
from .Sqlither import Sqlither

# __all__ = ["KuCoinAPI", "Loger"]

db = Sqlither("database.db")

from .models import Coin
from .api import KuCoinAPI

__all__ = ["KuCoinAPI", "Loger", 
           "Dataset", "Sqlither", "Coin"]