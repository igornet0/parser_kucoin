from . import db

class Coin:

    def __init__(self, name, time, open_price, close_price, max_price, min_price, value):
        self.name = name
        self.time = time

        self.open_price = open_price
        self.close_price = close_price
        
        self.max_price = max_price
        self.min_price = min_price
        
        self.value = value

    @staticmethod
    def create_table():
        db.create_table("coins", 
                        ["id INTEGER PRIMARY KEY AUTOINCREMENT",
                         "name TEXT", "time TEXT", "open_price REAL", "close_price REAL", 
                         "max_price REAL", "min_price REAL", "value REAL"])

    def insert(self):
        db.insert("coins", 
                  [self.name, self.time, self.open_price, self.close_price, self.max_price, self.min_price, self.value],
                  columns=["name", "time", "open_price", "close_price", "max_price", "min_price", "value"])
    