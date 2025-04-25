from functools import wraps
import pandas as pd
from typing import Dict

class AutoDecorator:

    func = None
    func_next = None
    _obj = None

    def __init__(self, obj):
        self._obj = obj
    
    @classmethod
    def set_func(cls, func):
        cls.func = func
    
    @classmethod
    def set_func_next(cls, func):
        cls.func_next = func

    def obj(self):
        return self._obj

    async def __call__(self, *args, **kwds):

        if self.func is None:
            raise Exception("Function not set")
        
        result = await self.func(*args, **kwds)
        
        for coin, dataset in result.items():
            dataset = self.obj().clear_dataset(dataset, coin)
            result[coin] = dataset

        self.obj().buffer_data.append(result)
        if self.func_next is None:
            return result
        
        result = await self.func_next(*args, **kwds)
        
        return result
    