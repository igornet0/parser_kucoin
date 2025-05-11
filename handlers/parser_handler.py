import asyncio
# from fastapi import UploadFile
UploadFile = "UploadFile"
from typing import Dict, Any
from inspect import signature, Parameter
import pandas as pd

from core.utils.gui_deps import GUICheck

if GUICheck.has_gui_deps():
    from parser_driver import ParserKucoin, ParserNews
else:
    class ParserKucoin: pass
    class ParserNews: pass

from parser_driver import ParserApi, KuCoinAPI

import logging

logger = logging.getLogger("parser_logger.Handler")

class Handler:

    _parsers = {
        "parser kucoin driver": ParserKucoin,
        "parser kucoin api": KuCoinAPI,
        "parser news": ParserNews,
        "parser api": ParserApi
    }

    @classmethod
    def get_parser(cls, parser_type: str, *args, **kwargs) -> Any:
        try:
            parser_class = cls._parsers.get(parser_type)
            
            if not parser_class:
                raise ValueError("Invalid parser type")
            
            parser = parser_class(*args, **kwargs)

            return parser
        
        except Exception as e:
            logger.error(f"Error {e=}")
            return None


    @classmethod
    async def run_parser(cls, parser_type: str, method: str, 
                         init_params: dict, method_params: dict, files: dict):
        try:
            parser_class = cls._parsers.get(parser_type)
            if not parser_class:
                raise ValueError("Invalid parser type")
            
            parser = parser_class(**init_params)

            if not hasattr(parser, method):
                raise ValueError(f"Method {method} not found in parser")
            
            # Заменяем файловые параметры
            for param_name, file in files.items():
                if param_name in method_params:
                    method_params[param_name] = file
                    
            method = getattr(parser, method)

            logger.info(f"Running method {method.__name__}")

            return await method(**method_params)
        except Exception as e:
            logger.error(f"Error {e=}")
            return None


    @classmethod
    def get_available_parsers(cls):
        return list(cls._parsers.keys())
    
    @classmethod
    def get_parser_params(cls, parser_type: str) -> Dict[str, Any]:
        parser_class = cls._parsers.get(parser_type)
        
        if not parser_class:
            raise ValueError("Invalid parser type")
        
        # Получаем параметры конструктора
        init_sig = signature(parser_class.__init__)
        init_params = {}
        
        for name, param in init_sig.parameters.items():
            if name == 'self':
                continue
                
            init_params[name] = {
                'type': param.annotation.__name__ if param.annotation != Parameter.empty else 'str',
                'default': param.default if param.default != Parameter.empty else None,
                'optional': param.default != Parameter.empty,
                'description': 'Параметр парсера'
            }
        
        return init_params
    
    @classmethod
    def get_parser_info(cls, parser_type: str) -> Dict[str, Any]:
        parser_class = cls._parsers.get(parser_type)
        
        init_params = cls.get_parser_params(parser_type)
        
        # Получаем информацию о методах
        methods_info = {}
        for method_name in dir(parser_class):
            method = getattr(parser_class, method_name)
            
            # Пропускаем специальные и приватные методы
            if method_name.startswith('_') or not callable(method):
                continue
            
            # Проверяем наличие UI декоратора
            if not (hasattr(method, '_is_ui_method') and getattr(method, '_is_exposed', True)):
                continue
            
            # Получаем метаданные из декоратора
            file_params = getattr(method, '_file_params', {})
            method_description = getattr(method, '_ui_description', method.__doc__ or '')
            
            # Анализируем сигнатуру метода
            method_sig = signature(method)
            method_params = {}
            
            for param_name, param in method_sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                # Базовые характеристики параметра
                param_info = {
                    'type': 'str',
                    'default': None,
                    'optional': False,
                    'description': '',
                    'extensions': []
                }
                
                # Определение типа
                if param.annotation != Parameter.empty:
                    param_info['type'] = param.annotation.__name__
                    if param.annotation == UploadFile:
                        param_info['type'] = 'file'
                
                # Значение по умолчанию и обязательность
                if param.default != Parameter.empty:
                    param_info['default'] = param.default
                    param_info['optional'] = True
                
                # Дополнительные параметры для файлов
                if param_info['type'] == 'file' and param_name in file_params:
                    param_info.update({
                        'extensions': file_params[param_name].get('extensions', []),
                        'description': file_params[param_name].get('description', '')
                    })
                
                method_params[param_name] = param_info
            
            methods_info[method_name] = {
                'description': method_description,
                'params': method_params,
                'exposed': getattr(method, '_is_exposed', True)
            }
        
        return {
            'init_params': init_params,
            'methods': methods_info
        }

    # def set_parser(self, parser: ParserKucoin | ParserNews):
    #     if not isinstance(parser, ParserApi):
    #         raise ValueError("Parser must be instance of ParserApi")
        
    #     self.parser = parser

    # async def start_parser(self, counter: int = 1, show_browser: bool = True, window_size: tuple = (780, 1000)) -> pd.DataFrame:
    #     if not self.parser:
    #         raise ValueError("Parser not found")
        
    #     try:
    #         await self.parser.start_web(URL=self.URL, show_browser=show_browser, window_size=window_size)

    #         if self.login and self.password:
    #             self.parser.entry(self.login, self.password)

    #         _, data = self.parser.start_parser(init_counter=counter)
    #         self.parser.close()
    #         return data

    #     except Exception as e:
    #         logger.error(f"Error {e=}")
    #         return None