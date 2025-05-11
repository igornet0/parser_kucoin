__all__ = ("DataParser", 
           "ParserApi", 
           "ParserNews", 
           "ParserKucoin", 
           "KuCoinAPI"
           )

from parser_driver.data import DataParser
from parser_driver.api import ParserApi
from parser_driver.parsers import KuCoinAPI
from core.utils.gui_deps import GUICheck

if GUICheck.has_gui_deps():
    from parser_driver.parsers import ParserNews, ParserKucoin
    
# from handlers.parser_handler import Handler as HandlerParser
# from .parser_bcs import Parser_bcs
# from .parser_marketcap import Parser_marketcap
# from .parser_kucoin import Parser_kucoin
# from .parser_news import Parser_news