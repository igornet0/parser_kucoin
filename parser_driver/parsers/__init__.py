__all__ = ("ParserNews", "ParserKucoin", "KuCoinAPI",
            "ParserNewsApi", "TelegramParser")

from core.utils.gui_deps import GUICheck

if GUICheck.has_gui_deps():
    from parser_driver.parsers.parser_news import ParserNews
    from parser_driver.parsers.parser_kucoin import ParserKucoin
else:
    class ParserKucoin: pass
    class ParserNews: pass

from parser_driver.parsers.parser_kucoin_api import KuCoinAPI
from parser_driver.parsers.parser_news_api import ParserNewsApi
from parser_driver.parsers.parser_telegram import TelegramParser