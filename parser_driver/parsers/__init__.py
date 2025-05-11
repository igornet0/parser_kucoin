__all__ = ("ParserNews", "ParserKucoin", "KuCoinAPI")

from core.utils.gui_deps import GUICheck

if GUICheck.has_gui_deps():
    from parser_driver.parsers.parser_news import ParserNews
    from parser_driver.parsers.parser_kucoin import ParserKucoin

from parser_driver.parsers.parser_kucoin_api import KuCoinAPI