import logging
from logging.handlers import RotatingFileHandler
from typing import Literal
from pydantic import BaseModel
from core.settings import settings
from core import DataManager
import sys

logging.getLogger("selenium").setLevel(logging.WARNING)
logging.getLogger("undetected_chromedriver").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("pytesseract").setLevel(logging.WARNING)

class OverwriteHandler(logging.StreamHandler):
    __on: bool = False

    @property
    def on(self):
        self.__on = True

    @property
    def off(self):
        self.__on = False

    def get_status_on(self):
        return self.__on

    def emit(self, record):
        if self.get_status_on():
            try:
                msg = self.format(record)
                stream = self.stream
                stream.write('\r' + msg)  # Перезаписываем строку
                stream.flush()
            except Exception as e:
                self.handleError(record)
        else:
            super().emit(record)

def setup_logging():
    # Очищаем все существующие обработчики
    for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
        logger.handlers = []

    # Корневой логгер (все сообщения)
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.logging.log_level)
    
    # Форматтер для всех логов
    formatter = logging.Formatter(settings.logging.format)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.logging.log_level)
    
    # Общий файловый хендлерs
    common_handler = RotatingFileHandler(DataManager()["log"] / "all.log", maxBytes=1e6, backupCount=3)
    common_handler.setFormatter(formatter)

    # Настройка для parser_logger
    parser_logger = logging.getLogger("parser_logger")
    parser_handler = RotatingFileHandler(DataManager()["log"] / "parser_logger.log", maxBytes=1e6, backupCount=3)
    parser_handler.setFormatter(formatter)
    parser_handler.setLevel(settings.logging.log_level)
    parser_logger.addHandler(parser_handler)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(common_handler)

