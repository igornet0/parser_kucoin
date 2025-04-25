from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import logging

LOG_DEFAULT_FORMAT = '[%(asctime)s] %(name)-35s:%(lineno)-3d - %(levelname)-7s - %(message)s'

class AppBaseConfig:
    """Базовый класс для конфигурации с общими настройками"""
    case_sensitive = False
    env_file = "./settings/prod.env"
    env_file_encoding = "utf-8"
    env_nested_delimiter="__"
    extra = "ignore"

class LoggingConfig(BaseSettings):
    
    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, 
                                      env_prefix="LOGGING_")
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    format: str = LOG_DEFAULT_FORMAT
    
    access_log: bool = Field(default=True)

    @property
    def log_level(self) -> int:
        return getattr(logging, self.level)

class ConfigParserDriver(BaseSettings):

    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, 
                                      env_prefix="DRIVER_")
    
    url_parsing: str = Field(default=...)

    show_browser: bool = Field(default=True)
    window_size_w: int = Field(default=780)
    window_size_h: int = Field(default=700)

    @property
    def window_size(self) -> tuple[int]:
        return self.window_size_w, self.window_size_h
    
    def get_url(self, coin: str) -> str:
        return self.url_parsing.replace("{coin}", coin)

class ConfigParser(BaseSettings):

    model_config = SettingsConfigDict(
        **AppBaseConfig.__dict__,
        env_prefix="KUCOIN__"
    )
    """Конфигурация для KuCoin API"""

    api_key: str = Field(default=...)
    api_secret: str = Field(default=...)
    api_passphrase: str = Field(default=...)

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    driver: ConfigParserDriver = Field(default_factory=ConfigParserDriver)

settings = ConfigParser()