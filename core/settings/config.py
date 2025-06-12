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
                                      env_prefix="LOGGING__")
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    format: str = LOG_DEFAULT_FORMAT
    
    access_log: bool = Field(default=True)

    @property
    def log_level(self) -> int:
        return getattr(logging, self.level)


class ConfigParserDriver(BaseSettings):

    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, 
                                      env_prefix="DRIVER__")
    
    url_parsing: str = Field(default=...)

    show_browser: bool = Field(default=True)
    window_size_w: int = Field(default=780)
    window_size_h: int = Field(default=700)

    @property
    def window_size(self) -> tuple[int]:
        return self.window_size_w, self.window_size_h
    
    def get_url(self, coin: str) -> str:
        return self.url_parsing.replace("{coin}", coin)
    

class ConfigDatabase(BaseSettings):

    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, 
                                      env_prefix="DATABASE__")
    
    user: str = Field(default=...)
    password: str = Field(default=...)
    host: str = Field(default="localhost")
    db_name: str = Field(default="db_name")
    port: int = Field(default=5432)

    echo: bool = Field(default=False)
    echo_pool: bool = Field(default=False)
    pool_size: int = Field(default=20)
    max_overflow: int = 50
    pool_timeout: int = 5

    naming_convention: dict[str, str] = {
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_N_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
    
    def get_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"


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
    database: ConfigDatabase = Field(default_factory=ConfigDatabase)

settings = ConfigParser()