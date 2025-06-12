# модели для БД
from sqlalchemy import DateTime, ForeignKey, Float, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base

class Coin(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    price_now: Mapped[float] = mapped_column(Float, default=0)

    max_price_now: Mapped[float] = mapped_column(Float, default=0)
    min_price_now: Mapped[float] = mapped_column(Float, default=0)
    open_price_now: Mapped[float] = mapped_column(Float, default=0)
    volume_now: Mapped[float] = mapped_column(Float, default=0)

    parsed: Mapped[bool] = mapped_column(Boolean, default=True)

    timeseries: Mapped[list['Timeseries']] = relationship(back_populates='coin')


class Timeseries(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'))  
    timestamp: Mapped[str] = mapped_column(String(50)) 
    path_dataset: Mapped[str] = mapped_column(String(100), unique=True)

    coin: Mapped['Coin'] = relationship(back_populates='timeseries')


class DataTimeseries(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timeseries_id: Mapped[int] = mapped_column(ForeignKey('timeseriess.id'))  
    datetime: Mapped[DateTime] = mapped_column(DateTime, nullable=False) 
    open: Mapped[float] = mapped_column(Float)
    max: Mapped[float] = mapped_column(Float)
    min: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)