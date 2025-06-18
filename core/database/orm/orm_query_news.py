# файл для query запросов
from datetime import datetime
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from pydantic import BaseModel
from enum import Enum
from core.database.models import (News, NewsUrl, TelegramChannel)

class NewsType(Enum):
    
    telegram = "telegram"
    url = "url"

class NewsData(BaseModel):
    id_url: int
    title: str
    text: str
    type: NewsType
    date: datetime

##################### Добавляем новости в БД #####################################

async def orm_add_telegram_chanel(session: AsyncSession, name: str, chat_id: str, parsed: bool = True) -> TelegramChannel:
    
    channel = await session.execute(select(TelegramChannel).where(TelegramChannel.name == name))
    
    if channel.scalars().first():
        raise ValueError(f"Channel {name} already exists")
    
    session.add(
        TelegramChannel(name=name,
                        chat_id=chat_id,
                        parsed=parsed)
    )
    await session.commit()

async def orm_add_news_url(session: AsyncSession, url: str, a_pup: float = 0.9, parsed: bool = True) -> NewsUrl:
    
    channel = await session.execute(select(NewsUrl).where(NewsUrl.url == url))
    
    if channel.scalars().first():
        raise ValueError(f"Url {url} already exists")
    
    news = NewsUrl(url=url,
                a_pup=a_pup,
                parsed=parsed)

    session.add()
    await session.commit()

    await session.refresh(news)

    return news

async def orm_get_news_url(session: AsyncSession, url: str) -> NewsUrl:
    query = select(NewsUrl).where(NewsUrl.url == url)
    result = await session.execute(query)
    return result.scalars().first()

async def orm_get_news_url_by_id(session: AsyncSession, id: int) -> NewsUrl:
    query = select(NewsUrl).where(NewsUrl.id == id)
    result = await session.execute(query)
    return result.scalars().first()

async def orm_get_telegram_channel(session: AsyncSession, name: str) -> TelegramChannel:
    query = select(TelegramChannel).where(TelegramChannel.name == name)

    result = await session.execute(query)
    return result.scalars().first()

async def orm_get_telegram_channel_by_id(session: AsyncSession, id: int) -> TelegramChannel:
    query = select(TelegramChannel).where(TelegramChannel.id == id)
  
    result = await session.execute(query)
    return result.scalars().first()

async def orm_get_news_urls(session: AsyncSession, parsed: bool = None) -> list[NewsUrl]:
    query = select(NewsUrl)

    if parsed:
        query = query.where(NewsUrl.parsed == parsed)

    result = await session.execute(query)
    return result.scalars().all()

async def orm_get_news_telegram_channels(session: AsyncSession, parsed: bool = None) -> list[TelegramChannel]:
    query = select(TelegramChannel)

    if parsed:
        query = query.where(TelegramChannel.parsed == parsed)

    result = await session.execute(query)
    return result.scalars().all()

async def orm_add_news(session: AsyncSession, data: NewsData) -> News:
    
    news = News(id_url=data.id_url,
                type=data.type.value,
                title=data.title,
                text=data.text,
                date=data.date)
    
    session.add(news)

    await session.commit()
    await session.refresh(news)
    
    return news