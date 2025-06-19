from typing import Any
from datetime import timezone, timedelta
from telethon import TelegramClient, events
from telethon.tl.types import Channel

from core import settings
from core.database import NewsData, orm_add_news

import logging

logger = logging.getLogger("parser_logger.ParserTelegram")

BUFFER_SIZE = 100

class TelegramParser(TelegramClient):

    def __init__(self):
        self.phone = settings.tg.phone
        super().__init__('session', settings.tg.api_id, 
                         settings.tg.api_hash)
        self.buffer_messages = []
        self._db = None
        self.filter = None
        self.clear_text = None

    def set_filter(self, filter: callable):
        self.filter = filter

    def set_clear_text(self, clear_text: callable):
        self.clear_text = clear_text

    def init_db(self, db):
        self._db = db

    @property
    def db(self):
        return self._db

    async def add_message_to_buffer(self, message: NewsData):
        self.buffer_messages.append(message)

        if self.db:
            async with self.db.get_session() as session:
                await orm_add_news(session, message)

        if len(self.buffer_messages) > BUFFER_SIZE:
            self.buffer_messages.pop(0)

    def get_buffer_messages(self):
        return self.buffer_messages
    
    def procces_event(self, event) -> NewsData | None:
        if self.filter:
            if not self.filter(event):
                return
        
        if self.clear_text:
            text = self.clear_text(event.message.message)
        else:
            text = event.message.message
        
        id_url = event.sender_id
        title = "Telegram Channel {}".format(event.chat.title)
        date = event.message.date
        utc_plus_3 = timezone(timedelta(hours=3))
        date = date.astimezone(utc_plus_3)
        date = date.replace(tzinfo=None)

        news_data = NewsData(
            id_url=id_url,
            type="telegram",
            title=title,
            text=text,
            date=date
        )

        return news_data
    
    async def start_parser_event(self, events):
        try:
            logger.info(f"Start parser - {events=}")
            await self.start()
            # for events_parsing in events:
            for event_parsing, kwargs in events.items():
                await event_parsing(**kwargs)
                # self.add_event_handler(handler)
                logger.info(f"Start {event_parsing}")

            await self.run_until_disconnected()
        except Exception as e:
            logger.error(f"Error in start_parser_event: {e}")
        finally:
            # Корректное отключение
            await self.disconnect()
    
    async def event_parsing_telegram_channel_pattern(self, pattern):

        @self.on(events.NewMessage(pattern=pattern))
        async def handler(event):
            news_data = self.procces_event(event)
            if not news_data:
                return

            await self.add_message_to_buffer(news_data)
        
        # return handler
    
    async def event_parsing_telegram_channel_chat(self, chat: str | list[str]):

        @self.on(events.NewMessage(chats=chat))
        async def handler(event):
            news_data = self.procces_event(event)
            if not news_data:
                return

            await self.add_message_to_buffer(news_data)
        
        # return handler

    async def get_channel_id(self, username_chanel: str) -> int:
        if not await self.is_user_authorized():
            await self.start(self.phone)

        async with self as client:
            channel = await client.get_entity(username_chanel)
            print(f"ID канала: {channel.id}")
            print(f"Доступный username: {channel.username}")

        return channel.id
    
    async def get_telegram_channels(self) -> list[dict[str, Any]]:
        if not await self.is_user_authorized():
            await self.start(self.phone)
        
        # Получаем все диалоги
        dialogs = await self.get_dialogs()
        
        # Фильтруем каналы
        channels = []
        for dialog in dialogs:
            if isinstance(dialog.entity, Channel):
                channel_info = {
                    'id': dialog.entity.id,
                    'title': dialog.entity.title,
                    'username': dialog.entity.username,
                    'participants_count': dialog.entity.participants_count
                }
                channels.append(channel_info)
        
        await self.disconnect()

        return channels
