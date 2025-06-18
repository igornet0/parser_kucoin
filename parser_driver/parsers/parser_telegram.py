from typing import Any
from telethon import TelegramClient
from telethon.tl.types import Channel


class TelegramParser(TelegramClient):

    def __init__(self, api_id, api_hash, phone):
        super().__init__('session', api_id, api_hash)
        self.phone = phone

    async def get_channel_id(self, username_chanel: str) -> int:
        if not self.is_user_authorized():
            await self.start(self.phone)

        async with self as client:
            channel = await client.get_entity(username_chanel)
            print(f"ID канала: {channel.id}")
            print(f"Доступный username: {channel.username}")

        return channel.id
    
    async def get_telegram_channels(self) -> list[dict[str, Any]]:
        if not self.is_user_authorized():
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
