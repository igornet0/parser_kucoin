import asyncio
from parser_driver.parsers.parser_telegram import TelegramParser
from core import settings

async def main():

    channel_username = "@markettwits"
    tg_parser = TelegramParser(api_id=settings.tg.api_id, 
                               api_hash=settings.tg.api_hash, 
                               phone=settings.tg.phone)
    try:
        await tg_parser.start()

        await tg_parser.parsing_telegram_channel(channel_username)
        
        # Бесконечный цикл прослушивания
        print(f"Слушаем канал: {channel_username}...")
        await tg_parser.run_until_disconnected()
        
    finally:
        # Корректное отключение
        await tg_parser.disconnect()
    


if __name__ == '__main__':

    asyncio.run(main())
    # for idx, channel in enumerate(channels, 1):
    #     print(f"{idx}. {channel['title']} (@{channel['username']})")