import asyncio
from parser_driver.parsers.parser_telegram import TelegramParser
import re
from core import settings

async def main():

    channel_username = "@markettwits"
    tg_parser = TelegramParser(api_id=settings.tg.api_id, 
                               api_hash=settings.tg.api_hash, 
                               phone=settings.tg.phone)
    try:

        words = ["сша", "btc", "crypto", "news", "ftt", "крипто", "стейблкоин",
                 "геополитика", "эконом", "трейдинг_рф", "нефть"]
        escaped = sorted([re.escape(word) for word in words], key=len, reverse=True)
        pattern = r'(?<!\w)(?:' + '|'.join(escaped) + r')(?!\w)'
        regex = re.compile(pattern, flags=re.IGNORECASE)
        # channels = await tg_parser.get_telegram_channels_p()  
        # print(f"Количество каналов: {len(channels)}")
        # parsing = {}
        # for idx, channel in enumerate(channels, 1):
        #     print(f"{idx}. {channel['title']} (@{channel['username']}) ({channel['participants_count']})")
        #     if "Full-Time" in channel['title']:
        # await tg_parser.parsing_telegram_channel(channel['title'])

        # Бесконечный цикл прослушивания
        # print(f"Слушаем канал: {channel['title']}...")
        # await tg_parser.start()
        # await tg_parser.event_parsing_telegram_channel_pattern(regex)
        # await tg_parser.event_parsing_telegram_channel_chat(-1001292964247)
        # print(f"Слушаем {regex}")
        await tg_parser.start_parser_event({tg_parser.event_parsing_telegram_channel_pattern: {"pattern": regex}})
        # await tg_parser.run_until_disconnected()
        
    finally:
        # Корректное отключение
        await tg_parser.disconnect()
    


if __name__ == '__main__':

    asyncio.run(main())
    # for idx, channel in enumerate(channels, 1):
    #     print(f"{idx}. {channel['title']} (@{channel['username']})")