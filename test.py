import asyncio

if __name__ == '__main__':
    import asyncio
    from parser_driver.parsers.parser_telegram import get_telegram_channels
    
    channels = asyncio.run(get_telegram_channels())
    for idx, channel in enumerate(channels, 1):
        print(f"{idx}. {channel['title']} (@{channel['username']})")