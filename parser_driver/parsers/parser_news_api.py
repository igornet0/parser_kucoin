import asyncio
import aiohttp
import time
from datetime import datetime 

from parser_driver.api import ParserApi
from core.database import NewsData
from core.settings import settings 

import logging

logger = logging.getLogger("parser_logger.ParserNewsApi")

class ParserNewsApi(ParserApi):
    # 4 request max per minute

    api_key = settings.coindesk.api_key

    URL_API = "https://data-api.coindesk.com"

    @classmethod
    async def get_last_news(cls, limit: int = 10) -> list[NewsData] | None:

        try:
            with aiohttp.ClientSession() as session:
                async with session.get(f"{cls.URL_API}/news/v1/article/list",
                                        params={"lang":"EN","limit":limit,"api_key": cls.api_key},
                                        headers={"Content-type":"application/json; charset=UTF-8"}) as response:
                    
                    if response.status != 200:
                        raise Exception(f"Error get_last_news {response.status} {response}")
                    
                    json_response = response.json()

        except Exception as e:
            logger.error(f"Error get_last_news {e}")
            return None
        
        news = []
        for data in json_response["data"]:
            news.append(NewsData(
                id_url=data["id"],
                type=data["type"],
                title=data["TITLE"],
                text=data["BODY"],
                date=datetime.fromtimestamp(data["PUBLISHED_ON"])
            ))

        return news