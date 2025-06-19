import asyncio
import aiohttp
import time
from datetime import datetime 

from parser_driver.api import ParserApi
from core.database import NewsData, orm_add_news
from core.settings import settings 

import logging

logger = logging.getLogger("parser_logger.ParserNewsApi")

class ParserNewsApi(ParserApi):
    # 4 request max per minute

    api_key = settings.coindesk.api_key

    URL_API = "https://data-api.coindesk.com"

    clieat_text = lambda text: text

    def set_clear_text(self, func):
        self.clear_text = func

    async def get_last_news(self, limit: int = 10, 
                            last_publish: datetime = None) -> list[NewsData] | None:

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.URL_API}/news/v1/article/list",
                                        params={"lang":"EN","limit":limit,"api_key": self.api_key},
                                        headers={"Content-type":"application/json; charset=UTF-8"}) as response:
                    
                    if response.status != 200:
                        raise Exception(f"Error get_last_news {response.status} {response}")
                    
                    json_response = await response.json()

        except Exception as e:
            logger.error(f"Error get_last_news {e}")
            return None
        
        news_list = []

        for data in json_response["Data"]:
            date = datetime.fromtimestamp(data["PUBLISHED_ON"])
            if last_publish and date <= last_publish:
                break

            news = NewsData(
                id_url=data["ID"],
                type=data["SOURCE_DATA"]["SOURCE_TYPE"],
                title=data["TITLE"],
                text=self.clear_text(data["BODY"]),
                date=date
            )

            news_list.append(news)

            if self.db:
                async with self.db.get_session() as session:
                    try:
                        await orm_add_news(session, news)
                    except:
                        break

        return news_list