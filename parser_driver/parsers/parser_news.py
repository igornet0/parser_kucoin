
import os 
import time
from datetime import datetime 
import requests
import threading
import multiprocessing as mlp
import re

from selenium.webdriver.common.by import By
import pandas as pd

from parser_driver.api import ParserApi


class ParserNews(ParserApi):

    def __init__(self, setting: dict = None, tick: int = 1) -> None:

        super().__init__(tick=tick)

        self.setting = setting

    # def load_settings(self, url: str):
    #     self.setting = URL_SETTINGS.get(url)
    #     return self.setting
    
    def clear_text(self, text: str):
        cleaned = re.sub(r'b[A-Z]+s+d+s+[A-Z]+b', '', text)
        # Удаляем лишние пробелы
        cleaned = cleaned.strip()

        return cleaned
    
    def url_getter(self, data: list,  urls: dict = {}, filter_text = lambda x: True) -> dict[str, str]:
        for link in data:
            text = link.text.strip()
            url = link.get_attribute("href")
            if text and filter_text(text):
                if self.setting.get("clear", False):
                    text = self.clear_text(text)
                urls[text] = url
        
        return urls
    

    def parser_elements(self, elements, title, setting):

        if setting.get("CAPTHA", False):
            self.captcha_solver()
        
        if setting.get("filter_text", False):
            filter_text = setting.get("filter_text")
        else:
            filter_text = lambda x: True        

        filter_tags: list[str] = setting.get("filter_tags", [])
        text_start: str = setting.get("text_start") 
        text_end: list[str] = setting.get("text_end")
        tag_end: dict[str, str] = {tag.split("//")[-1]: tag.split("//")[0] for tag in setting.get("tag_end")}
        text_continue: list[str] = setting.get("text_continue")
        img_continue: list[str] = setting.get("img_continue")
        date_format: str = setting.get("date_format")

        flag = False
        date = None
        text_page = []
        imgs = []
        n = 1

        for element in elements:
            if not element.tag_name in filter_tags or element.tag_name in tag_end.values():
                text = element.text.strip().replace("\n", "").lower()
                if tag_end and text and tag_end.get(text, False):
                    break

                continue

            if flag and element.tag_name == "img":

                if img_continue and any(element.get_attribute(x.split("@")[0]).lower().startswith(x.split("@")[-1].lower()) for x in img_continue):
                    continue

                img_src = element.get_attribute("src")

                try:
                    text = f"IMG_{n}"
                    response = requests.get(img_src)     
                    path_file = f"{len(os.listdir(os.path.join(PATH_DATASET, 'images'))) + 1}.png"
                    Img(response.content).save(path_file)
                    imgs.append(path_file)  
                    n += 1

                except Exception as e:
                    # print(f"Error loading image: {e}")
                    continue
            
            else:
                text = element.text.strip().replace("\n", "").lower()

            if text:
                if text_start == "title":
                    if title.lower().startswith(text):
                        flag = True
                        continue
                else:
                    if any(text.startswith(x.lower()) for x in text_start):
                        flag = True
                        continue

                if text_continue and any(text.startswith(x.lower()) for x in text_continue):
                    continue

                elif text_end and any(text.endswith(x.lower()) for x in text_end):
                    break

                elif date is None:
                    try:
                        date = datetime.strptime(text, date_format)
                    except Exception as e:
                        pass
                
                if flag:
                    if not text in text_page:
                        text_page.append(text)

        return date, text_page, imgs
    
    def captcha_solver(self):
        lfk = threading.Thread(target=self.device.kb.create_lfk, args=("s", "[INFO] For start press '{}'"), daemon=True,)
        lfk.start()

        while True:
            if self.device.kb.get_loop():
                break

        return True
            
    
    def start_parser(self, ulr:str=None, counter_news=1) -> pd.DataFrame:

        if not self.setting:
            raise Exception("Setting not found")
        
        self.load_settings(ulr)
        
        self.start_web(ulr, show_browser=self.setting.get("CAPTHA", self.setting.get("SHOW", False)))

        if self.setting.get("CAPTHA", False):
            self.captcha_solver()

        for url in self.urls:
            
            self.start_web(url, show_browser=self.setting.get("CAPTHA", False))

            if self.setting.get("CAPTHA", False):
                self.captcha_solver()
            
            if self.setting.get("filter_text", False):
                filter_text = self.setting.get("filter_text")
            else:
                filter_text = lambda x: True        

            news_urls = self.url_getter(self.get(), filter_text=filter_text)

            if self.setting.get("next_page"):
                for _ in range(counter_news):
                    element = self.search_element(self.get(), self.setting.get("next_page"))
                    self.click(element)
                    time.sleep(2)
                    news_urls = self.url_getter(self.get(), news_urls)
            
            print(f"[INFO parser] {url} {len(news_urls)=}")

            settings_news = self.setting.get("news")

            if not settings_news:
                data = pd.DataFrame(columns=[ "url", "title"])
            else:
                data = pd.DataFrame(columns=["datetime", "url", "title", "text", "imgs"])

            for title, url_news in news_urls.items():
                if not settings_news:
                    data.loc[len(data)] = [url_news, title]
                    continue
                    
                if title in data["title"].values:
                    continue

                self.start_web(url_news, show_browser=self.setting.get("CAPTHA", settings_news.get("ZOOM", False)))
                
                if settings_news.get("CAPTHA", False):
                    self.captcha_solver()

                if settings_news.get("ZOOM", False):
                    self.driver_instance.execute_script(f"document.body.style.zoom = '{settings_news.get('ZOOM') * 100}%'")

                if settings_news.get("SHOW", False):
                    scroll = settings_news.get("SCROLL", False)
                    if scroll:
                        for _ in range(abs(scroll//100)):
                            self.device.cursor.scroll(scroll//10)
                            time.sleep(self.tick)
                setting_news = self.setting.get("news")
                
                elements = self.get(tag="*")
                date, text_page, imgs = self.parser_elements(title, setting_news)
                data.loc[len(data)] = [date, url_news, title, " ".join(text_page), " ".join(imgs)]
            
        return self.save_data(data) if self.save else data