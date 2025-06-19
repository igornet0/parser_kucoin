from pathlib import Path
import json

class NewsSettings:
    
    pass


class TelegramSettings:

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    FILE_TG_WORDS: Path = BASE_DIR / "settings" / "tg_words.json"

    def __init__(self):

        
        data = self._load_data()
        self.words = data["words_pattern"]
        self.pop_words = data["pop_words"]
    
    def _load_data(self):
        with open(self.FILE_TG_WORDS, "r") as f:
            data = json.load(f)
        return data

news_settings = NewsSettings()
telegram_settings = TelegramSettings()

# URL_SETTINGS = {
#     # "https://ambcrypto.com/category/new-news/":{
#     #     "next_page": "load more articles",
#     #     "filter_text": lambda x: len(x) > 26,
#     #     "news": {
#     #         "SHOW": True,
#     #         "SCROLL": -1000,
#     #         "ZOOM": 0.6,
#     #         "text_start": "title",
#     #         "text_end": ["Disclaimer:"],
#     #         "tag_end": ["a//take a survey:"],
#     #         "text_continue": ["2min", "source:"],
#     #         "img_continue": ["alt@avatar"],
#     #         "date_format": "posted: %B %d, %Y",
#     #         "filter_tags": ["h1", "h2", "p", "em", "span", "img"]
#     #         }
#     #     },
#     # "https://cryptoslate.com/crypto/":{
#     #     "CAPTHA": True,
#     #     "next_page": "next page",
#     #     "filter_text": lambda x: len(x) > 20,
#     #     "clear": True,
#     #     "news": {
#     #         "SHOW": True,
#     #         "SCROLL": -1000,
#     #         "ZOOM": 0.6,
#     #         "text_start": "title",
#     #         "tag_end": ["div//Posted In:"],
#     #         "text_continue": ["2min", "source:"],
#     #         "img_continue": ["data-del@avatar"],
#     #         "date_format": "Updated: %b. %d, %Y at %I:%M %p %Z",
#     #         "filter_tags": ["h1", "h2", "p", "em", "span", "img"]
#     #         }
#     # },
    
#     "telegram":{
#         "CAPTHA": False,
#         "chanels": {
#             "🔥Full-Time Trading": {
#                 "filter_text": ["#Крипта"]
#             },
#         }
#     }
# }