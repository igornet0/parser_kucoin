from parser_driver.api import ParserApi
import pandas as pd

class Parser_marketcap(ParserApi):

    def __init__(self, tick = 1, save = False, path_save="datasets_coins", DEBUG=False):
        super().__init__(tick, save, path_save, DEBUG)

        self.xpath_defaul_vxod = []
        self.page = 1

    def next_page(self):
        self.driver_instance.get(f"{self.URL}/?page={self.page + 1}")
        self.page += 1

    def default_xpath_marketcap(self):
        self.add_xpath("coins", "//div/a/div/div/div/div/p")

    def start_parser(self, pages=99):
        data = pd.DataFrame(columns=self.xpath.keys())

        for _ in range(pages):
            self.device.cursor.scroll(-10000)
        
            coins_pages = self.get_element(self.xpath["coins"]["xpath"], text=True, all=True)
            print(f"[INFO] {len(coins_pages)=}")
            for coin in coins_pages:
                data.loc[len(data)] = coin

            if not self.check_loop():
                break

            self.next_page()

        return self.save_data(data) if self.save else data