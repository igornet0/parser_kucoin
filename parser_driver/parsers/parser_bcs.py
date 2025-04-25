from .parser_kucoin import Parser_kucoin

class Parser_bcs(Parser_kucoin):

    def __init__(self, tick = 1, save = False, path_save="datasets", DEBUG=False, 
                 xpath_default=["login", "password", "click_login", "frame", "filename", "timetravel"]):
        super().__init__(tick, save, path_save, DEBUG, xpath_default)

        self.login = None
        self.password = None


    def login_xpath(self):
        self.add_xpath("login", "/html/body/div[1]/div[3]/div/div[2]/div/div[2]/div/form/div[1]/div/div/input")
        self.add_xpath("password", "/html/body/div[1]/div[3]/div/div[2]/div/div[2]/div/form/div[2]/div/div/input")
        self.add_xpath("click_login", "/html/body/div[1]/div[3]/div/div[2]/div/div[2]/div/form/div[3]/button[1]")

    
    def default_xpath(self):
        self.add_xpath("frame", "/html/body/div[1]/div[3]/div[3]/div[2]/div/div/div[3]/div/div[1]/div[2]/div/div/div[2]/div/div/div[2]/iframe")
        self.add_xpath("filename", "/html/body/div[1]/div[3]/div[3]/div[2]/div/div/div[3]/div/div/div[2]/div/div/div[1]/div[1]/div/div/div/div/div[2]/label[1]/span/div/span")

        self.add_xpath("timetravel", "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[1]/td[2]/div/div[2]/div[1]/div/div[1]/div[1]/div[3]")
        self.add_xpath("datetime", "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[4]/td[2]/div/canvas[2]")
        self.add_xpath("open", "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[1]/td[2]/div/div[2]/div[1]/div/div[2]/div/div[2]/div[2]")
        self.add_xpath("max",  "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[1]/td[2]/div/div[2]/div[1]/div/div[2]/div/div[3]/div[2]")
        self.add_xpath("min", "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[1]/td[2]/div/div[2]/div[1]/div/div[2]/div/div[4]/div[2]")
        self.add_xpath("close", "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[1]/td[2]/div/div[2]/div[1]/div/div[2]/div/div[5]/div[2]")
        # self.add_xpath("volume", "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[1]/td[2]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]/div")
        self.add_xpath("volume", "/html/body/div[2]/div[1]/div[2]/div[1]/div[2]/table/tr[3]/td[2]/div/div[2]/div/div[2]/div[2]/div[2]/div/div[1]/div")