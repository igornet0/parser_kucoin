import undetected_chromedriver as uc

class WebDriver(uc.Chrome):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def WebOptions(cls, *args, **kwargs):
        options = uc.ChromeOptions(*args, **kwargs)
        # options.add_argument("--disable-blink-features=AutomationControlled")
        # options.add_argument("--window-size=1920,1080")
        return options