import sys
from colorama import Fore, Style, init

init(autoreset=True)

class Loger:

    def __init__(self, file=None):
        self.file = file
        self.args = "INFO"
        self._off = False

    @property
    def off(self):
        self._off = True
        return self

    @property
    def on(self):
        self._off = False
        return self
    
    def write(self, message):
        if self.file is None:
            sys.stdout.write(message)
        else:
            with open(self.file, "a") as f:
                f.write(f"{message}\n")

    def flush(self):
        if self.file is None:
            sys.stdout.flush()
        else:
            with open(self.file, "a") as f:
                f.flush()

    def log(self, message, type="INFO", reset=False):
        if type == "INFO":
            message = f"{Style.BRIGHT}{Fore.GREEN}{type}: {Style.RESET_ALL}{message}"
        elif type == "ERROR":
            message = f"{Style.BRIGHT}{Fore.RED}{type}: {Style.RESET_ALL}{message}"
        elif type == "WARNING":
            message = f"{Style.BRIGHT}{Fore.YELLOW}{type}: {Style.RESET_ALL}{message}"
        else:
            message = f"{Style.BRIGHT}{Fore.BLUE}{type}: {Style.RESET_ALL}{message}"
        
        if reset:
            message = f"\r{message}"
        else:
            message = f"\n{message}"
            
        self.write(message)
        self.flush()

    def __call__(self, message, reset=False):
        if self._off:
            return 
        return self.log(message, type=self.args, reset=reset)

    def __getitem__(self, item):
        self.args = item
        return self