from sys import argv
import os
from datetime import datetime, timedelta
import time, keyboard
import multiprocessing as mp

from config import ConfigKucoin
from App_kucoin import Loger, KuCoinAPI, Dataset, Coin

loger = Loger()

Coin.create_table()

def start_parser(coins_list: str, time_parser="5m", pause=60):

    coins_list = Dataset(coins_list, save=False).get_dataset()["coins"]

    api = KuCoinAPI(api_key=ConfigKucoin.api_key, 
                    api_secret=ConfigKucoin.api_secret, 
                    api_passphrase=ConfigKucoin.api_passphrase,
                    save=True, logger=loger)
    
    coins:dict = {coin: None for coin in coins_list}

    loger["INFO"](f"Start parser {datetime.now()}")
    loger["INFO"]("For stop press 'q'")
    loger["INFO"]("Time: " + time_parser)
    loger["INFO"](f"Count coins: {len(coins_list)}")

    count_cpu = mp.cpu_count()
    byffer_process = {}
    pause_loop = 0

    while True:

        if keyboard.is_pressed("q"):
            break
        elif pause_loop:
            time.sleep(1)
            pause_loop -= 1
            continue

        for coin, _ in filter(lambda x: x[1] is None or datetime.now() - x[1] > timedelta(minutes=pause), coins.items()):
            if len(byffer_process) >= count_cpu:
                break
            
            if coin not in byffer_process:
                byffer_process[coin] = mp.Process(target=api.get_kline, 
                                                  args=(coin, time_parser))
                byffer_process[coin].start()
                coins[coin] = datetime.now()

        if len(byffer_process) == 0:
            loger["INFO"](f"All processes are complete, wait {pause*60} minute")
            pause_loop = pause*60
            continue
        
        for process in byffer_process.values():
            process.join()
        
        byffer_process = {}

if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: python start_parser.py <time> <pause>")
        exit(0)

    time_parser = argv[1]
    pause = argv[2] if len(argv) > 2 else 60
    coins_list = os.path.join(os.getcwd(), "coins_list.csv")

    start_parser(coins_list, time_parser, pause)

