import asyncio
from datetime import datetime, timedelta
from typing import List, Any, Callable, Dict
import pandas as pd
import multiprocessing as mp
from emoji import replace_emoji
import copy
import re
import os

from parser_driver import (ParserApi, KuCoinAPI, ParserNewsApi, 
                           ParserKucoin, TelegramParser)
from core.models import Dataset, DatasetTimeseries
from core.database.orm import *
from core.utils import AutoDecorator
from core.utils.tesseract_img_text import image_to_text
from core import data_manager, Database, telegram_settings

import logging

logger = logging.getLogger("parser_logger.att")

BUFFER_SIZE = 100

class AttParser:

    def __init__(self, api: ParserApi, pause: int = 60, clear: bool = False) -> None:
        
        self.coin_list: List[str] = []
        self.driver_lock = False

        self.pause = pause

        # Autoclear
        self.flag_clear = clear
        self.buffer_data = {}
        self.autodecorator = AutoDecorator(self)

        self.flag_save = False
        self.save_type = "raw"

        self.path_save = None

        self.api = api
        self.db = None

    async def update_coin_list(self, db: Database):
        """Update coin list from database"""
        if not self.db:
            self.db = db
        
        async with self.db.get_session() as session:
            coins = await orm_get_coins(session)
            self.coin_list = list(map(lambda x: x.name, filter(lambda x: x.parsed, coins)))

    async def init_db(self, db: Database):
        self.db = db

    async def _load_last_launch(self, time_parser):
        self.path_save = data_manager.get_last_launch()
        tmp = self.flag_save
        self.flag_save = False
        tasks = {}
        if self.path_save:
            logger.info(f"Last launch: {self.path_save}")
            for coin in data_manager.get_path_from(self.path_save):
                for csv_file in data_manager.get_path_from(coin, filter_path=lambda x: time_parser in str(x.name)):
                    dt = DatasetTimeseries(csv_file, timetravel=time_parser)
                    task = asyncio.create_task(self.add_buffer_dataTimeseries(coin, dt, time_parser))
                    tasks[coin] = task
        if tasks:
            [data for data in await asyncio.gather(*tasks.values())]

        self.flag_save = tmp

    async def _init_kucoin_parsers(self, miss, last_launch, time_parser, count, *input_args):
        
        if not last_launch:
            self.path_save = None
        else:
            await self._load_last_launch(time_parser)
        
        if isinstance(self.api, KuCoinAPI):
            if miss:
                raise NotImplementedError("Missed data not implemented for KuCoinAPI")
            
            # func_parser = self.api.get_kline
            func_parser = self.api.async_parsed_coins
            self.driver_lock = True

            check_stop = self._check_stop
            input_args = ('USDT', time_parser, *input_args)

        elif isinstance(self.api, ParserApi):
            self.api.set_timetravel(time_parser)
            if miss and isinstance(self.api, ParserKucoin):
                func_parser = self.api.parser_missing_data
            else:
                func_parser = self.api.start_parser

            check_stop = self._check_stop
            input_args = (count, )

            self.driver_lock = True

        return func_parser, check_stop, input_args

    async def _init_news_parsers(self, *input_args):

        def clear_text(text):
            for word in telegram_settings.pop_words:
                text = text.replace(word, '')
            
            text = replace_emoji(text, replace="")
            text = re.sub(r'[\n\r\t]+', ' ', text)
            text = re.sub(r'\s{2,}', ' ', text)

            return text.strip()
        
        self.api.set_clear_text(clear_text)

        if isinstance(self.api, ParserNewsApi):
            
            func_parser = self.api.get_last_news
            self.api.init_db(self.db)
            check_stop = self._check_stop

            self.driver_lock = True
            input_args = (100, )

        elif isinstance(self.api, TelegramParser):

            escaped = sorted(map(re.escape, telegram_settings.words), key=len, reverse=True)
            pattern = r'(?<!\w)(?:' + '|'.join(escaped) + r')(?!\w)'
            regex = re.compile(pattern, flags=re.IGNORECASE)
            async with self.db.get_session() as session:
                channels = await orm_get_telegram_channels(session, parsed=True)

            def filter_event(event):
                if not self.db:
                    return True
                
                return any(filter(lambda chanel: chanel.name == event.chat.title, channels))

            self.api.set_filter(lambda event: filter_event(event))

            self.api.init_db(self.db)
            # tg_parser.start_parser_event({tg_parser.event_parsing_telegram_channel_pattern: {"pattern": regex}})
            func_parser = self.api.start_parser_event
            # func_parser = self.api.get_last_news
            check_stop = self._check_stop
            self.driver_lock = True
            input_args = ({self.api.event_parsing_telegram_channel_pattern: {"pattern": regex}}, )

        return func_parser, check_stop, input_args

    async def parse(self, count: int = 10, miss: bool = False,
                    last_launch: bool = False, time_parser="5m", 
                    save: bool = False, save_type: str = "raw",
                    *input_args) -> dict[str, pd.DataFrame]:

        if isinstance(self.api, KuCoinAPI) or isinstance(self.api, ParserKucoin):
            func_parser, check_stop, input_args = await self._init_kucoin_parsers(miss, last_launch, time_parser, count, *input_args)
        
        elif isinstance(self.api, ParserNewsApi) or isinstance(self.api, TelegramParser):
            func_parser, check_stop, input_args = await self._init_news_parsers(*input_args)

        if save:
            self.flag_save = True
            self.save_type = save_type

        data: dict[str, Dataset] = await self.start_parser(func_parser, check_stop, count, 
                                                            time_parser,
                                                            *input_args)    

        return data
    
    def filter_coins(self, data_coin: dict[str, Any], filter_value: Callable[[Any], bool]):
        for coin, value in data_coin.items():
            if filter_value(value):
                yield coin

    @staticmethod
    def _wrapper(func: Callable, queue: mp.Queue, input_args: tuple, output_args: tuple = None):
        """Wrapper to get data and put result in queue"""
        try:
            output_args = output_args if isinstance(output_args, tuple) else (output_args, )
            queue.put((*output_args, func(*input_args)))
        except Exception as e:
            logger.error(f"Error processing {func.__name__} {e}")
            queue.put(None)

        return queue
    
    @staticmethod
    async def _async_wrapper(func: Callable, queue: mp.Queue, input_args: tuple, output_args: tuple = None):
        """Wrapper to get data and put result in queue"""
        try:
            if output_args:
                output_args = output_args if isinstance(output_args, tuple) else (output_args, )

            result = await func(*input_args)
            # logger.debug(f"Result {func.__name__} {result}")
            queue.put((*output_args, result) if output_args is not None else result)
        except Exception as e:
            logger.error(f"Error processing {func.__name__} {e}")
            queue.put(None)

        return queue

    async def dataset_clear(self, coin: str, time_parser: str, dataset: Dataset, processed_dir, filename):
        
        new_dataset = self.clear_dataset(dataset, coin, time_parser)
        
        dataset.set_dataset(new_dataset.get_dataset())
        dataset.set_path_save(processed_dir)
        dataset.set_filename(filename)

        await self.update_db_timeseries_path(coin, dataset, time_parser)

        dataset.save_dataset()

        logger.debug("Save clear data for coin: %s, count: %d in %s", coin, len(dataset), dataset.get_path_save())
    
    async def save_data(self, dataset: DatasetTimeseries, path_type: str = "raw", 
                  coin: str = "coin", time_parser: str = "5m") -> dict[str, Dataset]:
        """Save data to file"""

        self.api.set_save_path(data_manager[path_type])

        if self.path_save is None:
            if path_type == "raw":
                path_save = self.api.create_launch_dir()

            self.path_save = path_save
        else:
            path_save = self.path_save
            
        filename = "{coin}_{time}.csv".format(coin=coin, time=time_parser)

        if not self.flag_clear:
            path_save_coin = os.path.join(path_save, coin)
            dataset.set_path_save(path_save_coin)
            dataset.set_filename(filename)

            dataset.save_dataset()

            await self.update_db_timeseries_path(coin, dataset, path_save_coin)

            logger.info("Save data for coin: %s, count: %d in %s", coin, len(dataset), dataset.get_path_save())
        else:
            path_save = data_manager["processed"]
            path_save = data_manager.create_dir("processed", coin)
            path_save = data_manager.create_dir("processed", coin + "/" + time_parser)
            await self.dataset_clear(coin, time_parser, dataset, path_save, filename)

        return dataset

    async def update_db_timeseries(self, coin: str, dataset: DatasetTimeseries, time_parser: str):
        if self.db:    
            async with self.db.get_session() as session:
                ts = await orm_get_timeseries_by_coin(session, coin, time_parser)

                if not ts:
                    dataset.set_path_save(data_manager["processed"] / coin )
                    ts = await orm_add_timeseries(session, coin=coin, timestamp=time_parser,
                                                  path_dataset=str(dataset.get_path_save()))

                for data in dataset:

                    data = {
                        "datetime": data.get("datetime"),
                        "open": float(data.get("open")),
                        "close": float(data.get("close")),
                        "max": float(data.get("max")),
                        "min": float(data.get("min")),
                        "volume": float(data.get("volume"))
                    }
                    
                    if not await orm_add_data_timeseries(session, ts.id, data_timeseries=data):
                        break

    async def update_db_timeseries_path(self, coin: str, dataset: DatasetTimeseries, time_parser: str):
        if self.db and self.flag_save:    
            async with self.db.get_session() as session:
                await orm_add_timeseries(session, coin, time_parser, str(dataset.get_path_save()))
    
    async def update_db_last_price(self, coin: str, dataset: DatasetTimeseries):
        if self.db:
            async with self.db.get_session() as session:
                data = dataset.get_last_row()

                if isinstance(data["close"].item(), str) and data["close"].item() == "x":
                    return False
                
                price_data = PriceData(
                    price_now=float(data["close"].item()),
                    max_price_now=float(data["max"].item()),
                    min_price_now=float(data["min"].item()),
                    open_price_now=float(data["open"].item()),
                    volume_now=float(data["volume"].item())
                )

                await orm_update_coin_price(session, coin, price_data)

                return True

    async def add_byffer_data(self, data):
        self.buffer_data[datetime.now()] = data

        if len(self.buffer_data) > BUFFER_SIZE:
            self.buffer_data.pop(list(self.buffer_data.keys())[0])

    async def add_buffer_dataTimeseries(self, coin: str, data: DatasetTimeseries, time_parser: str = "5m"):

        self.buffer_data.setdefault(coin, {})
        self.buffer_data[coin].setdefault(time_parser, None)

        if data is None:
            return

        await self.update_db_last_price(coin, data)
        # logger.debug(f"{coin} - {data.get_datetime_last()=}")
        data_pd = data.get_dataset()
        data_pd = data_pd[data_pd["datetime"] < data.get_datetime_last()]
        data.set_dataset(data_pd)

        if self.buffer_data[coin][time_parser] is None:
            self.buffer_data[coin][time_parser] = data
            # logger.debug(f"{coin} - {data.get_datetime_last()=}")
        else:
            self.buffer_data[coin][time_parser].append(data)
            # self.buffer_data[coin][time_parser] = DatasetTimeseries(DatasetTimeseries.concat_dataset(self.buffer_data[coin][time_parser], data))

        await self.update_db_timeseries(coin, self.buffer_data[coin][time_parser], time_parser)

        if len(self.buffer_data[coin][time_parser]) >= BUFFER_SIZE:
            if self.flag_save:
                await self.save_data(self.buffer_data[coin][time_parser], self.save_type, coin, time_parser)
            
            self.buffer_data[coin][time_parser].pop_last_row(BUFFER_SIZE // 2)


        # self.buffer_data.setdefault(coin, {})
        # self.buffer_data[coin].setdefault(time_parser, None)

        # if self.buffer_data[coin][time_parser] is None:
        #     self.buffer_data[coin][time_parser] = data
        # else:
        #     if data.get_datetime_last() - self.buffer_data[coin][time_parser].get_datetime_last() < timedelta(minutes=5):
        #         return
            
        #     self.buffer_data[coin][time_parser] = DatasetTimeseries(DatasetTimeseries.concat_dataset(self.buffer_data[coin][time_parser], data))

        # await self.update_db_timeseries(coin, self.buffer_data[coin][time_parser], time_parser)

        # # logger.debug("Add data to buffer for coin: %s, count: %d", coin, len(self.buffer_data[coin][time_parser]))

        # if len(self.buffer_data[coin][time_parser]) >= BUFFER_SIZE:

        #     if self.flag_save:
        #         await self.save_data(self.buffer_data[coin][time_parser], self.save_type, coin, time_parser)
            
        #     self.buffer_data[coin][time_parser].pop_last_row(BUFFER_SIZE // 2)

    def get_data_from_buffer(self, coin, time_parser):
        return self.buffer_data.get(coin, {}).get(time_parser, None)
    
    def get_buffer_data(self):
        return self.buffer_data
    
    async def update_coins(self, coins: Dict[str, datetime]):
        await self.update_coin_list(self.db)

        pop_list = []
        coins_list = copy.deepcopy(self.coin_list)
        
        for coin in coins.keys():
            if coin not in coins_list:
                pop_list.append(coin)
            else:
                coins_list.remove(coin)

        for coin in pop_list:
            coins.pop(coin)

        if coins_list:
            for coin in coins_list:
                coins[coin] = None

        return coins
    
    async def start_manager_kucoin(self, check_stop, func_parser, time_parser, count, *args):
        coins = {}
        all_dataframes = {}
        result_queue = mp.Queue()
        count_cpu = mp.cpu_count() if not self.driver_lock else 1
        buffer_processes = {}
        tasks = {}
        # stop_event = asyncio.Event()
        stop_event = None

        # Запускаем фоновую задачу для проверки ввода
        # input_task = asyncio.create_task(self._check_stop_input(stop_event))

        try:
            while check_stop(**{"stop_event": stop_event, 
                              "count": count, "all_dataframes": all_dataframes}):

                # Управление процессами
                all_dataframes = await self._manage_processes_kucoin(
                    func_parser,
                    buffer_processes,
                    coins,
                    result_queue,
                    time_parser,
                    count_cpu,
                    *args,
                )

                for coin, data in all_dataframes.items():
                    data: DatasetTimeseries
                    # data.sort(ascending=True)
                    # data.pop_last_row(-1)
                    # data.sort()
                    task = asyncio.create_task(self.add_buffer_dataTimeseries(coin, data, time_parser))
                    tasks[coin] = task

                # all_dataframes = {}

                [_ for _ in await asyncio.gather(*tasks.values())]

                tasks = {}

                await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Error in parser: {e}")

        finally:
            # input_task.cancel()
            self._cleanup_processes(buffer_processes)

    async def _manage_processes_telegram(self, func_parser, *args):
        result_queue = mp.Queue()

        logger.info(f"Start parser - {args} {datetime.now()}")

        # asyncio.run(func_parser(args[0]))
        await self._async_wrapper(func_parser, result_queue, (*args,))
        logger.info(self.api.buffer_messages)
        logger.info("End parser - %s", datetime.now())

    async def start_manager_telegram(self, check_stop, func_parser, count, *args):
        logger.debug(f"start_manager_telegram {args=}")
        try:
            while check_stop(**{"count": count, }):

                # Управление процессами
                await self._manage_processes_telegram(
                    func_parser,
                    *args
                )

                await asyncio.sleep(self.pause * 60)

        except Exception as e:
            logger.error(f"Error in parser: {e}")

    async def _manage_processes_news(self, func_parser, *args):
        result_queue = mp.Queue()

        logger.info(f"Start parser - {args} {datetime.now()}")

        # asyncio.run(func_parser(args[0]))
        if self.get_buffer_data():
            last_publish = min(self.get_buffer_data(), key=lambda k: self.get_buffer_data()[k].date)
            input_data = (*args, last_publish)
        else:
            input_data = (*args,)

        await self._async_wrapper(func_parser, result_queue, input_data)

        news = await self._collect_results(result_queue)

        logger.info("End parser - %s", datetime.now())

        return news
    
    def add_buffer_dataNews(self, data: NewsData):
        self.buffer_data[data.date] = data

        if len(self.buffer_data) >= BUFFER_SIZE:
            self.buffer_data.pop(list(self.buffer_data.keys())[0])

    async def start_manager_news(self, check_stop, func_parser, count, *args):
        logger.debug(f"start_manager_news {args=}")

        try:
            while check_stop(**{"count": count, }):

                # Управление процессами
                news = await self._manage_processes_news(
                    func_parser,
                    *args
                )

                for _, new in news.items():
                    await self.add_buffer_dataNews(new)

                await asyncio.sleep(self.pause * 60)

        except Exception as e:
            logger.error(f"Error in parser: {e}")

    async def start_parser(self, func_parser:Callable, 
                            check_stop: Callable, count: int = -1, 
                           time_parser: str ="5m", *args) -> List[pd.DataFrame]:

        if isinstance(self.api, ParserKucoin) or isinstance(self.api, KuCoinAPI):
            print(args)
            await self.start_manager_kucoin(check_stop, func_parser, time_parser, 
                                            count, *args)
            
        elif isinstance(self.api, TelegramParser):
            await self.start_manager_telegram(check_stop, func_parser, count, *args)
        elif isinstance(self.api, ParserNewsApi):
            await self.start_manager_news(check_stop, func_parser, count, *args)
        else:
            raise Exception("Unknown api")

        logger.info("End parser  - %s", datetime.now())
    
    def _check_stop_parser(self, stop_event, count, all_dataframes, len_coins):
        return len(list(filter(lambda list: len(list)==count, all_dataframes.values()))) != len_coins
    
    # def _check_stop(self, stop_event, count, all_dataframes: dict, len_coins):
    #     return not all_dataframes or \
    #             len(all_dataframes.keys()) <= len_coins or \
    #             not all(map(lambda key: (not all_dataframes[key] is None) and len(all_dataframes[key]) >= count, all_dataframes.keys()))
    def _check_stop(self, *args, **kwargs):
        return True
    
    # async def _check_stop_input(self, stop_event: asyncio.Event):
    #     """Асинхронная проверка ввода 'q' для остановки"""
    #     while True:
    #         if await ainput("") == 's':
    #             stop_event.set()
    #             break

    #         elif await ainput("") == 'p':
    #             while True:
    #                 if await ainput("") == 'p':
    #                     stop_event.set()
    #                     break
    #                 await asyncio.sleep(0.1)

    #         await asyncio.sleep(0.1)
        
    async def _sheduler_processes(self, buffer_processes: Dict[str, mp.Process]):
        pop_list = []
        # tasks = {}

        for coin, process_data in filter(lambda items: items[1]["status"], buffer_processes.items()):
            # task = asyncio.create_task(self.async_join(process_data["process"]))
            # tasks[coin] = task
            if process_data["process"].is_alive():
                logger.debug(f"Process {coin} is alive")
                process_data["process"].join(timeout=30)

            pop_list.append(coin)

        # [_ for _ in await asyncio.gather(*tasks.values())]
        logger.debug("End Join processed")

        for coin in pop_list:
            if buffer_processes[coin]["process"].is_alive():
                buffer_processes[coin]["process"].terminate()

            buffer_processes[coin]["process"].close()
            del buffer_processes[coin]

    async def _init_manager_processes(self, func_parser, coins, buffer_processes, 
                                result_queue, time_parser, count_cpu, *args):
        
        logger.info("Start parser - %s, count_cpu: %d", datetime.now(), count_cpu)
        logger.info(f"Tracking {len(coins)} coins")

        for coin, last_time in coins.items():
            if len([*filter(lambda process: process["status"], buffer_processes.values())]) >= count_cpu:
                logger.debug(f"Yield Sheduler processes {buffer_processes=}")
                yield buffer_processes
                logger.debug(f"Sheduler processes {buffer_processes=}")
        
            if self._should_process_coin(last_time, self.pause):
                if self.get_data_from_buffer(coin, time_parser):
                    dataset: DatasetTimeseries = self.get_data_from_buffer(coin, time_parser)
                    last_datetime = dataset.get_datetime_last() - timedelta(minutes=self.pause % 5 * 5 + self.pause)
                    logger.debug("Start coin: %s with data from buffer %s last_datetime: %s", coin, len(dataset), last_datetime)
                else:
                    logger.debug(f"Start new coin: {coin}")
                    last_datetime = None

                await self._start_process_for_coin(
                        func_parser,
                        coin,
                        buffer_processes,
                        result_queue,
                        last_datetime,
                        *args,
                    )
                
                coins[coin] = datetime.now()

        logger.info("End parser - %s", datetime.now())

    async def _init_manager_processes_v2(self, func_parser, coins: Dict[str, datetime],
                                result_queue, time_parser, count_cpu, *args):
        
        logger.info("Start parser - %s, count_cpu: %d", datetime.now(), count_cpu)
        
        if isinstance(self.api, KuCoinAPI) and self.driver_lock:

            logger.debug("Start coins")
            
            coin_last_datetimes = {coin: self.get_data_from_buffer(coin, time_parser).get_datetime_last() if self.get_data_from_buffer(coin, time_parser) else None 
                              for coin in filter(lambda x: self._should_process_coin(coins[x], self.pause), coins.keys())}
            
            if not coin_last_datetimes:
                logger.info("No coins to process - %s", datetime.now())
                return False

            await self._start_process_for_coins(
                        func_parser,
                        coin_last_datetimes,
                        result_queue,
                        *args,
                    )
            
            logger.debug("End coins full")

            for coin in coins.keys():
                coins[coin] = datetime.now()

        elif isinstance(self.api, ParserNewsApi) and self.driver_lock:
            logger.debug("Start news")
            coin_last_datetimes = {coin: self.get_data_from_buffer(coin, time_parser).get_datetime_last() if self.get_data_from_buffer(coin, time_parser) else None 
                              for coin in filter(lambda x: self._should_process_coin(coins[x], self.pause), coins.keys())}
            
            if not coin_last_datetimes:
                logger.info("No coins to process - %s", datetime.now())
                return False

            await self._start_process_for_coins(
                        func_parser,
                        coin_last_datetimes,
                        result_queue,
                        *args
                    )
        
        elif isinstance(self.api, TelegramParser) and self.driver_lock:
            pass

        logger.info("End parser - %s", datetime.now())
        return True

    async def _manage_processes_kucoin(self, func_parser, buffer_processes: dict, coins: dict,
                                result_queue, time_parser, count_cpu, *args):
        
        if self.driver_lock:
            coins = await self.update_coins(coins)
            if not await self._init_manager_processes_v2(func_parser, coins, result_queue, 
                                                            time_parser, count_cpu, *args):
                buffer_processes.clear()
                return {}
                
        else:
            manager = self._init_manager_processes(func_parser, coins, buffer_processes, 
                                    result_queue, time_parser, count_cpu, *args)
            
            async for buffer_processes in manager:
                await self._sheduler_processes(buffer_processes)
        
        result = await self._collect_results(result_queue)

        return result

    def create_process(self, target, args, deamon=True) -> mp.Process:
        p = mp.Process(target=target, 
                            args=args,
                            daemon=deamon)
            
        return p
    
    async def _start_process_for_coins(self, func_parser: Callable, coins: Dict[str, datetime], result_queue, *args):
        
        """Запуск процесса для конкретной монеты"""
        if not self.driver_lock:
            raise NotImplementedError("This method is used for multiple coins, but driver_lock is True")
        else:
            await self._async_wrapper(func_parser, result_queue, (coins, *args))

    async def _start_process_for_coin(self, func_parser: Callable, coin: str, 
                                buffer_processes, result_queue, last_datetime, *args):
        
        """Запуск процесса для конкретной монеты"""
        if not self.driver_lock:
            p = self.create_process(self._wrapper, (func_parser, result_queue, (coin, *args, last_datetime), (coin,)))
            logger.debug(f"Start process for coin: {coin}")
            p.start()
        else:
            await self._async_wrapper(func_parser, result_queue, (coin, *args, last_datetime), (coin,))
            p = None

        buffer_processes[coin] = {"process": p, "status": True}

    async def _collect_kucoin(self, data):
        all_dataframes = {}

        if data is None:
            return
        elif isinstance(data, list):
            for coin, dt in data:
                if dt is None:
                    logger.warning(f"Data for coin {coin} is None")
                    continue

                all_dataframes[coin] = dt
        else:
            coin = data[0]
            data = data[1:]

            if len(data) == 0:
                return
            elif len(data) == 1:
                data = data[0]

            if data is None:
                logger.warning(f"Data for coin {coin} is None")
            else:
                all_dataframes[coin] = data
            
        return all_dataframes

    async def _collect_results(self, result_queue):
        """Сбор результатов из очереди"""
        all_dataframes = {}

        while not result_queue.empty():
            # coin, *data = result_queue.get()
            data = result_queue.get()

            if isinstance(self.api, KuCoinAPI) or isinstance(self.api, ParserKucoin):
                collect_dataframes = await self._collect_kucoin(data)
                if not collect_dataframes:
                    continue
                else:
                    all_dataframes.update(collect_dataframes)
            elif isinstance(self.api, ParserNewsApi):
                for news in data:
                    all_dataframes[news.id_url] = news
            else:
                raise Exception("Unknown api")
        
        return all_dataframes

    def _cleanup_processes(self, processes):
        """Очистка запущенных процессов"""
        for p in processes.values():

            if isinstance(p, dict):
                p = p["process"]

            if not p is None and p.is_alive():
                p.terminate()

            if not p is None:
                p.join()

    @staticmethod
    def _should_process_coin(last_time, pause):
        """Определяет нужно ли обрабатывать монету"""        
        if last_time is None:
            return True
        elif pause <= 0:
            return False
        
        return (datetime.now() - last_time) - timedelta(minutes=pause) > timedelta(seconds=10)
    
    async def parser_img(self, count: int = 10, time_parser="5m"):
        if isinstance(self.api, ParserKucoin):
            coin = self.coin_list[0]
            await self.api.init(coin)

            for _ in range(count):
                img = await self.api.get_element_datetime(get_img=True)
                name = image_to_text(img)
                data_manager.save_img(img, time_parser, name)

                self.api.device.cursor.move("left")
                await asyncio.sleep(0.2)

    def clear_dataset(self, dataset: Dataset | pd.DataFrame, coin: str = None, time: str = "5m"):
        
        dt = DatasetTimeseries(dataset.get_dataset() if isinstance(dataset, Dataset) else dataset,
                               timetravel=time)
        dt.set_dataset(dt.clear_dataset())
        
        logger.debug("Clear dataset for coin: %s, count Nan: %d", coin, len(dt.get_dataset_Nan()))
        return dt