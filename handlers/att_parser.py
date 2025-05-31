import asyncio
from datetime import datetime, timedelta
from typing import List, Any, Callable
from collections import namedtuple
import pandas as pd
import multiprocessing as mp
from aioconsole import ainput
from functools import wraps
import copy
import os

from core.utils.gui_deps import GUICheck

if GUICheck.has_gui_deps():
    from parser_driver import ParserKucoin 
else:
    class ParserKucoin: pass


from parser_driver import ParserApi, KuCoinAPI
from core.models import Dataset, DatasetTimeseries
from core.database.orm_query import (orm_add_data_timeseries, orm_add_timeseries,
                                     orm_get_timeseries_by_coin, orm_update_coin_price)
from core.utils import AutoDecorator
from core.utils.tesseract_img_text import image_to_text
from core import data_manager, Database

import logging

logger = logging.getLogger("parser_logger.att")

class AttParser:

    def __init__(self, api: ParserApi, pause: int = 60, clear: bool = False) -> None:
        
        self.coin_list: List[str] = data_manager.coin_list[::-1]
        self.driver_lock = False

        self.pause = pause

        # Autoclear
        self.flag_clear = clear
        self.buffer_data = []
        self.autodecorator = AutoDecorator(self)

        self.flag_save = False
        self.save_type = "raw"

        self.path_save = None

        self.api = api
        self.db = None

    def init_db(self, db: Database):
        self.db = db
    
    async def parse(self, count: int = 10, miss: bool = False,
                    last_launch: bool = False, time_parser="5m", 
                    save: bool = False, save_type: str = "raw",
                    *input_args) -> dict[str, pd.DataFrame]:

        coins_list = list(map(lambda x: x.strip(), self.coin_list))
        
        if not last_launch:
            self.path_save = None
        else:
            self.path_save = data_manager.get_last_launch()

        if isinstance(self.api, KuCoinAPI):
            if miss:
                raise NotImplementedError("Missed data not implemented for KuCoinAPI")
            
            func_parser = self.api.get_kline
            check_stop = self._check_stop_parser
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

        if save:
            self.flag_save = True
            self.save_type = save_type

        data: dict[str, Dataset] = await self.start_parser(func_parser,
                                                    check_stop,
                                                coins_list, count, 
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
            output_args = output_args if isinstance(output_args, tuple) else (output_args, )
            result = await func(*input_args)
            logger.debug(f"Result {func.__name__} {result}")
            queue.put((*output_args, result))

        except Exception as e:
            logger.error(f"Error processing {func.__name__} {e}")
            queue.put(None)

        return queue
    
    def get_timesireas_data_genersater(self, dataset: dict):
        dataset_dict = copy.deepcopy(dataset)
        while dataset_dict:
            pop_list = []
            new_data = {}
            for key in dataset_dict.keys():
                for i, data in dataset_dict[key].items():
                    if key == "datetime":
                         data = str(data)

                    new_data[key] = data
                    pop_list.append(i)
                    break
            
            for key in dataset_dict.keys():
                for i in pop_list:
                    dataset_dict[key].pop(i)
                    break
            
                if not dataset_dict[key]:
                    dataset_dict = {}
                    break

            yield new_data

    async def dataset_clear(self, coin: str, time_parser: str, dataset: Dataset, processed_dir, filename):
        new_dataset: DatasetTimeseries = self.clear_dataset(dataset, coin, time_parser)
        dataset.set_dataset(new_dataset.get_dataset())
        dataset.set_path_save(processed_dir)
        dataset.set_filename(filename)
        dataset.save_dataset()

        if self.db:
            async with self.db.get_session() as session:
                ts = await orm_get_timeseries_by_coin(session, coin, time_parser)
                dataset_dict = dataset.to_dict()
                for data in self.get_timesireas_data_genersater(dataset_dict):
                    if data.get("open") == "x":
                        continue
                    
                    result = await orm_add_data_timeseries(session, ts.id, data_timeseries=data)
                    if not result:
                        break
    
    async def save_data(self, data: dict[str, Dataset] | Dataset, path_type: str = "raw", 
                  coin: str = "coin", time_parser: str = "5m") -> dict[str, Dataset]:
        """Save data to file"""

        self.api.set_save_path(data_manager[path_type])

        if self.path_save is None:
            if path_type == "raw":
                path_save = self.api.create_launch_dir()

            self.path_save = path_save
        else:
            path_save = self.path_save

        if isinstance(data, DatasetTimeseries):
            data = {data.timetravel: data}
        elif isinstance(data, Dataset):

            data = {coin: data}
            
        for coin, dataset in data.items():
        
            path_save_coin = os.path.join(path_save, coin)
            filename = "{coin}_{time}.csv".format(coin=coin, time=time_parser)

            if not self.flag_clear:
                full_path_save_coin = os.path.join(path_save_coin, filename)
            else:
                data_manager.create_dir("processed", coin)
                path_save = data_manager.create_dir("processed", coin + "/" + time_parser)
                full_path_save_coin = os.path.join(path_save, filename)

            if self.db:
                async with self.db.get_session() as session:
                    await orm_add_timeseries(session, coin, time_parser, full_path_save_coin)
        

            dataset.set_path_save(path_save_coin)
            dataset.set_filename(filename)

            if isinstance(self.api, KuCoinAPI):
                dataset_new = dataset.get_dataset().copy()

                if self.db:
                    price_now = float(dataset_new.iloc[0]["close"])
                    async with self.db.get_session() as session:
                        await orm_update_coin_price(session, coin, price_now=price_now)

                dataset_new = dataset_new.drop(dataset_new.index[0])
                dataset.set_dataset(dataset_new)

            dataset.save_dataset()

            logger.info("Save data for coin: %s, count: %d", coin, len(dataset))
            
            if self.flag_clear:
                await self.dataset_clear(coin, time_parser, dataset, path_save, filename)

        return data

    async def start_parser(self, func_parser:Callable, 
                            check_stop: Callable,
                           coins_list: List, count: int = -1, 
                           time_parser: str ="5m", *args) -> List[pd.DataFrame]:
        
        coins = {coin: None for coin in coins_list}
        len_coins = len(coins)
        all_dataframes = {}
        result_queue = mp.Queue()
        count_cpu = mp.cpu_count() if not self.driver_lock else 1
        buffer_processes = {}
        # stop_event = asyncio.Event()
        stop_event = None

        # Запускаем фоновую задачу для проверки ввода
        # input_task = asyncio.create_task(self._check_stop_input(stop_event))

        logger.info("Start parser - %s, count_cpu: %d", datetime.now(), count_cpu)
        logger.info(f"Tracking {len_coins} coins")

        try:
            while check_stop(**{"stop_event": stop_event, 
                              "count": count, "all_dataframes": all_dataframes, 
                              "len_coins": len_coins}):

                # Управление процессами
                await self._manage_processes(
                    func_parser,
                    coins, 
                    buffer_processes,
                    result_queue,
                    count_cpu,
                    *args
                )

                # Сбор результатов
                await self._collect_results(result_queue, all_dataframes, buffer_processes, time_parser)

                await asyncio.sleep(0.1)

        # except Exception as e:
        #     logger.error(f"Parser error - {e}")

        finally:
            # input_task.cancel()
            self._cleanup_processes(buffer_processes)

        logger.info("End parser  - %s", datetime.now())

        return all_dataframes
    
    def _check_stop_parser(self, stop_event, count, all_dataframes, len_coins):
        return len(list(filter(lambda list: len(list)==count, all_dataframes.values()))) != len_coins
    
    def _check_stop(self, stop_event, count, all_dataframes: dict, len_coins):
        return not all_dataframes or \
                len(all_dataframes.keys()) <= len_coins or \
                not all(map(lambda key: (not all_dataframes[key] is None) and len(all_dataframes[key]) >= count, all_dataframes.keys()))

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

    async def _manage_processes(self, func_parser, coins, buffer_processes, 
                                result_queue, count_cpu, *args):
        
        """Управление пулом процессов"""
        for coin, last_time in coins.items():
            if len(buffer_processes) >= count_cpu:
                break
        
            if self._should_process_coin(last_time, self.pause):
                if self.buffer_data:
                    coins_last = self.buffer_data.pop()
                    dataset: DatasetTimeseries = coins_last[coin]

                    logger.info("Start coin: %s with data from buffer %s", coin, len(dataset))
                    
                    if self.driver_lock:
                        last_datetime = dataset.get_dataset()["datetime"].min()
                        args = (*args, last_datetime)
                else:
                    logger.info(f"Start new coin: {coin}")

                await self._start_process_for_coin(
                        func_parser,
                        coin,
                        buffer_processes,
                        result_queue,
                        *args
                    )
                coins[coin] = datetime.now()

    async def _start_process_for_coin(self, func_parser: Callable, coin: str, 
                                buffer_processes, result_queue, *args):
        
        """Запуск процесса для конкретной монеты"""
        if not self.driver_lock:
            p = mp.Process(target=self._wrapper, 
                            args=(func_parser, result_queue, (coin, *args), (coin,)),
                            daemon=True)
            
            logger.debug(f"Start process for coin: {coin}")
            p.start()

        else:
            await self._async_wrapper(func_parser, result_queue, (coin, *args), (coin,))
            p = None

        buffer_processes[coin] = p

    async def _collect_results(self, result_queue, all_dataframes, buffer_processes, time_parser):
        """Сбор результатов из очереди"""
        while not result_queue.empty():
            # coin, *data = result_queue.get()
            data = result_queue.get()

            if data is None:
                continue

            coin = data[0]
            data: Dataset = data[1:]

            if len(data) == 0:
                continue

            if len(data) == 1:
                data = data[0]

            if data is None:
                logger.warning(f"Data for coin {coin} is None")
                del buffer_processes[coin]
                continue

            all_dataframes.setdefault(coin, None)

            if isinstance(data, Dataset) and self.flag_save:
                data = await self.save_data(data, self.save_type, coin, time_parser)
                data = data[coin]

            all_dataframes[coin] = data

            logger.info("Get data for coin: %s, count: %d", coin, len(all_dataframes[coin]))

            del buffer_processes[coin]

    def _cleanup_processes(self, processes):
        """Очистка запущенных процессов"""
        for p in processes.values():
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
        
        return (datetime.now() - last_time) > timedelta(minutes=pause)
    
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
        
        dt = DatasetTimeseries(dataset.get_dataset().copy() if isinstance(dataset, Dataset) else dataset,
                               timetravel=time)
        dt.set_dataset(dt.clear_dataset())
        
        logger.info("Clear dataset for coin: %s, count Nan: %d", coin, len(dt.get_dataset_Nan()))
        return dt