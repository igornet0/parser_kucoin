from __future__ import annotations
from core.utils.gui_deps import GUICheck

if GUICheck.has_gui_deps():
    from sktime import utils
    from matplotlib import pyplot as plt
else:
    class plt: pass
    class utils: pass

import pandas as pd
from datetime import datetime
from pathlib import PosixPath
from typing import Union
from os import walk, mkdir, path, getcwd
import re

from core import data_manager
from core.utils.clear_datasets import *
from core.utils.tesseract_img_text import RU_EN_timetravel

import logging

logger = logging.getLogger("Dataset")

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {execution_time}")
        return result
    
    return wrapper

class Dataset:

    def __init__(self, dataset: Union[pd.DataFrame, dict, str], transforms=None, target_column: str=None) -> None:
        
        if isinstance(dataset, str) or isinstance(dataset, PosixPath):
            path_open = self.searh_path_dateset(dataset)

            if isinstance(path_open, list):
                raise FileNotFoundError(f"File {dataset} not found in {getcwd()}")
    
            dataset = pd.read_csv(path_open)
            self.set_filename(str(path_open).split("/")[-1])

        elif not isinstance(dataset, pd.DataFrame):
            logger.error(f"Invalid dataset type {type(dataset)}")
            self.set_filename("clear_dataset.csv")
        
        self.drop_unnamed(dataset)

        if "date" in dataset.columns:
            dataset.rename(columns={"date": "datetime"}, inplace=True)

        if "datetime" in dataset.columns:
            dataset["datetime"] = pd.to_datetime(dataset["datetime"], format='%Y-%m-%d %H:%M:%S')

        self.dataset = dataset
        self.transforms = transforms

        if target_column:
            self.targets = dataset[target_column]
            self.dataset.drop(target_column, axis=1, inplace=True)
        else:
            self.targets = None

        self.path_save = data_manager["processed"]

    def get_datetime_last(self) -> datetime:
        return self.dataset['datetime'].iloc[-1]
    
    def to_dict(self):
        return self.dataset.to_dict()
    
    def set_path_save(self, path_save: str) -> None:
        self.path_save = path_save

    def set_filename(self, file_name: str) -> None:
        self.file_name = file_name

    def get_filename(self) -> str:
        return self.file_name

    def get_dataset(self) -> pd.DataFrame:
        return self.dataset.copy()
    
    def set_dataset(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset.copy()
    
    def get_data(self, idx: int):
        return self.dataset.iloc[idx]
    
    @timer
    def clear_dataset(self) -> pd.DataFrame:
        return clear_dataset(self.dataset)

    @classmethod
    def drop_unnamed(cls, dataset):
        try:
            dataset.drop('Unnamed: 0', axis=1, inplace=True)
        except Exception:
            pass

    @classmethod
    def searh_path_dateset(cls, pattern: str, root_dir=getcwd()) -> list[str]:
        # Преобразуем шаблон в регулярное выражение
        if path.exists(pattern) and path.isfile(pattern):
            return pattern

        regex_pattern = '^' + '.*'.join(re.escape(part) for part in pattern.split('*')) + '$'
        regex = re.compile(regex_pattern)
        
        matched_files = []
        
        for dirpath, _, filenames in walk(root_dir):
            for filename in filenames:
                if regex.match(filename):
                    full_path = path.join(dirpath, filename)
                    matched_files.append(full_path)

        if not matched_files:
            raise FileNotFoundError(f"File {pattern} not found in {root_dir}")
        
        return matched_files

    @classmethod
    def concat_dataset(
        cls, 
        *dataset: pd.DataFrame | Dataset
    ) -> pd.DataFrame:
        """
        Объединяет DataFrame, добавляя отсутствующие строки из последующих датафреймов.
        
        :param dataset: Произвольное количество DataFrame или Dataset объектов
        :return: Итоговый объединенный DataFrame
        """
        dataset = filter(lambda x: isinstance(x, pd.DataFrame) or isinstance(x, Dataset), dataset)
        dataset = list(map(lambda x: x.get_dataset() if isinstance(x, Dataset) else x, dataset))
        result = pd.concat(dataset, ignore_index=True)
        # dublicates = result.duplicated(subset=['datetime', "open"], keep=False)
        # dublicates = result[dublicates]
        
        result.drop_duplicates(subset=['datetime', "open"], inplace=True)
        result.drop_duplicates(subset=['datetime'], inplace=True)

        return result
    
    def save_dataset(self, name_file: str = None) -> None:
        if not path.exists(self.path_save):
            mkdir(self.path_save)

        if name_file is None:
            name_file = self.file_name

        if path.exists(path.join(self.path_save, name_file)):
            dataset = Dataset(path.join(self.path_save, name_file))
            self.dataset = self.concat_dataset(self.get_dataset(), dataset)

        self.dataset.to_csv(path.join(self.path_save, name_file), index=False, encoding='utf-8')
        logger.info(f"Dataset saved to {path.join(self.path_save, name_file)}")

    def __iter__(self):
        for index, data in self.dataset.iterrows():
            yield data

    def __getitem__(self, idx: int):
            
        sample = self.get_data(idx)
        
        if self.transforms:
            sample = self.transforms(sample)

        # if self.targets:
        #     target = self.targets.iloc[idx]
        #     target = torch.tensor(target, dtype=torch.long)  
        #     return sample, target

        return sample, self.targets

    def __len__(self):
        return len(self.dataset)


class DatasetTimeseries(Dataset):
    
    def __init__(self, dataset: Union[pd.DataFrame, dict, str] , timetravel: str = "5m") -> None:
        
        super().__init__(dataset)

        if "datetime" not in self.dataset.columns and "date" in self.dataset.columns:
            self.dataset.rename(columns={"date": "datetime"}, inplace=True)

        elif "datetime" not in self.dataset.columns and "date" not in self.dataset.columns:
            raise ValueError("Columns 'datetime' or 'date' not found in dataset")
        elif "open" not in self.dataset.columns:
            raise ValueError("Column 'open' not found in dataset")
        elif "close" not in self.dataset.columns:
            raise ValueError("Column 'close' not found in dataset")
        elif "max" not in self.dataset.columns:
            raise ValueError("Column 'max' not found in dataset")
        elif "min" not in self.dataset.columns:
            raise ValueError("Column 'min' not found in dataset")
        elif "volume" not in self.dataset.columns:
            raise ValueError("Column 'volume' not found in dataset")
        
        # self.dataset["datetime"] = self.dataset["datetime"].apply(safe_convert_datetime)
        self.dataset["datetime"] = pd.to_datetime(self.dataset["datetime"], 
                                                  format='%Y-%m-%d %H:%M:%S', 
                                                  errors='coerce')
    
        self.dataset = self.dataset.dropna(subset=["datetime"])

        # self.dataset["datetime"] = pd.to_datetime(self.dataset["datetime"], 
        #                                           format="%Y-%m-%d %H:%M:%S",
        #                                           errors='coerce')
        
        # self.dataset = self.dataset.dropna(subset=["datetime"])

        self.timetravel = timetravel

    @timer
    def sort(self):
        self.dataset = self.dataset.sort_values(by='datetime', 
                                        ignore_index=True,
                                        ascending=True)
        return self

    @timer
    def clear_dataset(self) -> pd.DataFrame:
        # self.dataset = clear_dataset(self.dataset, sort=True, timetravel=self.timetravel)
        dataset = self.dataset.copy()

        for col in dataset.columns:
            if col in ["datetime", "volume"]:
                continue

            dataset[col] = dataset[col].apply(str_to_float) 

        dataset = convert_volume(dataset)
        logger.debug("Volume converted to float %d", len(dataset))

        # dataset = dataset.drop_duplicates(subset=['datetime'], ignore_index=True)
        grouped = find_most_common_df(dataset)

        dataset = dataset.drop_duplicates(subset=['datetime'], ignore_index=True)
        dataset = pd.concat([grouped, dataset])

        dataset = dataset.drop_duplicates(subset=['datetime'], ignore_index=True)

        dataset = conncat_missing_rows(dataset, timetravel=self.timetravel)
        
        logger.debug("Missing rows concatenated %d", len(dataset))

        dataset = dataset.drop_duplicates(subset=['datetime'], ignore_index=True)
        logger.debug("Duplicates removed %d", len(dataset))

        dataset = dataset.sort_values(by='datetime', 
                                        ignore_index=True,
                                        ascending=False)
        logger.debug("Dataset sorted %d", len(dataset))
        
        return dataset
    
    def set_timetravel(self, timetravel: str):
        if not timetravel in RU_EN_timetravel.keys() or timetravel.isdigit():
            raise ValueError(f"Invalid timetravel: {timetravel}")

        self.timetravel = timetravel
    
    def duplicated(self):
        return self.dataset[self.dataset.duplicated(keep=False)]

    def plot_series(self, dataset: list | None = None, param: str = "close") -> None:
        plt.figure(figsize=(12, 8))

        if dataset is None:
            y = self.dataset[param]
            utils.plot_series(y)
            plt.title(param)
            plt.tick_params(axis='both', which='major', labelsize=14)

            plt.show()
        else:
            dates = [item['datetime'] for item in dataset]
            closes = [item[param] for item in dataset]

            # Построение графика
            plt.figure(figsize=(10, 5))
            plt.plot(dates, closes, marker='o')
            # plt.title('График цены закрытия')
            plt.xlabel('Время')
            plt.ylabel('Цена')
            plt.xticks(rotation=45)
            plt.grid()
            plt.tight_layout()
            plt.show()

    def get_dataset_Nan(self) -> pd.DataFrame:
        return self.dataset.loc[self.dataset['open'] == "x"]
    
    def dataset_clear(self) -> pd.DataFrame:
        return self.dataset.loc[self.dataset['open'] != "x"]
    
    def get_datetime_last(self) -> datetime:
        return self.dataset['datetime'].iloc[-1]