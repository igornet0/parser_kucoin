import pandas as pd
from datetime import datetime

from typing import Union
from os import walk, mkdir, getcwd, path


class Dataset:

    main_dir = getcwd()

    def __init__(self, dataset: Union[pd.DataFrame, str], save: bool = True, path_save: str = "datasets", 
                 file_name: str = "clear_dataset.csv", search: bool = True) -> None:
        
        if not isinstance(dataset, pd.DataFrame):
            if search:
                dataset = self.searh_dateset(dataset)
                path_save = path.dirname(dataset)
                
            dataset = pd.read_csv(dataset)

        if 'Unnamed: 0' in dataset.columns:
            dataset.drop('Unnamed: 0', axis=1, inplace=True)

        if "date" in dataset.columns:
            dataset.rename(columns={"date": "datetime"}, inplace=True)

        if "datetime" in dataset.columns:
            dataset["datetime"] = pd.to_datetime(dataset["datetime"])

        self.dataset = dataset
        self.save = save
        self.path_save = path_save
        self.file_name = file_name

        if save:
            self.save_dataset()

    def get_datetime_last(self) -> datetime:
        return self.dataset['datetime'].iloc[-1]

    @classmethod
    def searh_dateset(cls, path_searh: str) -> str:
        if path_searh.endswith(".csv"):
            return path_searh
        
        for root, _, files in walk(path_searh):
            for file in files:
                if file.endswith(".csv"):
                    return path.join(root, file)
    
    def get_dataset(self) -> pd.DataFrame:
        return self.dataset
    
    def concat_dataset(self, dataset: pd.DataFrame, sort: bool = True) -> pd.DataFrame:
        # if isinstance(dataset, DatasetTimeseries):
        #     dataset = dataset.dataset_clear()
        if isinstance(dataset, Dataset):
            dataset = dataset.get_dataset()
            
        elif not isinstance(dataset, pd.DataFrame):
            raise ValueError("Dataset must be DatasetTimeseries or Dataset or pd.DataFrame")

        self.dataset = pd.concat([self.get_dataset(), dataset], ignore_index=True)

        self.dataset = self.dataset.drop_duplicates(subset=['datetime'])

        if sort:
            self.dataset = self.dataset.sort_values('datetime', ignore_index=True)

        return self
    
    def set_dataset(self, dataset: pd.DataFrame) -> None:
        if isinstance(dataset, Dataset):
            dataset = dataset.get_dataset()
        elif not isinstance(dataset, pd.DataFrame):
            raise ValueError("Dataset must be DatasetTimeseries or Dataset or pd.DataFrame")
        
        self.dataset = dataset

    def save_dataset(self, name_file: str = None) -> None:
        if not path.exists(self.path_save):
            mkdir(self.path_save)

        if name_file is None:
            name_file = self.file_name

        if path.exists(path.join(self.path_save, name_file)):
            dataset = Dataset(path.join(self.path_save, name_file), save=False)
            self.concat_dataset(dataset)

        self.dataset.to_csv(path.join(self.path_save, name_file))

    def get_filename(self) -> str:
        return self.file_name

    def __getitem__(self, date):
        pass

    def __len__(self):
        return len(self.dataset)
