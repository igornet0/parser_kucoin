import zipfile
from datetime import datetime
import shutil
import os

from core import DataManager
from apps.data_processing.dataset import Dataset, DatasetTimeseries, NewsDataset
from .dataset_types import get_dataset_type

import logging

logger = logging.getLogger("process_logger.dataset")

def try_catch(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {type(e)}-{e}")
    return wrapper

class Handler:

    @classmethod
    @try_catch
    def concat_dataset(cls, paths_dataset: list, save: bool = True, backup: bool = False) -> Dataset:
        datasets = []
        for path_dataset in paths_dataset:
            dataset_type = get_dataset_type(path_dataset)
            dataset: Dataset = dataset_type(path_dataset)

            if datasets and not dataset_type in list(map(lambda x: type(x), datasets)):
                raise Exception("Error concat dataset: datasets must be the same type")
            
            datasets.append(dataset)

        logger.info(f"Concat {len(datasets)} datasets")

        if len(datasets) > 1:
            dataset = Dataset.concat_dataset(*datasets)
            dataset_type = get_dataset_type(dataset)
            dataset = dataset_type(dataset)

        elif not datasets:
            raise Exception("Error concat dataset: datasets is empty")
        else:
            dataset = datasets[0]

        if save:
            dataset.set_filename(f"concat_{datasets[0].get_filename()}")
            dataset.save_dataset()

        if backup:
            cls.backup_dataset(paths_dataset)

        return dataset
    
    @classmethod
    @try_catch
    def clear_dataset(cls, path_dataset: str, save: bool = True, backup: bool = False) -> Dataset:
        dataset = get_dataset_type(path_dataset)(path_dataset)

        logger.info(f"Clear {dataset.get_filename()}")

        dataset.clear_dataset()

        if save:
            dataset.set_filename(f"clear_{dataset.get_filename()}")
            dataset.save_dataset()

        if backup:
            cls.backup_dataset([path_dataset])

        return dataset

    @classmethod
    def backup_dataset(cls, paths_dataset: list, backup_dir: str = DataManager()["backup"]) -> None:
        """Создает ZIP-архив с CSV-файлами из указанных путей"""
    
        # Создаем имя архива с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        backup_name = f'backup_{timestamp}.zip'
        backup_path = os.path.join(backup_dir, backup_name)

        # Создаем архив
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            try:
                for folder_path in paths_dataset:
                    if not os.path.exists(folder_path):
                        logger.warning(f"Path {folder_path} does not exist")
                        continue

                    if os.path.isfile(folder_path):
                        folder_path = os.path.dirname(folder_path)

                    if not os.path.isdir(folder_path):
                        logger.warning(f"Path {folder_path} is not a directory")
                        continue

                    # Получаем базовое имя папки для структуры архива
                    folder_name = os.path.basename(os.path.normpath(folder_path))
                    
                    # Рекурсивно ищем CSV-файлы
                    for root, _, files in os.walk(folder_path):
                        for file in files:
                            if file.lower().endswith('.csv'):
                                full_path = os.path.join(root, file)
                                relative = os.path.relpath(full_path, folder_path)
                                arcname = os.path.join(folder_name, relative)
                                zipf.write(full_path, arcname)
                                logger.info(f"Added {full_path} to {backup_path}")
                    
                    shutil.rmtree(folder_path)

            except Exception as e:
                logger.error(f"Error backup dataset: {type(e)}-{e}")
                return None

        return backup_path