from core.utils.gui_deps import GUICheck

# В файлах с GUI-логикой используйте:
if GUICheck.has_gui_deps():
    from PIL import Image
else:
    class Image: pass

from pydantic import Field
from os import walk, listdir
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional, Literal, Any, Dict, Union, Generator, Callable
import pandas as pd
import aiofiles
import json
import zipfile
import yaml
import shutil
from datetime import datetime
from functools import cached_property

import logging

logger = logging.getLogger("DataManager")

class SettingsTrade(BaseSettings):

    # Пути к данным
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_PATH: Path = DATA_DIR / "raw"
    CACHED_DATA_PATH: Path = DATA_DIR / "cached"
    PROCESSED_DATA_PATH: Path = DATA_DIR / "processed"
    BACKUP_DATA_PATH: Path = DATA_DIR / "backup"
    TRACH_PATH: Path = DATA_DIR / "trach"
    LOG_PATH: Path = DATA_DIR / "log"
    
    COIN_LIST_PATH: Path = DATA_DIR / "coins_list.csv"

    TYPE_DATASET_FOR_COIN: Optional[Literal["clear", "train", "test"]] = Field(default="clear", description="Тип датасета (clear, train, test) для сохранения в БД")
    TIMETRAVEL: Optional[str] = Field(default="5m", description="Временная метка данных")

    # Модели ML
    MODELS_DIR: Path = BASE_DIR / "models"
    MODELS_CONFIGS_PATH: Path = MODELS_DIR / "model_configs"
    # ACTIVE_MODEL: FilePath = TRAINED_MODELS_PATH / "current_model.pkl"

class DataManager:

    _settings = SettingsTrade()

    def __init__(self):
        self.required_dirs = {
            "raw": self.settings.RAW_DATA_PATH,
            "processed": self.settings.PROCESSED_DATA_PATH,
            "cached": self.settings.CACHED_DATA_PATH,
            "backup": self.settings.BACKUP_DATA_PATH,
            "log": self.settings.LOG_PATH,
            "trach": self.settings.TRACH_PATH,
            }

        self._ensure_directories_exist()
        self._setup_logging()

    @property
    def settings(self) -> SettingsTrade:
        return self._settings

    def _ensure_directories_exist(self):
        """Создает все необходимые директории при инициализации"""
        for directory in self.required_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Настройка логирования для DataManager"""
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(self.settings.LOG_PATH / "data_manager.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    def __getitem__(self, key):
        if not self.required_dirs.get(key):
            raise AttributeError(f"SettingsTrade has no attribute '{key}'")
        
        return self.required_dirs[key]

    @cached_property
    def coin_list(self) -> list[str]:
        """Кэшированный список монет"""
        return pd.read_csv(self.settings.COIN_LIST_PATH)["name"]
    
    def update_coin_list(self, new_list: list[str]) -> None:
        dt = pd.DataFrame(new_list, columns=["name"])
        dt.to_csv(self.settings.COIN_LIST_PATH)

    def get_last_launch(self) -> Path:
        paths = list(self.get_path("raw", filter_path=lambda x: "launch_parser" in str(x.name)))

        if not paths:
            return None
        
        return max(paths, key=lambda x: int(str(x.name).split("_")[-1]))

    def get_path(
        self,
        data_type: Literal["raw", "processed", "cached", "backup"],
        coin: Optional[str] = None,
        dataset_type: Optional[str] = None,
        timetravel: Optional[str] = None,
        filter_path: Optional[Callable] = lambda x: True
    ) -> Generator[Path, None, None]:
        """
        Генерирует путь к данным на основе параметров
        """

        base_path = self.required_dirs[data_type]

        if not coin and not dataset_type and not timetravel:
            for dirs in listdir(base_path):
                if not filter_path(Path(base_path) / dirs):
                    continue

                yield Path(base_path) / dirs

        for root, dirs, files in walk(base_path):
            if not all(map(lambda x: ".csv" in x, files)):
                continue

            for file in files:

                if coin:
                    if not coin in file.replace("_", "-").split("-"):
                        continue

                if (timetravel and file.endswith(f"{timetravel}.csv")) or dataset_type:
                    if (dataset_type and file.startswith(f"{dataset_type}")) or not dataset_type:
                        if not filter_path(Path(root) / file):
                            continue

                        yield Path(root) / file


    def save_img(self, img: Image, time_parser: str = "5m", name: str = "img") -> None:
        path = self.create_dir("raw", "img")
        path = self.create_dir("raw", "img/" +time_parser)
        name = name.replace(":", "_").replace(" ", "_").replace("/", "-")

        img_path = path / f"{name}.png"
        img.save(img_path)
        logger.info(f"Image saved to {img_path}")

    async def read_file(self, path: Path, format: str = "csv") -> pd.DataFrame:
        """
        Асинхронное чтение файлов данных
        Поддерживаемые форматы: csv, parquet, json
        """
        try:
            if format == "csv":
                async with aiofiles.open(path, mode="r") as f:
                    return pd.read_csv(await f.read())
            elif format == "parquet":
                return pd.read_parquet(path)
            elif format == "json":
                async with aiofiles.open(path, mode="r") as f:
                    return pd.read_json(await f.read())
            else:
                raise ValueError(f"Unsupported format: {format}")
        except FileNotFoundError:
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            raise

    async def write_file(
        self,
        data: Union[pd.DataFrame, Dict],
        path: Path,
        format: str = "csv",
        mode: str = "w"
    ) -> None:
        """
        Асинхронная запись данных в файл
        """
        try:
            if format == "csv":
                async with aiofiles.open(path, mode=mode) as f:
                    await f.write(data.to_csv(index=False))

            elif format == "parquet":
                data.to_parquet(path)

            elif format == "json":
                async with aiofiles.open(path, mode=mode) as f:
                    await f.write(json.dumps(data))
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data successfully saved to {path}")
        except Exception as e:
            logger.error(f"Error writing to {path}: {str(e)}")
            raise

    def cache_data(self, data: Any, key: str) -> None:
        """
        Кэширование данных в памяти и на диске
        """
        try:
            cache_path = self.settings.CACHED_DATA_PATH / f"{key}.pkl"
            pd.to_pickle(data, cache_path)
            logger.info(f"Data cached: {key}")
        except Exception as e:
            logger.error(f"Cache error for {key}: {str(e)}")

    def load_cache(self, key: str) -> Any:
        """
        Загрузка данных из кэша
        """
        try:
            cache_path = self.settings.CACHED_DATA_PATH / f"{key}.pkl"
            return pd.read_pickle(cache_path)
        except FileNotFoundError:
            logger.warning(f"Cache not found: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache load error for {key}: {str(e)}")
            return None

    def backup_data(self, paths: list[Path], backup_name: str = None) -> Path:
        """
        Создание резервной копии данных
        """
        backup_name = backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        backup_path = self.settings.BACKUP_DATA_PATH / backup_name
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for path in paths:
                    if path.is_file():
                        zipf.write(path, arcname=path.name)
                        
                    elif path.is_dir():
                        for file in path.rglob('*'):
                            if file.is_file():
                                zipf.write(file, arcname=file.relative_to(path.parent))
            
            logger.info(f"Backup created: {backup_path}")
            return backup_path
        
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            raise

    def validate_dataset(self, df: pd.DataFrame, expected_columns: list) -> bool:
        """
        Валидация структуры датасета
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Invalid data type. Expected DataFrame")
            return False
        
        missing_cols = set(expected_columns) - set(df.columns)

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        if df.empty:
            logger.warning("Empty dataset")
            return False
        
        return True

    def create_dir(self, type_dir: Literal["raw", "processed", "cached", "backup"], 
                   name_of_dir: str) -> Path:
        """
        Создание директории
        """
        try:
            path = self.required_dirs[type_dir] / name_of_dir
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created: {path}")
            return path
        except Exception as e:
            logger.error(f"Error creating directory: {str(e)}")
            raise
        
    def get_latest_processed_data(self, coin: str, type_data: str = "processed") -> Optional[pd.DataFrame]:
        """
        Получение последней версии обработанных данных для указанной монеты
        """
        try:
            coin_dir = self.required_dirs[type_data] / coin
            if not coin_dir.exists():
                return None

            latest_file = max(
                coin_dir.glob("*.parquet"),
                key=lambda x: x.stat().st_mtime,
                default=None
            )

            return pd.read_parquet(latest_file) if latest_file else None
        except Exception as e:
            logger.error(f"Error getting latest data for {coin}: {str(e)}")
            return None

    async def cleanup_trach(self, max_age_days: int = 30) -> None:
        """
        Очистка устаревших файлов в директории trach
        """
        try:
            now = datetime.now().timestamp()
            for file in self.settings.TRACH_PATH.glob('*'):
                if (now - file.stat().st_mtime) > max_age_days * 86400:
                    if file.is_dir():
                        shutil.rmtree(file)
                    else:
                        file.unlink()
                    logger.info(f"Removed old file: {file}")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
            raise

    def get_model_config(self, model_name: str) -> dict:
        """
        Загрузка конфигурации модели
        """
        config_path = self.settings.MODELS_CONFIGS_PATH / f"{model_name}.yaml"
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading model config {model_name}: {str(e)}")
            raise

    async def save_processed_data(
        self,
        coin: str,
        data: pd.DataFrame,
        dataset_type: str = "clear",
        version: str = None
    ) -> Path:
        """
        Сохранение обработанных данных с версионированием
        """
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = self.get_path(
            "processed",
            coin=coin,
            dataset_type=dataset_type,
            timetravel=self.settings.TIMETRAVEL
        )
        
        filename = f"{coin}_{dataset_type}_{self.settings.TIMETRAVEL}_{version}.parquet"
        save_path = save_dir / filename
        
        await self.write_file(data, save_path, format="parquet")
        
        return save_path
    
data_manager = DataManager()