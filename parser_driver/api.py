from abc import abstractmethod
import asyncio
from core.utils.gui_deps import GUICheck

if GUICheck.has_gui_deps():
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.remote.webelement import WebElement
    from selenium.common.exceptions import TimeoutException
    from PIL import Image
    from core.utils import image_to_text, str_to_datatime
    from .device_real import Device
else:
    # Заглушки для типизации
    class By: XPATH = None  # noqa
    class EC: pass  # noqa
    class WebElement: pass  # noqa
    class WebDriverWait: pass  # noqa
    class TimeoutException: pass  # noqa
    class Device: pass
    class Image: pass  # noqa

import pandas as pd
from threading import Event
from typing import Any, Callable, Union, Generator
import base64, json
from io import BytesIO
from datetime import datetime
from shutil import rmtree
from os import path, listdir, mkdir

from .web_driver import WebDriver
from .data import DataParser
from core import data_manager
from core.models import Dataset

import logging

logger = logging.getLogger("parser_logger.Api")

class ParserApi:

    def __init__(self, tick: int = 1, driver = None, import_device: bool = False) -> None:

        self.driver_class = WebDriver if driver is None else driver
        self.driver_instance = None
        self.flag_open_web = False

        self.filename = None
        self.path_trach = data_manager["trach"]
        self.path_save = data_manager["raw"]
        self.name_launch = "launch_parser"

        self.tick = tick

        self.xpath = {}

        self._device = Device
        self.buffer_date = []

        self.default_options = WebDriver.WebOptions

        if import_device:
            task = self.import_device()


    def import_device(self, device: Device = None) -> asyncio.Task:
        self._device = device if device is not None else self.device(self.tick)
        logger.info(f"Device import {type(self._device)}")
        # self.keypress_thread = threading.Thread(target=self.device.kb.start_listener, daemon=True)
        # self.keypress_thread.start()

    @property
    def device(self) -> Device:
        return self._device
    
    def get_default_options(self) -> WebDriver.WebOptions:
        return self.default_options()

    def open_driver(self, options=None, use_subprocess: bool=True) -> WebDriver:
        if self.driver_instance is not None:
            self.driver_instance.quit()
        
        # Создаем новый экземпляр драйвера
        self.driver_instance = self.driver_class(
            options=options, 
            use_subprocess=use_subprocess
        )

        return self.driver_instance

    def set_options(self, new_options: WebDriver.WebOptions) -> None:
        self.default_options = new_options

    def add_data_buffer(self, data: dict):
        self.buffer_date.append(data)

    def get_data_buffer(self) -> list[datetime]:
        return self.buffer_date
    
    def clear_data_buffer(self):
        self.buffer_date = []

    async def entry(self, login: str, password: str):
        if not self.driver_instance:
            raise ValueError("Driver not found")
        
        assert self.flag_open_web, "Web not open"
        
        assert login and password, "Login or password is empty"

        assert self.xpath.get("login") is not None
        assert self.xpath.get("password") is not None
        assert self.xpath.get("click_login") is not None

        await self.wait_for_page_load()

        self.login = login
        self.password = password

        await self.get_element(self.xpath["login"]["xpath"], name="login")[0].send_keys(login)
        await self.get_element(self.xpath["password"]["xpath"], name="password")[0].send_keys(password)
        await self.get_element(self.xpath["click_login"]["xpath"], name="click_login")[0].click()

        logger.info(f"Entry {login=}")

        return True

    async def start_web(self, url_open: str = None, show_browser: bool = True, window_size: tuple = (1100, 1000)) -> WebDriver:
        if self.driver_instance is None or not self.flag_open_web:
            option = self.get_default_options()

            if not show_browser:
                option.add_argument("--headless=new")

            self.open_driver(option)
        
        if show_browser:
            self.driver_instance.set_window_size(*window_size)

        if not url_open is None:
            self.driver_instance.get(url_open) 

        await self.wait_for_page_load()

        self.driver_instance.switch_to.default_content()

        logger.info(f"Start web {url_open=}")

        self.flag_open_web = True

        return self.driver_instance
    
    @abstractmethod
    async def start_parser(self, counter: int = 1) -> pd.DataFrame:
        pass

    async def check_connection(self) -> bool:
        if self.driver_instance is None or not self.flag_open_web:
            return False
        
        try:
            url = self.driver_instance.current_url
            if "new-tab-page" in url:
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Check connection error: {e}")
            return False

    def close(self) -> bool:
        if self.driver_instance is None or not self.flag_open_web:
            return False
        
        logger.info("Close web")

        self.driver_instance.quit()
        self.driver_instance = None
        self.flag_open_web = False

        return True
    
    async def restart(self):
        assert self.driver_instance is not None

        logger.info("Restart web")

        connect = await self.check_connection()
        if connect:
            url = self.driver_instance.current_url
            self.close()
        else:
            url = None

        await self.start_web(url)
    
    async def switch_frame(self, frame: str = "frame") -> bool:
        assert self.xpath.get(frame) is not None
        assert self.driver_instance is not None
        try:
            frame = await self.get_element(self.xpath[frame]["xpath"], by=By.TAG_NAME, name="frame")
            self.driver_instance.switch_to.frame(frame) 
        except Exception as e:
            logger.error(f"Switch frame error: {e}")
            return False
        
        return True

    def click(self, element: WebElement) -> None:
        assert self.driver_instance is not None
        
        try:
            element.click()
        except Exception as e:
            logger.error(f"Click error: {e}")

    def search_element_text(self, elements, text):

        for element in elements:
            if text and text in element.text.strip().replace("\n", "").lower():
                return element
            
        return False
    
    async def get_filename(self, default: str = "data") -> str:

        if "filename" not in self.xpath.keys():
            filename = default
        else:
            filename = await self.get_element(self.xpath["filename"]["xpath"], 
                                        text=True, name="filename")
            if not filename:
                 filename = default

        if not filename and self.filename:
            filename = self.filename
        
        self.filename = f"{filename}.csv"
        self.filename = self.filename.replace("/", "_")

        logger.debug(f"Get filename {self.filename=}")

        return self.filename

    def add_xpath(self, key:str, xpath:str, parse:bool = True, 
                  func_get:Callable | None = None, args: tuple = (), kwargs: dict = {}) -> None:
        
        self.xpath[key] = {"xpath": xpath, "parse": parse, 
                           "func_get": func_get, "args": args, "kwargs": kwargs}
        
    async def wraper_get_element(self, get_elem: WebElement, key: str = "element", *args, **kwargs) -> WebElement:
        assert self.driver_instance is not None
        element = await get_elem(*args, **kwargs)
        return {key: element}

    async def get_elements(self) -> DataParser:
        data_d = DataParser()
        tasks = {}
        for key, xpath_data in self.xpath.items():
            xpath, parse, func_get = xpath_data["xpath"], xpath_data["parse"], xpath_data["func_get"]

            if not parse:
                continue

            logger.debug(f"Getting element {key=}")

            if not func_get is None:
                args, kwargs = xpath_data["args"], xpath_data["kwargs"]

                # element = await func_get(*args, **kwargs)
                task = asyncio.create_task(self.wraper_get_element(func_get, key, *args, **kwargs))
            else:
                # element = await self.get_element(xpath, text=True, name=key)
                # asyncio.create_task(self.get_element(xpath, text=True, name=key))
                # element = None
                task = asyncio.create_task(self.wraper_get_element(self.get_element, key, xpath, text=True, name=key))

            tasks[key] = task

        [tasks.update(data) for data in await asyncio.gather(*tasks.values())]

        for key, element in tasks.items():
            if not element:
                logger.error(f"Element {key} not found")

            logger.debug(f"Get element {key=} {element=}")
                
            data_d[key] = element
        
        return data_d

    async def get_element(self, xpath:str, by=By.XPATH, name="element", text=False, all=False) -> Union[str, list]:
        assert self.driver_instance is not None

        iter = 20
        while True:
            try:
                if all:
                    element = WebDriverWait(self.driver_instance, max(self.tick, 1)).until(
                            EC.presence_of_all_elements_located((by, xpath))
                        )
                else:
                
                    element = WebDriverWait(self.driver_instance, max(self.tick, 1)).until(
                            EC.presence_of_element_located((by, xpath))
                        )

                if element is None:
                    raise ValueError(f"element {xpath} is None")

                if text:
                    if all:
                        return [e.text if e.text else "" for e in element]
                    
                    if not element.text:
                        raise ValueError(f"element {xpath} not text")

                    return element.text
                
                return element

            except Exception as e:
                iter -= 1
                await asyncio.sleep(self.tick)
                if iter == 0:
                    n_error = len(list(filter(lambda x: x.endswith('_screenshot_error.png'), listdir(self.path_trach)))) + 1
                    self.driver_instance.save_screenshot(path.join(self.path_trach, "img", f"{n_error}_screenshot_error.png"))

                    logger.error(f"error:{n_error} Not found element {name=}")
                    break

    def set_filename(self, filename: str):
        self.filename = filename

    async def wait_for_page_load(self, timeout=30):
        """
        Ожидает полной загрузки страницы
        :param timeout: максимальное время ожидания в секундах
        :return: True если страница загружена, False при таймауте
        """
        assert self.driver_instance is not None

        try:
            await self.check_loop()
            WebDriverWait(self.driver_instance, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
                and len(d.find_elements(By.TAG_NAME, "body")) > 0
            )
            await asyncio.sleep(5)
            
            logger.info("Page loaded")
            return True
        except TimeoutException:
            logger.error("Page load timeout")
            return False
    
    async def finally_parser(self, data: pd.DataFrame, counter: int = 1) -> Dataset:

        if len(data) != counter:
            logger.warning(f"Length data = {len(data)}!={counter}")
            
        if len(data) == 0:
            logger.error("No data")
        else:
            data = pd.DataFrame(data)
            data = data.drop_duplicates(subset=["datetime"])
            data["datetime"] = pd.to_datetime(data["datetime"])
        
            datetime_last = data['datetime'].min()

            logger.info(f"Last datetime = {datetime_last}")
            logger.info("End parser")
            
            self.clear_data_buffer()

            return Dataset(data)
        
    def wrapper_gen(self, func: Callable, *args: tuple, **kwargs: dict) -> Generator[Any, None, None]:
        yield func(*args, **kwargs)

    async def get_element_datetime(self, process=False, get_img=False) -> datetime | Generator[datetime, None, None]:
        error_buffer = []
        while True:
            try:
                element = await self.get_element(self.xpath["datetime"]["xpath"], name="datetime")
                img = self.get_img(element)

                if img:
                    if get_img:
                        return img
                    
                    if process:
                        date = self.get_datetime(img)

                        if not date:
                            raise ValueError("Date not found")
                    else:
                        date = self.wrapper_gen(self.get_datetime, img)
                    
                    return date
                
                raise ValueError("Image not found")
            
            except ValueError:
                error_buffer.append(date)
                if len(error_buffer) > 10:
                    logger.error(f"error_buffer = {set(error_buffer)}")
                    return None
                
    def get_img(self, element: WebElement) -> Image:
        image_data = base64.b64decode(self.driver_instance.execute_script('return arguments[0].toDataURL().substring(21);', element))
        img = Image.open(BytesIO(image_data))

        return img

    def get_datetime(self, img: Image) -> datetime | None:
        try:
            text = image_to_text(img)
            date = str_to_datatime(text)
        except Exception as e:
            img.save(path.join(self.path_trach, "img", f"{len(list(filter(lambda x: x.endswith('_error.png'), listdir(self.path_trach)))) + 1}_error.png"))
            self.driver_instance.save_screenshot(path.join(self.path_trach, "img", f"{len(list(filter(lambda x: x.endswith('screenshot_error.png'), listdir(self.path_trach)))) + 1}_screenshot_error.png"))
            logger.error(f"Get datetime error: {e}")
            date = None
        
        return date
        
    # async def handler_loop(self):
    #     while True:
    #         stop_event = await self.get_stop_event()

    #         if stop_event.is_set():
    #             logger.info("Stop parser by keypress")
    #             return False
            
    #         pause_event = self.get_pause_event()
                
    #         if pause_event.is_set():
    #             logger.info("Pause parser by keypress")

    #             while pause_event.is_set():
    #                 await asyncio.sleep(self.tick)  
    #                 stop_event = self.get_stop_event()

    #                 if stop_event.is_set():
    #                     return False
                    
    #                 pause_event = self.get_pause_event()

    #             logger.info("Resuming parser")
            
    #         await asyncio.sleep(self.tick)

    async def check_loop(self):

        if self.get_stop_event():
            return False
        
        if self.get_pause_event():
            while self.get_pause_event():
                await asyncio.sleep(self.tick)  
                if self.get_stop_event():
                    return False

        return True

    def get_stop_event(self) -> Event:
        return self.device.kb.get_stop_loop()
    
    def get_pause_event(self) -> Event:
        return self.device.kb.get_pause_loop()

    async def search_datetime(self, target_datetime: datetime, 
                              right_break: bool = False) -> bool:
        buffer_life = 3
        logger.info(f"Search datetime {target_datetime}")

        self.device.cursor.scroll_to_start()

        while True:

            if not await self.check_loop():
                return False
            
            if self.device.cursor.get_position_now() != self.device.cursor.get_position["start"]:
                self.device.cursor.move_to_position()

            date = await self.get_element_datetime(process=True) or self.get_last_buffer_date()

            if date is None:
                continue

            self.add_data_buffer(date)

            delta = abs((target_datetime - date).total_seconds())

            if delta == 0:
                logger.info(f"datetime {target_datetime} found")
                self.clear_data_buffer()
                return True
            
            if self.should_clear_buffer():
                date = self.get_data_buffer()[-1]
                buffer_life -= 1
                logger.debug(f"clear buffer")

                if buffer_life == 0:
                    self.device.cursor.scroll(-25)
                    logger.info(f"datetime {target_datetime} not found")
                    logger.info(f"buffer {self.buffer_date}")
                    return False
                
                self.clear_data_buffer()
                self.device.cursor.scroll(25)
            
            direction = self.determine_direction(target_datetime, date)

            if direction == "right" and right_break:
                break
            
            interval = self.determine_interval(delta)
            self.device.cursor.move(direction + interval)

        return True

    def get_last_buffer_date(self) -> datetime | None:
        return self.buffer_date[-1] if self.buffer_date else None

    def should_clear_buffer(self) -> bool:
        if len(self.buffer_date) > 10:
            if len(set(self.buffer_date)) <= 5:
                return True
        return False

    def determine_direction(self, target_datetime: datetime, date: datetime) -> str:
        if target_datetime < date:
            return "left"
        else:
            return "right"

    def determine_interval(self, delta: float) -> str:
        if delta / 60 < 60 * 4:
            return ""
        if delta / 60 < 60 * 8:
            return "_middle"
        else:
            return "_fast"
        
    def set_save_trach(self, path:str):
        self.path_trach = path

    def set_save_path(self, path:str):
        self.path_save = path

    def save_data(self, data: pd.DataFrame, path_save=None, file_name=None) -> pd.DataFrame:
        if path_save is None:
            path_save = self.create_launch_dir()

        if file_name is None:
            file_name = self.get_filename()

        data.to_csv(path.join(path_save, file_name), index=False)

        return data
    
    def create_launch_dir(self) -> str:
        n = len([f for f in listdir(self.path_save) if f.startswith(self.name_launch)]) + 1

        mkdir(path.join(self.path_save, f"{self.name_launch}_{n}"))
        
        logger.info(f"Create dir {path.join(self.path_save, f'{self.name_launch}_{n}')}")

        return path.join(self.path_save, f"{self.name_launch}_{n}")

    def remove_launch_dir(self, launch_number: int) -> None:
        path_remove = path.join(self.path_save, f"{self.name_launch}_{launch_number}")
        rmtree(path_remove)

    async def rec_xpath(self, url):
        "TEST"
        await self.start_web(url)    

        xpath = {}

        # Функция для получения XPath элемента
        self.driver_instance.execute_script("""
            let xpathList = [];
            let classNamesList = [];
                                   
            document.addEventListener('click', function(event) {
                event.preventDefault();
                let element = event.target;
                let xpath = '';
                let currentNode = element;

                // Получаем название классов
                let classNames = Array.from(element.classList).join(' ');

                while (currentNode) {
                    let name = currentNode.localName;
                    let index = Array.from(currentNode.parentNode ? currentNode.parentNode.children : []).indexOf(currentNode) + 1;
                    xpath = '/' + name + '[' + index + ']' + xpath;
                    currentNode = currentNode.parentNode;
                }

                xpathList.push(xpath);
                classNamesList.push(classNames);
                console.log('XPath:', xpath);  // Выводим XPath в консоль
                console.log('Class Names:', classNames);  // Выводим названия классов в консоль
            });

            window.getXpathList = function() { return xpathList; };  // Функция для получения списка
            window.getclassNamesList = function() { return classNamesList; };
        """)

        print(f"[INFO rec_xpath] Start rec xpath")
        while True:
            await asyncio.sleep(0.5)

            [xpath.setdefault(c, set()).add(x) for x, c in zip(self.driver_instance.execute_script("return getXpathList()"), self.driver_instance.execute_script("return getclassNamesList()"))]

            if not await self.check_loop():
                break

        print(f"[INFO rec_xpath] End rec xpath")
        print(xpath)

        for key, value in xpath.items():
            xpath[key] = list(value)

        with open("xpath_rec.json", "w") as f:
            json.dump(xpath, f)

    def __del__(self):
        if self.driver_instance is None:
            return
        
        self.close()
