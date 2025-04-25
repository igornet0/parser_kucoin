import pyautogui
import time
import logging
import asyncio
from pynput import keyboard
import signal

logger = logging.getLogger("Device")

class Cursor:

    def __init__(self, tick=0.1):
        self.tick = tick
        self.position_cursor = {"start": None}
        self.scroll_direction = 0


    def set_position(self, types: list = ["start",]):
        for type in types:
            for try_pos in range(3):
                time.sleep(self.tick+3)
                logger.info(f"{try_pos=} {type=} Porsition cursor = {self.get_position_now()}")
            
            self.position_cursor[type] = self.get_position_now()
            logger.info(f"{type=} {self.position_cursor[type]=}")


    def add_position(self, type):
        self.position_cursor[type] = pyautogui.position()


    @property
    def get_position(self):
        return self.position_cursor
    
    @classmethod
    def get_position_now(cls):
        return pyautogui.position()


    def move_to_position(self, type:str = "start"):
        pyautogui.moveTo(*self.position_cursor[type], duration=self.tick)

    @classmethod
    def click(cls):
        pyautogui.click()


    def scroll(self, direction: int):
        # pyautogui.scroll(direction)
        self.scroll_direction += direction 


    def scroll_to_start(self):
        if self.scroll_direction == 0:
            return
        
        self.scroll(-self.scroll_direction)


    def move(self, direction):

        if direction.endswith("fast"):
            interval = 2
        elif direction.endswith("middle"):
            interval = 1
        else:
            pyautogui.press(direction)
            return

        direction = direction.split("_")[0]
        pyautogui.hotkey("option", direction, interval=interval)


class Keyboard:

    def __init__(self, tick=0.1):
        self.tick = tick
        self.paused = False
        self.stopped = False
        self.lock = asyncio.Lock()
        self._loop = asyncio.get_event_loop()

        # Настройка обработчика Ctrl+C
        # signal.signal(signal.SIGINT, self.signal_handler)
        
        # Запуск слушателя клавиатуры в отдельном потоке
        logger.info("Controls: [s] - Stop | [p] - Pause")

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            suppress=False
        )
        self.listener.start()

    def signal_handler(self, signum, frame):
        """Обработчик сигнала Ctrl+C"""
        self.stopped = True
        logger.info("Get Ctrl+C")

    def on_press(self, key):
        """Обработчик нажатий клавиш"""
        try:
            if key == keyboard.KeyCode.from_char('p'):
                logger.debug("Pause key pressed")
                asyncio.run_coroutine_threadsafe(self.toggle_pause(), self.get_loop())

            elif key == keyboard.KeyCode.from_char('s'):
                self.stopped = True
                logger.debug("Stop key pressed")
                
        except Exception as e:
            logger.error(f"Error in on_press: {e}")

    async def toggle_pause(self):
        async with self.lock:
            self.paused = not self.paused
            state = "on" if self.paused else "off"
            logger.info(f"Paused {state}")

    def get_loop(self):
        return self._loop
    
    def get_pause_loop(self):
        return self.paused
    
    def get_stop_loop(self):
       return self.stopped
    
    def hotkey(self, *key: str, interval: int = 0):
        pyautogui.hotkey(*key, interval=self.tick+interval)


class Device:

    def __init__(self, tick=0.1):
        self.cursor = Cursor(tick)
        self.kb = Keyboard(tick)