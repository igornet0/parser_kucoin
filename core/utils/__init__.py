__all__ = (
    "image_to_text",
    "preprocess_image",
    "str_to_datatime",
    "RU_EN_timetravel",
    "timetravel_seconds_int",
    "AutoDecorator",
    "GUICheck",
)

from core.utils.gui_deps import GUICheck

# # В файлах с GUI-логикой используйте:
# if GUICheck.has_gui_deps():
from core.utils.tesseract_img_text import (RU_EN_timetravel, timetravel_seconds_int, 
                                        image_to_text, str_to_datatime, preprocess_image)

# from core.utils.configure_logging import setup_logging, OverwriteHandler

from core.utils.decorater_auto import AutoDecorator
