from datetime import datetime
import re
from PIL import Image
import pytesseract
import cv2 
import numpy as np
# pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

months_rus_int = {"Дек": 12, "Ноя": 11, "Окт": 10, "Сен": 9, "Авг": 8, "Июл": 7, 
                "Июн": 6, "Май": 5, "Апр": 4, "Мар": 3, "Фев": 2, "Янв": 1}

RU_EN_timetravel = {"1 день":"1D", "4 часа":"4H", "1 час":"1H", "30 минут":"30m", "5 минут":"5m", "15 минут":"15m"}
timetravel_seconds_int = {"1D":24*3600, "4H":4*3600, "1H":3600, "5m":5*60, "15m":15*60}

def preprocess_image(image):
    # Загрузка и инверсия цветов
    inverted = cv2.bitwise_not(image)  # Инверсия RGB
    
    # Увеличение разрешения
    scaled = cv2.resize(inverted, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # Конвертация в оттенки серого
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    
    # Адаптивная бинаризация для светлого фона
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 8
    )
    
    # Удаление шума
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Дополнительная инверсия для Tesseract
    final = cv2.bitwise_not(cleaned)
    
    return final

def image_to_text(image) -> str:

    opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed_img = preprocess_image(opencv_img)
    
    # Настройки Tesseract с улучшенным whitelist
    custom_config = r'''
        --oem 3 --psm 6 
        -c tessedit_char_whitelist=0123456789/: 
    '''

    text = pytesseract.image_to_string(processed_img, config=custom_config).strip()
    
    # Постобработка текста: добавление пробела, если его нет
    text = re.sub(r'(\d{2})(\d{2}:\d{2}:\d{2})', r'\1 \2', text) 
    
    return text


# def image_to_text(img: Image) -> str:
#         # bbox = img.getbbox()
#         # img = img.crop(bbox).convert('L')
#         # img = Image.eval(img, lambda x: 255 - x)
#         # threshold = 150        # Пороговое значение для бинаризации (настройте по необходимости)
#         # img = img.point(lambda p: p > threshold and 255)  # Бинаризация

#         # Распознавание текста с указанием параметров
#         # text = pytesseract.image_to_string(img, config='--psm 6 --oem 3 -l rus')
#         # text = pytesseract.image_to_string(img, lang='rus')
#         text = pytesseract.image_to_string(img, config='--psm 6', lang='rus')
#         text = text.strip()
        
#         return text

def first_format_date(date_str):
    # Первый формат: 26 Дек '23 20:19 или 26 Дек '23
    russian_date_pattern = r"(\d{1,2})\s+([А-Яа-я]{3})\s+'?(\d{2})\s*(\d{2}:\d{2})?"

    # Проверка первого формата
    match_russian = re.match(russian_date_pattern, date_str)
    if match_russian:
        day = match_russian.group(1)
        month_str = match_russian.group(2)
        year = match_russian.group(3)
        time_str = match_russian.group(4)

        month = months_rus_int[month_str]

        # Формирование даты
        if time_str:
            return datetime.strptime(f"{day} {month} 20{int(year)} {time_str}", "%d %m %Y %H:%M")
        else:
            return datetime.strptime(f"{day} {month} 20{int(year)}", "%d %m %Y")

def second_format_date(date_str: str) -> datetime:
    # Второй формат: 2024/02/11 или 2024/07/02 09:30:00
    date_str = date_str.replace("_", " ")
    date_str = date_str.strip()

    iso_date_pattern = r'(\d{4})/(\d{2})/(\d{2})\s*(\d{2}):(\d{2}):(\d{2})'

    # Проверка второго формата
    match = re.match(iso_date_pattern, date_str)

    if not match:
        raise ValueError(f"Не распознан формат даты в тексте: {date_str}")

    return datetime(
        year=int(match.group(1)),
        month=int(match.group(2)),
        day=int(match.group(3)),
        hour=int(match.group(4)),
        minute=int(match.group(5)),
        second=int(match.group(6))
    )

def str_to_datatime(date_str):
    fist_format = first_format_date(date_str)
    if fist_format:
        return fist_format

    second_format = second_format_date(date_str)
    if second_format:
        return second_format
    
