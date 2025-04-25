import pandas as pd
from datetime import datetime, timedelta
from sktime.forecasting.base import ForecastingHorizon

from core.utils.tesseract_img_text import timetravel_seconds_int

# def safe_convert_datetime(date_str):
#     try:
#         # Пробуем преобразовать в datetime
#         dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
#         # dt = pd.to_datetime(date_str, errors='raise')
        
#         # Проверяем, входит ли дата в допустимый диапазон pandas
#         if dt < pd.Timestamp.min or dt > pd.Timestamp.max:
#             return pd.NaT  # Возвращаем NaT для выходящих за пределы
#         return dt
#     except:
#         return pd.NaT  # Обработка некорректных форматов

def volume_to_float(item: str) -> float:
    if item == "x":
        return item
    
    volume_int = {"K": 10**3, "M": 1, "B": 10**(-3)}

    if item[-1] not in volume_int.keys():
        return float(item)
    
    return round(float(item.replace(item[-1], "")) / volume_int[item[-1]], 2)

def convert_volume(dataset: pd.DataFrame, volume_column: str = "volume") -> pd.DataFrame:
    if volume_column not in dataset.columns:
        raise ValueError(f"Column '{volume_column}' does not exist in the DataFrame.")

    pop_list = []

    for index, row in dataset.iterrows():
        if isinstance(row[volume_column], float):
            continue

        if not isinstance(row[volume_column], str) or row[volume_column] == "x":
            pop_list.append(index)
            continue

        dataset.at[index, volume_column] = volume_to_float(row[volume_column])

    dataset.drop(pop_list, inplace=True)

    return dataset

def str_to_float(item: str) -> float | None:
    if not isinstance(item, str):
        return None
    
    result = item.replace(' ', '').replace(',', '.')

    try:
        return float(result)
    except ValueError:
        pass

    return None
    
def clear_datetime_false(df: pd.DataFrame, datetime_column: str = "datetime") -> pd.DataFrame:
    if datetime_column not in df.columns:
        raise ValueError(f"Column '{datetime_column}' does not exist in the DataFrame.")

    # Преобразуем столбец в тип datetime, если он еще не в этом формате
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
    # df[datetime_column] = pd.to_datetime(df[datetime_column])
    df = df.sort_values(datetime_column, ignore_index=True)

    # Итерируем по строкам DataFrame
    for index, row in df.iterrows():
        if row[datetime_column] is False:
            # Находим предыдущую строку
            previous_value = df.at[index - 1, datetime_column]
            
            # Если предыдущая строка тоже не является NaT (Not a Time)
            if pd.notna(previous_value):
                # Вычисляем разницу между предыдущими значениями
                time_difference = df.at[index - 1, datetime_column] - df.at[index - 2, datetime_column]
                
                # Записываем новое значение
                df.at[index, datetime_column] = previous_value + time_difference

    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')

    return df

def get_time_range(timetravel) -> dict:
        time_start = "07:00"

        if timetravel[-1] == "m":
            time_end = "18:55"

        elif timetravel == "1H":
            time_end = "18:00"

        elif timetravel == "4H":
            time_end = "15:00"

        elif timetravel == "1D":
            time_end, time_start = "00:00", "00:00"
    
        time_range = pd.date_range(start=time_start, end=time_end, freq=timetravel.replace("m", "T"))

        return time_range

def conncat_missing_rows(df, timetravel: str = "5m", datetime_column: str='datetime') -> pd.DataFrame:

    timetravel = timetravel_seconds_int[timetravel]

    df = df.sort_values(datetime_column, ignore_index=True)

    missing_rows = []
    buffer_rows = []
    count_missing = 0
    for index, row in df.iterrows():

        if row[datetime_column] is pd.NaT:
            time = df.iloc[index - 1][datetime_column] + timedelta(seconds=timetravel)

            row[datetime_column] = time
            for col in df.columns[1:]:
                row[col] = 'x'
            missing_rows.append(row)

        buffer_rows.append(row)

        if len(buffer_rows) == 2:
            
            delta = buffer_rows[1][datetime_column] - buffer_rows[0][datetime_column]

            while delta != timedelta(seconds=timetravel):
                buffer_rows[0][datetime_column] += timedelta(seconds=timetravel)

                new_row = {datetime_column: buffer_rows[0][datetime_column]}

                for col in df.columns[1:]:
                    new_row[col] = 'x'

                missing_rows.append(new_row)
                count_missing += 1
                delta = buffer_rows[1][datetime_column] - buffer_rows[0][datetime_column]

            buffer_rows.pop(0)
            count_missing = 0

    df_missing_rows = pd.DataFrame(missing_rows)

    if len(missing_rows) > 0:
        df_missing_rows[datetime_column] = pd.to_datetime(df_missing_rows[datetime_column], errors='coerce')
        return pd.concat([df, df_missing_rows])
    
    return df

def check_dt(dfs: list[pd.DataFrame], datetime_column: str = "datetime") -> bool:
    # Объединяем все DataFrame в один
    combined_df = pd.concat(dfs, ignore_index=True)

    # Группируем по 'datetime'
    grouped = combined_df.groupby(datetime_column)

    result_rows = []
    for datetime, group in grouped:
        # Группируем по целевым столбцам для подсчёта комбинаций
        value_counts = group.groupby(['open', 'max', 'min', 'close', 'volume']).size()
        total = len(group)
        required = 0.8 * total
        # Проверяем каждую комбинацию
        meets_condition = False
        most_common_combo = None
        max_count = 0
        for combo, count in value_counts.items():
            if count >= required:
                meets_condition = True

                if count > max_count:
                    max_count = count
                    most_common_combo = combo

        if meets_condition:
            # Извлекаем первую строку с наиболее частой комбинацией
            mask = (
                (group['open'] == most_common_combo[0]) &
                (group['max'] == most_common_combo[1]) &
                (group['min'] == most_common_combo[2]) &
                (group['close'] == most_common_combo[3]) &
                (group['volume'] == most_common_combo[4])
            )
            selected_row = group[mask].iloc[0].copy()
            selected_row[datetime_column] = datetime  # Восстанавливаем datetime
            result_rows.append(selected_row)

    # Создаём итоговый DataFrame
    result_df = pd.DataFrame(result_rows)
    result_df = result_df[[datetime_column, 'open', 'max', 'min', 'close', 'volume']]

    # Сортируем по времени и сбрасываем индекс
    result_df.sort_values(datetime_column, ascending=False, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    
    return result_df

def clear_dataset(dataset: pd.DataFrame, timetravel: str = None, sort: bool = False) -> pd.DataFrame:
    dataset = clear_datetime_false(dataset)

    for col in dataset.columns:
        if col in ["datetime", "volume"]:
            continue

        dataset[col] = dataset[col].apply(str_to_float) 

    dataset = convert_volume(dataset)
    dataset = dataset.drop_duplicates(subset=['datetime'], ignore_index=True)

    if timetravel:
        dataset = conncat_missing_rows(dataset, timetravel=timetravel)

    dataset = dataset.drop_duplicates(subset=['datetime'], ignore_index=True)

    if sort:
        dataset = dataset.sort_values(by='datetime', 
                                        ignore_index=True,
                                        ascending=False)

    return dataset


if __name__ == "__main__":
    dataset = pd.read_csv(input("dataset: "), index_col="Unnamed: 0")
    clear_dataset(dataset)