from typing import Generator

class DataParser(dict):

    def __getitem__(self, key):
        value = super().__getitem__(key)
        data_buffer = []
        while isinstance(value, Generator):
            try:
                data_buffer.append(next(value))
            except StopIteration:
                if data_buffer:
                    self[key] = data_buffer if len(data_buffer) > 1 else data_buffer[0]
                else:
                    self[key] = None

                break

        return super().__getitem__(key)