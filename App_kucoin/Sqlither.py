import sqlite3
import pandas as pd

class Sqlither:

    def __init__(self, database):
        self.conn = sqlite3.connect(database)
        self.cursor = self.conn.cursor()

    def create_table(self, table_name, columns):
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})")
        self.conn.commit()

    def insert(self, table_name, values, columns=None):
        if columns:
            columns_str = f"({', '.join(columns)})"
        else:
            columns_str = ""
    
        self.cursor.execute(f"INSERT INTO {table_name} {columns_str} VALUES ({', '.join(['?'] * len(values))})", values)
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get(self, table_name, columns=None, where=None, dict_flag=False):
        self.cursor.execute(f"SELECT {', '.join(columns) if columns else '*'} FROM {table_name} {f'WHERE {where}' if where else ''}")
        if dict_flag:
            result = []
            if columns is None:
                columns = [description[0] for description in self.cursor.description]

            for row in self.cursor.fetchall():
                result.append(dict(zip(columns, row)))
            return result
        return self.cursor.fetchall()
    
    def update(self, table_name, values, where=None):
        self.cursor.execute(f"UPDATE {table_name} SET {', '.join([f'{key} = ?' for key in values.keys()])} {f'WHERE {where}' if where else ''}", list(values.values()))
        self.conn.commit()
    
    def delete(self, table_name, where=None):
        self.cursor.execute(f"DELETE FROM {table_name} {f'WHERE {where}' if where else ''}")
        self.conn.commit()

    def delete_all(self, tables):
        for table in tables:
            self.cursor.execute(f"DROP TABLE IF EXISTS {table}")

    def close(self):
        self.conn.close()

    def get_table_pd(self):
        tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql(tables_query, self.conn)
        return tables