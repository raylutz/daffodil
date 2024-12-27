# daf_sql.py


import os
import sys
import re
#import json
#import time
import sqlite3
import functools
    
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# from daffodil.lib.daf_types import \  # T_dola, T_dodi, T_la, T_lota, T_doda, T_buff, T_ds, T_lb, T_rli, T_ta, T_lor
                              # # T_ls, T_lola, T_di, T_hllola, T_loda, T_da, T_li, T_dtype_dict, 
                     
import daffodil.lib.daf_utils    as utils
#from daffodil.keyedlist import KeyedList

from typing import List, Dict, Any, Tuple, Optional, Union, cast, Type, Callable, Iterable, Iterator #
def fake_function(a: Optional[List[Dict[str, Tuple[int,Union[Any, str, Callable, Iterable, Iterator, Type]]]]] = None) -> Optional[int]:
    return None or cast(int, 0)       # pragma: no cover

logs = utils                # alias


@functools.lru_cache()
def sql_escape_str(colname: str) -> str:
    """ Encode the original non-compliant characters using double underscores and hex values. 
        this is a reversible encoding. Running this twice will not hurt.
        
        1. escapes any illegal characters as __HH where HH is the hex value.
        2. escapes the first character of any names with leading numerics.
        3. escapes the first character of any reserved words.
        
    """
    new_colname = re.sub(r'[^0-9A-Za-z_]', lambda x: f'__{ord(x.group()):X}', colname)
    if bool(re.search(r'^\d', new_colname)) or (new_colname.upper() in ['GROUP', 'NAME', 'VALUE']):
        new_colname = f"__{ord(new_colname[0]):X}" + new_colname[1:]

    return new_colname
    

def lod_to_sqlite_table(lod, table_name='tempdata', db_file_path=None, key_col='rowkey'):

    # see also: https://www.sqlite.org/fasterthanfs.html

    if db_file_path is None:
        db_file_path=f'{table_name}.db'

    # Connect to the SQLite database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Drop the table if it exists
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Extract field names from the first dictionary in the list
    if len(lod) > 0:
        field_names = list(lod[0].keys())
    else:
        raise ValueError("The list of dictionaries is empty.")

    # Create the table using field names as columns
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(field_names)})"
    cursor.execute(create_table_query)
    
    # Create an index on the key_col column
    if key_col:
        create_index_at_cursor(
                cursor          = cursor,  
                index_colname   = key_col, 
                table_name      = table_name,
                diagnose        = True)


    # Insert data from the list of dictionaries into the table using parameterized queries
    for entry in lod:
        placeholders = ', '.join(['?'] * len(field_names))
        insert_query = f"INSERT INTO {table_name} ({', '.join(field_names)}) VALUES ({placeholders})"
        values = [entry[field] for field in field_names]
        cursor.execute(insert_query, values)

    # Commit changes and close the database connection
    conn.commit()
    conn.close()

def create_index_at_cursor(cursor, index_colname: str, table_name: str, diagnose: bool=False):

    # creating the index at the end is more efficient.
    logs.sts(f"{logs.prog_loc()} creating index for '{index_colname}' on '{table_name}'", 3, enable=diagnose)

    index_sqlcol = sql_escape_str(index_colname)
    sql_tablename = sql_escape_str(table_name)
    
    try:
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{index_sqlcol} ON {sql_tablename}({index_sqlcol})")
    except sqlite3.OperationalError as err:
        if "already exists" in str(err):
            logs.sts(f"{logs.prog_loc()} index already exists, don't need to add it again.", 3)
        else:
            logs.sts(f"Unexpected Error: {err}.", 3)
            logs.error_beep()
            breakpoint() #perm
            pass
    except Exception as err:        
        logs.sts(f"Unexpected Error: {err}.", 3)
        logs.error_beep()
        breakpoint() #perm
        pass

    logs.sts(f"{logs.prog_loc()} Added index of col '{index_colname}' successfully.", 3, enable=diagnose)

def sum_columns_in_sqlite_table(table_name='tempdata', db_file_path=None):

    if db_file_path is None:
        db_file_path=f'{table_name}.db'
        
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Get the column names from the table
    cursor.execute(f"PRAGMA table_info({table_name})")
    column_info = cursor.fetchall()
    column_names = [col[1] for col in column_info]

    # Create a SQL query to calculate the sum for each column
    sum_queries = [f"SUM({col}) AS {col}" for col in column_names]
    query = f"SELECT {', '.join(sum_queries)} FROM {table_name}"

    # Execute the query and fetch the result
    cursor.execute(query)
    result = cursor.fetchone()

    # Close the database connection
    conn.close()

    # Convert the result into a dictionary
    if result:
        # Since column aliases are not supported, we need to manually alias the columns in the result
        sum_dict = dict(zip(column_names, result))
        return sum_dict
    else:
        return None

def get_memory_usage_of_table_in_memory(table_name='tempdata'):
    # Connect to an in-memory SQLite database
    conn = sqlite3.connect(table_name)
    cursor = conn.cursor()

    # Execute a PRAGMA statement to calculate the size of the table
    cursor.execute("PRAGMA page_size;")  # Get page size
    page_size = cursor.fetchone()[0]

    cursor.execute("PRAGMA page_count;")  # Get page count
    page_count = cursor.fetchone()[0]

    # Calculate the total size in bytes
    total_size_bytes = page_size * page_count

    # Close the database connection
    conn.close()

    return total_size_bytes


def print_table_summary(table_name='example', db_file_path=None):

    if db_file_path is None:
        db_file_path=f'{table_name}.db'
        
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Query the sqlite_master table to get table information
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table_name}'")
    table_info = cursor.fetchone()

    # Close the database connection
    conn.close()

    if table_info:
        print(f"Table '{table_name}' summary:")
        print(table_info[0])
    else:
        print(f"Table '{table_name}' not found.")



def sqlite_selectrow(table_name, key_col='rowkey', value='500'): 

    # Connect to the SQLite database
    conn = sqlite3.connect(f"{table_name}.db")
    cursor = conn.cursor()

    # Define the target rowkey value
    target_rowkey = value
    
    # Execute the SQL query to select the row with the specified rowkey
    cursor.execute(f"SELECT * FROM {table_name} WHERE {key_col}=?", (target_rowkey,))
    selected_row = cursor.fetchone()

    # Close the database connection
    conn.close()

    # If a row was found, create a dictionary from the row
    if selected_row:
        column_names = [description[0] for description in cursor.description]
        selected_dict = dict(zip(column_names, selected_row))
        return selected_dict
    else:
        # Return None if no matching row was found
        return None
        

