# test_pydf_vs_df

# copyright (c) 2024 Ray Lutz

import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
import timeit, os
import numpy as np
from pympler import asizeof
from collections import namedtuple
import sqlite3
import sys
import os

sys.path.append('..')

from Pydf.md_demo import pr, md_code_seg

global     datatable1, datatable2,sample_lod,sample_klod,df,kdf,pydf,kpydf,hdlol,hllol,hdnpa


""" Investigate the benefits to using df vs. lod or other simple data types. 

    Q: when is it worth changing a list-of-dict (lod) type to a df to do
        things like sum columns?
        i.e. to sum all columns one time, is it worth switching to df first?
                when does it pay off?
"""

def kdf_lookup(kdf, rowkey_col='rowkey', value='500'):
    # return dict(kdf[kdf[rowkey_col]==value].iloc[0])
    return kdf.loc[value].to_dict()
    
    # kdf.loc['500'].to_dict()


def lod_sum_cols_df(lod: List[Dict[str, int]], cols: Optional[List[str]] = None) -> Dict[str, int]:
    
    if not lod:
        return {}

    allcols = list(lod[0].keys())

    if cols is None:
        cols = allcols
    
    df = pd.DataFrame(lod)
    result = df[cols].sum().to_dict()
    
    for col in allcols:
        if col not in cols:
            result[col] = ''

    return result


def lod_sum_cols(lod: List[Dict[str, int]], cols: Optional[List[str]] = None) -> Dict[str, int]:

    if not lod:
        return {}
        
    allcols = list(lod[0].keys())

    if cols is None:
        cols = allcols
        
    result = dict.fromkeys(cols, 0)
    
    for row in lod:
        for col in cols:
            result[col] += row[col]

    for col in allcols:
        if col not in cols:
            result[col] = ''
    
    return result

    
def lod_sum_cols2(lod: List[Dict[str, int]], cols: Optional[List[str]] = None) -> Dict[str, int]:
    """ this one grabs a col and uses sum() """

    if not lod:
        return {}
        
    allcols = list(lod[0].keys())

    if cols is None:
        cols = allcols
        
    result = dict.fromkeys(cols, 0)
    
    for col in cols:
        #result[col] = sum([rowd[col] for rowd in lod])     # select an entire col 
        result[col] = sum(rowd[col] for rowd in lod)     # select an entire col

    for col in allcols:
        if col not in cols:
            result[col] = ''
    
    return result
    

def klod_row_lookup(klod, key_col='rowkey', target_rowkey='500'):

    for row in klod:
        if row.get(key_col) == target_rowkey:
            return row
    
    return None

    
def lont_sum_cols(lont, cols: Optional[List[str]] = None) -> Dict[str, int]:

    if not lont:
        return {}
        
    allcols = lont[0]._fields

    if cols is None:
        cols = allcols
        
    result = dict.fromkeys(cols, 0)
    
    for rownt in lont:
        for col in cols:
            result[col] += getattr(rownt, col)

    for col in allcols:
        if col not in cols:
            result[col] = ''
    
    return result
    
    
def lod_to_hdlol(lod: List[Dict[str, Any]]) -> Tuple[Dict[str, int], List[List[Any]]]:

    header_di = {col_name: index for index, col_name in enumerate(lod[0].keys())}
    lol = [list(d.values()) for d in lod]
    
    return (header_di, lol)
    
    
def lod_to_hllol(lod: List[Dict[str, Any]]) -> Tuple[List[str], List[List[Any]]]:

    header_ls = list(lod[0].keys())
    lol = [list(d.values()) for d in lod]
    
    return (header_ls, lol)
    
    
def lod_to_hdnpa(lod: List[Dict[str, Any]]):

    header_di = {col_name: index for index, col_name in enumerate(lod[0].keys())}
    npa = np.array([list(d.values()) for d in lod])
    
    return (header_di, npa)
    
    
def hdlol_sum_cols(hdlol: Tuple[Dict[str, int], List[List[Any]]], cols: Optional[List[str]] = None) -> List[int]:

    (header_di, lol) = hdlol

    if cols is None:
        cols_li = list(header_di.values())
    else:
        cols_li = [header_di[col] for col in cols]
        
    result_li = [0] * len(header_di)
    
    for row in lol:
        for coli in cols_li:
            result_li[coli] += row[coli]

    return result_li
    
    
def hllol_sum_cols(hllol: Tuple[List[str], List[List[Any]]], cols: Optional[List[str]] = None) -> Dict[str, int]:

    (header_ls, lol) = hllol

    if cols is None:
        cols_ls = header_ls
    else:
        cols_ls = cols
        
    result_di = {}
    
    for coli, col in enumerate(header_ls):
        if col not in cols_ls:
            continue
        sum_coli = 0    
        for row in lol:
            sum_coli += row[coli]
            
        result_di[col] = sum_coli

    return result_di
    
    
def hllol_sum_cols2(hllol: Tuple[List[str], List[List[Any]]], cols: Optional[List[str]] = None) -> Dict[str, int]:
    """ uses transpose """

    (header_ls, lol) = hllol
    # hd = {col: idx for idx, col in enumerate(header_ls)}
    
    # if cols:
    # icol_list = [hd[col] for col in cols]
    
    # result_di = {cols[icol]: sum(column) for icol, column in enumerate(zip(*lol_array)) if icol in icol_list}
    result_li = [sum(column) for column in zip(*lol)]

    # if cols is None:
        # cols_ls = header_ls
    # else:
        # cols_ls = cols
        
    # result_di = {}
    
    # for coli, col in enumerate(header_ls):
        # if col not in cols_ls:
            # continue
        # sum_coli = 0    
        # for row in lol:
            # sum_coli += row[coli]
            
        # result_di[col] = sum_coli

    return result_li
    
    
def hdlot_sum_cols(hdlot: Tuple[Dict[str, int], List[Tuple[Any, ...]]], cols: Optional[List[str]] = None) -> List[int]:

    (header_di, lot) = hdlot

    if cols is None:
        cols_li = list(header_di.values())
    else:
        cols_li = [header_di[col] for col in cols]
        
    result_li = [0] * len(header_di)
    
    for rowt in lot:
        for coli in cols_li:
            result_li[coli] += rowt[coli]

    return result_li
    
    
def hdlol_to_df(hdlol: Tuple[Dict[str, int], List[List[Any]]]):

    (header_di, data_lol) = hdlol
    columns = list(header_di.keys())
    df = pd.DataFrame(data_lol, columns=columns)
    
    return df
    
    
def lod_to_lont(lod):
    # Check if the list is not empty
    if not lod:
        return []
    # Extract the field names from the first dictionary
    field_names = list(lod[0].keys())
    
    # Define a named tuple type
    MyTuple = namedtuple('MyTuple', field_names)
    
    # Convert LOD to LOT
    lont = [MyTuple(**item) for item in lod]
    return lont


def lod_to_hdlot(lod):
    # Check if the list is not empty
    if not lod:
        return []
        
    header_di = {col_name: index for index, col_name in enumerate(lod[0].keys())}
    lot = [tuple(d.values()) for d in lod]
    
    return (header_di, lot)


def hdnpa_sum_cols(hdnpa, cols= None):
    
    (header_di, npa) = hdnpa

    if cols is None:
        cols_li = list(header_di.values())
    else:
        cols_li = [header_di[col] for col in cols]
        
    (rows, cols) = npa.shape
        
    result_li = [0] * cols
    
    for rowi in range(rows):
        for coli in cols_li:
            result_li[coli] += npa[rowi][coli]

    return result_li


def hdnpa_dotsum_cols(hdnpa, cols= None) -> List[float]:

    header_di, npa = hdnpa

    if cols is None:
        cols_li = list(header_di.values())
    else:
        cols_li = [header_di[col] for col in cols]

    result = np.sum(npa[:, cols_li], axis=0).tolist()

    return result    

def transpose_hdlol(hdlol):
    
    (header_di, original_lol) = hdlol

    transposed_lol = [[row[i] for row in original_lol] for i in range(len(original_lol[0]))]
    
    return (header_di, transposed_lol)


def transpose_hdlol2(hdlol):
    
    (header_di, original_lol) = hdlol

    transposed_lol = list(map(list, zip(*original_lol)))
    
    return (header_di, transposed_lol)
    
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
        cursor.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_rowkey_idx ON {table_name}({key_col})")

    # Insert data from the list of dictionaries into the table using parameterized queries
    for entry in lod:
        placeholders = ', '.join(['?'] * len(field_names))
        insert_query = f"INSERT INTO {table_name} ({', '.join(field_names)}) VALUES ({placeholders})"
        values = [entry[field] for field in field_names]
        cursor.execute(insert_query, values)

    # Commit changes and close the database connection
    conn.commit()
    conn.close()

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


# import sqlitepool

# Create a pool of database connections
#db_pool = sqlitepool.SimpleSQLitePool(f"{table_name}.db", maxconnections=5)

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


def main():
    from Pydf.Pydf import Pydf

    # Specify the number of columns you want
    num_columns = 1000      # You can change this number
    num_rows    = 1000      # Number of records
    global sample_lod
    global sample_klod
    global df
    global kdf
    global pydf
    global kpydf
    global hdlol
    global hllol
    global hdnpa
    global datatable1, datatable2
    
    md_report = pr("# Evaluate conversion and calculation time tradeoffs between Pandas, Pydf, Numpy, etc.\n\n")

    md_report += md_code_seg("Create sample_lod")
    """ Here we generate a table using a python list-of-dict structure,
        with 1000 columns and 1000 rows, consisting of
        integers from 0 to 100. Set the seed to an arbitrary value for
        reproducibility. Also, create other forms similarly or by converting
        the sample_lod.
    """
    np.random.seed(42)  # For reproducibility
    sample_lod = [dict(zip([f'Col{i}' 
                    for i in range(num_columns)], 
                        np.random.randint(0, 100, num_columns))) 
                            for _ in range(num_rows)]
                            
    # build a sizeof_di dict                        
    sizeof_di = {}
    sizeof_di['lod'] = asizeof.asizeof(sample_lod)
    md_report += pr(f"\n\nGenerated sample_lod with {len(sample_lod)} records\n"
                    f"- {sizeof_di['lod']=:,} bytes\n\n")

    md_report += md_code_seg("Create sample_klod")
    """ sample_klod is similar to sample_lod but it has a first column 
        'rowkey' is a string key that can be used to look up a row.
    """
    
    sample_klod = [dict(zip(['rowkey']+[f'Col{i}' 
                    for i in range(num_columns)], 
                        [str(i)] + list(np.random.randint(1, 100, num_columns)))) 
                            for i in range(num_rows)]

    sizeof_di['klod'] = asizeof.asizeof(sample_klod)
    md_report += pr(f"\n\nGenerated sample_klod with {len(sample_klod)} records\n"
                 f"- {sizeof_di['klod']=:,} bytes\n\n")

    md_report += md_code_seg("Create pydf from sample_lod")
    pydf = Pydf.from_lod(sample_lod)
    sizeof_di['pydf'] = asizeof.asizeof(pydf)
    md_report += pr(f"pydf:\n{pydf}\n\n"
                    f"{sizeof_di['pydf']=:,} bytes\n\n")

    md_report += md_code_seg("Create kpydf from sample_klod")
    kpydf = Pydf.from_lod(sample_klod, keyfield='rowkey')
    sizeof_di['kpydf'] = asizeof.asizeof(kpydf)
    md_report += pr(f"kpydf:\n{kpydf}\n\n"
                    f"{sizeof_di['kpydf']=:,} bytes\n\n")

    ## # create hdlol
    ## md_report += md_code_seg("Create hdlol from lod")
    ## hdlol = lod_to_hdlol(sample_lod)
    ## asizeof.asizeof(hdlol)

    ## create hllol
    ## print("creating hllol from lod")
    ## hllol = lod_to_hllol(sample_lod)

    md_report += md_code_seg("Create Pandas df")
    # Create a Pandas DataFrame
    # here we used an unadorned basic pre-canned method.
    df = pd.DataFrame(sample_lod, dtype=int)
    sizeof_di['df'] = asizeof.asizeof(df)

    md_report += pr(f"\n\n```{df}\n```\n\n"
                    f"{sizeof_di['df']=:,} bytes\n\n")

    md_report += md_code_seg("Create Pandas csv_df from Pydf thru csv")
    # Create a Pandas DataFrame by convering Pydf through a csv buffer.
    csv_df = pydf.to_pandas_df(use_csv=True)
    sizeof_di['csv_df'] = asizeof.asizeof(csv_df)

    md_report += pr(f"\n\n```{csv_df}\n```\n\n"
                    f"{sizeof_di['csv_df']=:,} bytes\n\n")

    md_report += md_code_seg("Create keyed Pandas df")
    """ Create a keyed Pandas df based on the sample_klod generated.
        Please note this takes far more memory than a Pandas df without this
        str column, almost 3x the size of Pydf instance. To test fast lookups,
        we also use set_index to get ready for fast lookups.
    """
    
    kdf = pd.DataFrame(sample_klod)
    
    # also set the rowkey as the index for fast lookups.
    kdf.set_index('rowkey', inplace=True)
    sizeof_di['kdf'] = asizeof.asizeof(kdf)
    md_report += pr(f"\n\n```{kdf}\n```\n\n")
    md_report += pr(f"- {sizeof_di['kdf']=:,} bytes\n\n")

    md_report += md_code_seg("create hdnpa from lod")
    hdnpa = lod_to_hdnpa(sample_lod)
    sizeof_di['hdnpa'] = asizeof.asizeof(hdnpa)
    md_report += pr(f"{sizeof_di['hdnpa']=:,} bytes\n\n")

    md_report += md_code_seg("Create lont from lod")
    lont = lod_to_lont(sample_lod)
    sizeof_di['lont'] = asizeof.asizeof(lont)
    md_report += pr(f"{sizeof_di['lont']=:,} bytes\n\n")

    md_report += md_code_seg("Create hdlot from lod")
    hdlot = lod_to_hdlot(sample_lod)
    sizeof_di['hdlot'] = asizeof.asizeof(hdlot)
    md_report += pr(f"{sizeof_di['hdlot']=:,} bytes\n\n")

    # create sqlite_table
    md_report += md_code_seg("Create sqlite_table from klod")
    lod_to_sqlite_table(sample_klod, table_name='tempdata1')

    datatable1 = 'tempdata1'
    datatable2 = 'tempdata2'
    sizeof_di['sqlite'] = os.path.getsize(datatable1+'.db')
    md_report += pr(f"{sizeof_di['sqlite']=:,} bytes\n\n")

    md_report += md_code_seg("Create table of estimated memory usage for all types")
    """ use Pydf.from_lod_to_cols to create a table with first colunm key names, and second column values. """
    all_sizes_pydf = Pydf.from_lod_to_cols([sizeof_di], cols=['Datatype' 'Size in Memory (bytes)'])
    md_report += all_sizes_pydf.to_md(smart_fmt=True)
    
    md_report += md_code_seg("Time conversions and operations")
    """ This secion uses the timeit() function to time conversions.
        For each convertion, the time wil be added to the (datatype)_times dicts.
    """

    setup_code = '''
import pandas as pd
import numpy as np
from collections import namedtuple
import sys
sys.path.append('..')
from Pydf.Pydf import Pydf

'''
    loops = 10
    pydf_times = {}
    pandas_times = {}
    numpy_times = {}
    sqlite_times = {}
    lod_times = {}
    loops_di = {}

    loops_di['from_lod']            = loops
    pydf_times['from_lod']          = timeit.timeit('Pydf.from_lod(sample_lod)',  setup=setup_code, globals=globals(), number=loops)
    print(f"Pydf.from_lod()             {loops} loops: {pydf_times['from_lod']:.4f} secs")

    pandas_times['from_lod']        = timeit.timeit('pd.DataFrame(sample_lod)',   setup=setup_code, globals=globals(), number=loops)
    lod_times['to_pandas_df']       = pandas_times['from_lod']
    print(f"lod_to_df() plain           {loops} loops: {pandas_times['from_lod']:.4f} secs")

    numpy_times['from_lod']         = timeit.timeit('lod_to_hdnpa(sample_lod)', setup=setup_code, globals=globals(), number=loops)
    lod_times['to_numpy']           = numpy_times['from_lod']
    print(f"lod_to_hdnpa()              {loops} loops: {numpy_times['from_lod']:.4f} secs")

    sqlite_times['from_lod']        = timeit.timeit('lod_to_sqlite_table(sample_klod, table_name=datatable2)', setup=setup_code, globals=globals(), number=loops)
    print(f"lod_to_sqlite_table()       {loops} loops: {sqlite_times['from_lod']:.4f} secs")

    loops_di['to_pandas_df']        = loops
    pydf_times['to_pandas_df']      = timeit.timeit('pydf.to_pandas_df()',        setup=setup_code, globals=globals(), number=loops)
    print(f"pydf.to_pandas_df()         {loops} loops: {pydf_times['to_pandas_df']:.4f} secs")

    loops_di['to_pandas_df_thru_csv']        = loops
    pydf_times['to_pandas_df_thru_csv']      = timeit.timeit('pydf.to_pandas_df(use_csv=True)', setup=setup_code, globals=globals(), number=loops)
    print(f"pydf.to_pandas_df(use_csv=True)  {loops} loops: {pydf_times['to_pandas_df_thru_csv']:.4f} secs")

    loops_di['from_pandas_df']      = loops
    pydf_times['from_pandas_df']    = timeit.timeit('Pydf.from_pandas_df(df)',    setup=setup_code, globals=globals(), number=loops)
    print(f"Pydf.from_pandas_df()       {loops} loops: {pydf_times['from_pandas_df']:.4f} secs")

    loops_di['to_numpy']            = loops
    pydf_times['to_numpy']          = timeit.timeit('pydf.to_numpy()',            setup=setup_code, globals=globals(), number=loops)
    print(f"pydf.to_numpy()             {loops} loops: {pydf_times['to_numpy']:.4f} secs")

    loops_di['from_numpy']          = loops
    pydf_times['from_numpy']        = timeit.timeit('Pydf.from_numpy(hdnpa[1])',  setup=setup_code, globals=globals(), number=loops)
    print(f"Pydf.from_numpy()           {loops} loops: {pydf_times['from_numpy']:.4f} secs")

    numpy_times['to_pandas_df']     = timeit.timeit('pd.DataFrame(hdnpa[1])',  setup=setup_code, globals=globals(), number=loops)
    pandas_times['from_numpy']      = numpy_times['to_pandas_df']
    print(f"numpy to pandas df          {loops} loops: {numpy_times['to_pandas_df']:.4f} secs")

    numpy_times['from_pandas_df']   = timeit.timeit('df.values',  setup=setup_code, globals=globals(), number=loops)
    pandas_times['to_numpy']        = numpy_times['from_pandas_df']
    print(f"numpy from pandas df          {loops} loops: {numpy_times['from_pandas_df']:.4f} secs")


    ## print(f"lod_to_hdlol()            {loops} loops: {timeit.timeit('lod_to_hdlol(sample_lod)', setup=setup_code, globals=globals(), number=loops):.4f} secs")
    ## print(f"lod_to_hllol()            {loops} loops: {timeit.timeit('lod_to_hllol(sample_lod)', setup=setup_code, globals=globals(), number=loops):.4f} secs")
    ## print(f"hdlol_to_df()             {loops} loops: {timeit.timeit('hdlol_to_df(hdlol)',       setup=setup_code, globals=globals(), number=loops):.4f} secs")
    ## print(f"lod_to_lont()              {loops} loops: {timeit.timeit('lod_to_lont(sample_lod)',  setup=setup_code, globals=globals(), number=loops):.4f} secs")
    ## print(f"lod_to_hdlot()             {loops} loops: {timeit.timeit('lod_to_hdlot(sample_lod)', setup=setup_code, globals=globals(), number=loops):.4f} secs")

    print("\nMutations:")

    loops_di['increment cell']      = loops * 100
    pydf_times['increment cell']    = timeit.timeit('pydf[500, 500] += 1', setup=setup_code, globals=globals(), number=loops_di['increment cell'])
    print(f"pydf[500, 500] += 1         {loops_di['increment cell']} loops: {pydf_times['increment cell']:.4f} secs")

    pandas_times['increment cell']  = timeit.timeit('df.at[500, "Col500"] += 1', setup=setup_code, globals=globals(), number=loops_di['increment cell'])
    print(f"df.at[500, 'Col500'] += 1   {loops_di['increment cell']} loops: {pandas_times['increment cell']:.4f} secs")

    loops_di['insert_irow']         = loops * 10
    pydf_times['insert_irow']       = timeit.timeit('pydf.insert_irow(irow=400, row_la=pydf[600, :].copy())', setup=setup_code, globals=globals(), number=loops_di['insert_irow'])
    print(f"pydf.insert_irow            {loops_di['insert_irow']} loops: {pydf_times['insert_irow']:.4f} secs")

    pandas_times['insert_irow']     = timeit.timeit('pd.concat([df.iloc[: 400], pd.DataFrame([df.loc[600].copy()]), df.iloc[400:]], ignore_index=True)', setup=setup_code, globals=globals(), number=loops_di['insert_irow'])
    print(f"df insert row               {loops_di['insert_irow']} loops: {pandas_times['insert_irow']:.4f} secs")

    loops_di['insert_icol']         = loops * 10
    pydf_times['insert_icol']       = timeit.timeit('pydf.insert_icol(icol=400, col_la=pydf[:, 600].copy())', setup=setup_code, globals=globals(), number=loops_di['insert_icol'])
    print(f"pydf.insert_icol            {loops_di['insert_icol']} loops: {pydf_times['insert_icol']:.4f} secs")

    pandas_times['insert_icol']     = timeit.timeit("pd.concat([df.iloc[:, :400], pd.DataFrame({'Col600_Copy': df['Col600'].copy()}), df.iloc[:, 400:]], axis=1)", setup=setup_code, globals=globals(), number=loops_di['insert_icol'])
    print(f"df insert col               {loops_di['insert_icol']} loops: {pandas_times['insert_icol']:.4f} secs")

    print("\nTime for sums:")

    loops_di['sum cols']            = loops
    pandas_times['sum cols']        = timeit.timeit('cols=list[df.columns]; df[cols].sum().to_dict()', setup=setup_code, globals=globals(), number=loops)
    print(f"df_sum_cols()               {loops} loops: {pandas_times['sum cols']:.4f} secs")

    pydf_times['sum cols']          = timeit.timeit('pydf.sum()', setup=setup_code, globals=globals(), number=loops)
    print(f"pydf.sum()                  {loops} loops: {pydf_times['sum cols']:.4f} secs")

    numpy_times['sum cols']         = timeit.timeit('hdnpa_dotsum_cols(hdnpa)', setup=setup_code, globals=globals(), number=loops)
    print(f"hdnpa_dotsum_cols()         {loops} loops: {numpy_times['sum cols']:.4f} secs")

    sqlite_times['sum cols']        = timeit.timeit('sum_columns_in_sqlite_table(table_name=datatable1)', setup=setup_code, globals=globals(), number=loops)
    print(f"sqlite_sum_cols()           {loops} loops: {sqlite_times['sum cols']:.4f} secs")

    lod_times['sum cols']           = timeit.timeit('lod_sum_cols(sample_lod)', setup=setup_code, globals=globals(), number=loops)
    print(f"lod_sum_cols()              {loops} loops: {lod_times['sum cols']:.4f} secs")

    loops_di['transpose']           = loops
    pandas_times['transpose']       = timeit.timeit('df.transpose()',           setup=setup_code, globals=globals(), number=loops)
    print(f"df.transpose()              {loops} loops: {pandas_times['transpose']:.4f} secs")

    pydf_times['transpose']         = timeit.timeit('pydf.transpose()',         setup=setup_code, globals=globals(), number=loops)
    print(f"pydf.transpose()            {loops} loops: {pydf_times['transpose']:.4f} secs")

    ##print(f"lod_sum_cols2()             {loops} loops: {timeit.timeit('lod_sum_cols2(sample_lod)',setup=setup_code, globals=globals(), number=loops):.4f} secs")
    ##print(f"lont_sum_cols()            {loops} loops: {timeit.timeit('lont_sum_cols(lont)',      setup=setup_code, globals=globals(), number=loops):.4f} secs")
    ##print(f"hdnpa_sum_cols()            {loops} loops: {timeit.timeit('hdnpa_sum_cols(hdnpa)',    setup=setup_code, globals=globals(), number=loops):.4f} secs")
    ## print(f"hdlol_sum_cols()          {loops} loops: {timeit.timeit('hdlol_sum_cols(hdlol)',    setup=setup_code, globals=globals(), number=loops):.4f} secs")
    ##print(f"hllol_sum_cols()          {loops} loops: {timeit.timeit('hllol_sum_cols(hllol)',    setup=setup_code, globals=globals(), number=loops):.4f} secs")
    ##print(f"hllol_sum_cols2()         {loops} loops: {timeit.timeit('hllol_sum_cols2(hllol)',    setup=setup_code, globals=globals(), number=loops):.4f} secs")
    ##print(f"hdlot_sum_cols()           {loops} loops: {timeit.timeit('hdlot_sum_cols(hdlot)',    setup=setup_code, globals=globals(), number=loops):.4f} secs")

    ## print(f"transpose_hdlol()         {loops} loops: {timeit.timeit('transpose_hdlol(hdlol)',   setup=setup_code, globals=globals(), number=loops):.4f} secs")
    ## print(f"transpose_hdlol2()        {loops} loops: {timeit.timeit('transpose_hdlol2(hdlol)',  setup=setup_code, globals=globals(), number=loops):.4f} secs")

    print("\nTime for lookups:")

    loops_di['keyed lookup']        = loops*10
    pydf_times['keyed lookup']      = timeit.timeit("kpydf.select_record_da('500')", setup=setup_code, globals=globals(), number=loops_di['keyed lookup'])
    print(f"kpydf row lookup            {loops_di['keyed lookup']} loops: {pydf_times['keyed lookup']:.4f} secs")

    pandas_times['keyed lookup']    = timeit.timeit("kdf.loc['500'].to_dict()", setup=setup_code, globals=globals(), number=loops_di['keyed lookup'])
    print(f"kdf row lookup (indexed)    {loops_di['keyed lookup']} loops: {pandas_times['keyed lookup']:.4f} secs")

    lod_times['keyed lookup']       = timeit.timeit('klod_row_lookup(sample_klod)', setup=setup_code, globals=globals(), number=loops_di['keyed lookup'])
    print(f"klod_row_lookup()           {loops_di['keyed lookup']} loops: {lod_times['keyed lookup']:.4f} secs")

    sqlite_times['keyed lookup']    = timeit.timeit('sqlite_selectrow(table_name=datatable1)', setup=setup_code, globals=globals(), number=loops_di['keyed lookup'])
    print(f"sqlite_row_lookup()         {loops_di['keyed lookup']} loops: {sqlite_times['keyed lookup']:.4f} secs")

    lod_times   = [pydf_times, pandas_times, numpy_times, sqlite_times, lod_times, loops_di]
    times_cols  = ['Attribute', 'Pydf', 'Pandas', 'Numpy', 'Sqlite', 'lod', '(loops)']

    report_pydf = Pydf.from_lod_to_cols(lod = lod_times, cols=times_cols)

    MB = 1024 * 1024

    report_pydf.append({'Attribute': 'Size of 1000x1000 array (MB)', 
                        'Pydf':     sizeof_di['pydf'] / MB, 
                        'Pandas':   sizeof_di['df'] / MB, 
                        'Numpy':    sizeof_di['hdnpa'] / MB, 
                        'Sqlite':   sizeof_di['sqlite'] / MB, 
                        'lod':      sizeof_di['lod'] / MB,
                        '(loops)':  '',
                        })

    report_pydf.append({'Attribute': 'Size of keyed 1000x1000 array (MB)', 
                        'Pydf':     sizeof_di['kpydf'] / MB, 
                        'Pandas':   sizeof_di['kdf'] / MB, 
                        'Numpy':    '' , 
                        'Sqlite':   '', 
                        'lod':      sizeof_di['klod'] / MB,
                        '(loops)':  '',
                        })

    md_report += report_pydf.to_md(smart_fmt = True, just = '>^^^^^')
    
    md_code_seg()
    # the above is required to conclude the last code segment.

    print ("\n\n" + md_report)

    sep = os.sep
    
    if sep == '/':
        md_path = '../docs/test_df_vs_pydf.md'
    else:
        md_path = '..\\docs\\test_df_vs_pydf.md'
    
    with open(md_path, 'w') as file:
        file.write(md_report)


    """
    estimated memory usage for 1000 records and 1000 columns
    lod:   124,968,872 bytes
    df:    9,767,776 bytes
    hdlol: 40,189,880 bytes
    hdnpa: 4,125,152 bytes
    lont:  40,048,872 bytes
    hdlot: 40,173,880 bytes

    Time for conversions:
    lod_to_df()         10 loops: 47.0079 secs
    lod_to_hdlol()      10 loops: 0.1808 secs
    hdlol_to_df()       10 loops: 49.6935 secs
    lod_to_hdnpa()      10 loops: 0.7467 secs
    lod_to_lont()       10 loops: 84.3453 secs
    lod_to_hdlot()      10 loops: 0.1823 secs

    Time for sums:
    df_sum_cols()       10 loops: 0.0544 secs
    lod_sum_cols()      10 loops: 1.4110 secs
    lont_sum_cols()     10 loops: 1.7490 secs
    hdnpa_sum_cols()    10 loops: 2.6609 secs
    hdnpa_dotsum_cols() 10 loops: 0.0299 secs
    hdlol_sum_cols()    10 loops: 0.9315 secs
    hdlot_sum_cols()    10 loops: 0.9669 secs
    transpose_hdlol()   10 loops: 1.2718 secs
    transpose_hdlol2()  10 loops: 0.9827 secs
    transpose_df()      10 loops: 0.0027 secs


        # regular python max and min are superior to pandas
        # pandas mean and stdev are superior (6x and 20x faster) to statistics functions.
        # should test with timeit
        
        # 2022-12-11 18:20:24.934378: Start pandas max  0.045466
        # 2022-12-11 18:20:24.979844: Start pandas min  0.046105
        # 2022-12-11 18:20:25.025949: Start pandas mean 0.05696
        # 2022-12-11 18:20:25.082909: Start pandas std  0.056037
        # 2022-12-11 18:20:25.138946: Start python min  0.010964
        # 2022-12-11 18:20:25.149910: Start python min  0.011031
        # 2022-12-11 18:20:25.160941: Start statistics mean   0.234966
        # 2022-12-11 18:20:25.395907: Start statistics stdev  1.008002
        # 2022-12-11 18:20:26.403909: end statistics stdev

        

    """
    """
    Creating test data:
    creating sample_lod with 1000 records and 1000 cols
    creating sample_klod with 1000 records with string key and 1000 additional cols
    converting to df from lod
    converting to kdf from klod
    creating hdlol from lod
    creating hllol from lod
    creating hdnpa from lod
    Creating lont from lod
    Creating hdlot from lod
    Creating sqlite_table from klod

    estimated memory usage for 1000 records and 1000 columns
    lod:   124,968,872 bytes
    klod:  125,024,928 bytes
    df:    9,767,776 bytes
    kdf:   9,890,112 bytes
    hdlol: 40,189,880 bytes
    hllol: 40,128,984 bytes
    hdnpa: 4,125,152 bytes
    lont:  40,048,872 bytes
    hdlot: 40,173,880 bytes
    sqlite:0 bytes (in memory)
    sqlite:5,148,672 bytes (on disk)

    Time for conversions:
    lod_to_df()         10 loops: 48.0653 secs
    lod_to_hdlol()      10 loops: 0.1403 secs
    lod_to_hllol()      10 loops: 0.1437 secs
    lod_to_hdnpa()      10 loops: 0.6339 secs
    lod_to_hdlot()      10 loops: 0.1543 secs
    lod_to_sqlite_table() 10 loops: 6.4056 secs

    Time for sums:
    df_sum_cols()       10 loops: 0.0869 secs
    lod_sum_cols()      10 loops: 1.2990 secs
    lod_sum_cols2()     10 loops: 3.2081 secs
    lont_sum_cols()     10 loops: 1.4774 secs
    hdnpa_sum_cols()    10 loops: 2.3331 secs
    hdnpa_dotsum_cols() 10 loops: 0.0240 secs
    hdlol_sum_cols()    10 loops: 0.8194 secs
    hllol_sum_cols()    10 loops: 1.0383 secs
    hdlot_sum_cols()    10 loops: 0.9704 secs
    sqlite_sum_cols()   10 loops: 2.8196 secs
    transpose_hdlol()   10 loops: 1.1439 secs
    transpose_hdlol2()  10 loops: 0.8502 secs
    transpose_df()      10 loops: 0.0025 secs

    Time for lookups:
    kdf row lookup      100 loops: 0.1148 secs
    klod_row_lookup()   100 loops: 0.0061 secs
    sqlite_row_lookup() 100 loops: 0.3037 secs
    """

    r"""
    The following uses Pandas 2.1

    (myenv) PS C:\Users\raylu\Documents\Github\audit-engine\test_data> python test_df_vs_lod_summing.py

    Creating test data:
    creating sample_lod with 1000 records and 1000 cols
    creating sample_klod with 1000 records with string key and 1000 additional cols
    converting to df from lod
    converting to kdf from klod
    creating hdlol from lod
    creating hllol from lod
    creating hdnpa from lod
    Creating lont from lod
    Creating hdlot from lod
    Creating sqlite_table from klod

    estimated memory usage for 1000 records and 1000 columns
    lod:   124,968,872 bytes
    klod:  125,024,928 bytes
    df:    9,711,456 bytes
    kdf:   9,833,016 bytes
    hdlol: 40,189,880 bytes
    hllol: 40,128,984 bytes
    hdnpa: 4,125,152 bytes
    lont:  40,048,872 bytes
    hdlot: 40,173,880 bytes
    sqlite:0 bytes (in memory)
    sqlite:5,148,672 bytes (on disk)

    Time for conversions:
    lod_to_df()         10 loops: 58.2945 secs

    """

    # reran pandas 2.1  Now see ~6% improvement.

    # hdlol_sum_cols()    10 loops: 0.8194 secs
    # df_sum_cols()       10 loops: 0.0869 secs
    #  benefit of df: 0.73 sec

    # lod_to_df()         10 loops: 44.7439 secs
    # lod_to_hdlol()      10 loops: 0.1403 secs
    # initialization Overhead of df: 44.6

    # number of operations to make it worth using df:
    # 44.6/0.73 = 61 matrix operations.

    # search:
    # kdf row lookup      100 loops: 0.1148 secs
    # klod_row_lookup()   100 loops: 0.0061 secs

    # dataframes are not good for looking up records
    # compared to keyed lod, which uses a hash.



    """
    lod_to_df() plain     10 loops: 44.7439 secs
    lod_to_df() dtype=int 10 loops: 44.3800 secs

    df:    9,711,456 bytes

    """

    # pandas 1.5.3

    """
    lod_to_df() plain     10 loops: 47.4442 secs
    lod_to_df() dtype=int 10 loops: 47.4183 secs

    df:    9,767,776 bytes

    """

if __name__ == "__main__":

    main()