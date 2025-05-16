# daf_demo.py

# copyright (c) 2024 Ray Lutz

import sys
import os

# sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'src'))
# Get the path to the 'src' directory
src_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Append the path only if it's not already in sys.path
if src_path not in sys.path:
    sys.path.append(src_path)


#from lib import daf_utils as utils 
from daffodil.lib.md_demo import md_code_seg

def main():

    md_report = "# Daffodil Demo\n\n"
    
    md_report += """ Daffodil is a simple and flexible dataframe package for use with Python.
This page will demonstrate the functionality of Daffodil by showing actual code and results of 
running that code. Daffodil is a good match for many common algorithms in data pipelines and other conversion use cases.
    
For more information about Daffodil, see [https://github.com/raylutz/Daffodil/blob/main/README.md]()

This page is the result of using simple "notebook" functionality (md_demo.py)
which will create a markdown "notebook" report by printing a code block and then run and capture the result. The report can
be viewed directly or converted to HTML for use on a static website.

"""

    md_report += md_code_seg('Create a new empty table')
    from daffodil.daf import Daf
    
    my_daf = Daf()
    
    md_report += f"The empty my_daf:\n{my_daf}\n"
    md_report += f"{bool(my_daf)=}\n"
    
    # note here that testing my_daf produces "False" if there is nothing in the array,
    # even though the instance itself exists.
    
    md_report += md_code_seg('Append some rows to the daf object') + "\n" 
    # here we append dictionaries to the daf array.
    # The column header is determined from the first dictionary added if it
    # is not otherwise initialized.
    
    my_daf.append({'A': 1,  'B': 2, 'C': 3})  
    my_daf.append({'A': 5,  'C': 7, 'B': 6})  
    my_daf.append({'C': 10, 'A': 8, 'B': 9})
    
    md_report += f"The appended my_daf:\n{my_daf}\n"
    # notice that when each row is appended, the columns are respected,
    # even if the data is provided in a different order.

    md_report += md_code_seg('Subset rows, cols using row and cols using indices:') + "\n" 
    """ Square brackets index using row, col syntax, using normal Python indexing
        using explicit numbering, slices, list of indices, string names, ranges
        of sring names, or list of string names. Row selection is fast because
        no additional copies are made of the data, whereas column dropping is not. 
        Although available, subsetting columns in this manner is
        relatively non-performant because a copy must be made, and is not recommended.
        Instead, indicate active columns in apply() and other functions using cols=[] parameter,
        and leave columns as-is.
    """    
   
    md_report += f"First 2 rows (and all columns) by slicing:\nmy_daf[0:2] = \n{my_daf[0:2]}\n"
    
    md_report += f"Row 0 and 2 (and all columns) using a list:\nmy_daf[[0,2]] = \n{my_daf[[0,2]]}\n"

    md_report += f"Just row 1:\nmy_daf[1] = \n{my_daf[1] = }\n"
    
    md_report += f"First 2 cols by slicing:\nmy_daf[:, 0:2] = \n{my_daf[:, 0:2]}\n"
    
    md_report += f"Columns 0 and 2 using a list:\nmy_daf[:, [0,2]] = \n{my_daf[:, [0,2]]}\n"

    md_report += f"Just col 1:\nmy_daf[:, 1] = \n{my_daf[:, 1]}\n"
    

    md_report += md_code_seg('Read and write individual cells by row,col indices') + "\n" 
    """ replace value at row 2, col 1 (i.e. 9) with value from row 1, col 0 (i.e. 5)
        multiplied by the value in cell [2,2] (i.e. 10) resulting in 50 at [2,1].
        Note that row and column indices start at 0, and are in row, col order (not x,y).
    """
    
    my_daf.retmode = 'val'
    
    my_daf[2,1] = my_daf[1,0] * my_daf[2,2]

    md_report += f"The modified my_daf:\n{my_daf}\n"

    md_report += md_code_seg('Read columns and rows') + "\n" 

    # when accessing a column or row using indices will return a list.
    # Columns can be indexed by number or by column name, which must be a str.
    
    col_2 = my_daf[:,2]
    row_1 = my_daf[1]
    col_B = my_daf[:,'B']

    md_report += f"- {col_2=}\n- {row_1=}\n- {col_B=}\n"    

    md_report += md_code_seg('Read rows and columns using methods') + "\n" 
    # when using methods to access: columns are returned as lists, 
    # and rows are returned as dicts.
    
    col_2 = my_daf.icol(2)
    row_1 = my_daf.irow(1)
    col_B = my_daf.col('B')

    md_report += f"- {col_2=}\n- {row_1=}\n- {col_B=}\n"
    
    md_report += md_code_seg('Insert a new column "Category" on left, and make it the keyfield') + "\n" 
    """ Rows in a Daf instance can be indexed using an existing column, by specifying that column as the keyfield.
        This will create the keydict kd which creates the index from each value. It must have unique hashable values. 
        The keyfield can be set at the same time the column is added.
        The key dictionary kd is maintained during all daf manipulations.
        A daf generated by selecting some rows from the source daf will maintain the same keyfield, and .keys()
        method will return the subset of keys that exist in that daf.
    """

    # Add a column on the left (icol=0) and set it as the keyfield.

    my_daf.insert_icol(icol=0, 
            col_la=['house', 'car', 'boat'], 
            colname='Category')
            
    my_daf.set_keyfield('Category')

    md_report += f"my_daf:\n{my_daf}\n"
    

    md_report += md_code_seg('Select a record by the key:') + "\n"
    """ Selecting one record by the key will return a dictionary.
    """

    da = my_daf.select_record('car')

    md_report += f"Result:\n\n- {da=}\n"
    

    md_report += md_code_seg('Append more records from a lod') + "\n"
    """ When records are appended from a lod (list of dict), they are appended as rows,
    the columns are respected, and the kd is updated. Using a daf
    instance is about 1/3 the size of an equivalent lod because the dictionary
    keys are not repeated for each row in the array.
    """

    lod = [{'Category': 'mall',  'A': 11,  'B': 12, 'C': 13},
           {'Category': 'van',   'A': 14,  'B': 15, 'C': 16},
           {'A': 17,  'C': 19, 'Category': 'condo', 'B': 18},
          ]

    my_daf.append(lod)  
    
    md_report += f"The appended my_daf:\n{my_daf}\n"

    md_report += md_code_seg('Update records') + "\n"
    """ updating records mutates the existing daf instance, and works
        like a database table. The keyvalue in the designated keyfield
        determines which record is updated. This uses the append()
        method because appending respects the keyfield, if it is defined.
    """
    
    lod = [{'Category': 'car',  'A': 25,  'B': 26, 'C': 27},
           {'Category': 'house', 'A': 31,  'B': 32, 'C': 33},
          ]

    my_daf.append(lod)  
    
    md_report += f"The updated my_daf:\n{my_daf}\n"

    md_report += md_code_seg('Add a column "is_vehicle"') + "\n" 
    my_daf.insert_col(colname='is_vehicle', col_la=[0,1,1,0,1,0], icol=1)

    md_report += f"The updated my_daf:\n{my_daf}\n"
    
    md_report += md_code_seg('daf bool') + "\n" 
    """ For daf, bool() simply determines if the daf exists and is not empty.
        This functionality makes it easy to use `if daf:` to test if the
        daf is not None and is not empty. This does not evaluate the __content__
        of the array, only whether contents exists in the array. Thus,
        an array with 0, False, or '' still is regarded as having contents.
        In contrast, Pandas raises an error if you try: ```bool(df)```.
        Normally, a lol structure that has an internal empty list is True,
        i.e. `bool([[]])` will evaluate as true while `bool(Daf(lol=[[]]))` is False. 
    """
    md_report += f"- {bool(my_daf)=}\n"
    md_report += f"- {bool(Daf(lol=[]))=}\n"
    md_report += f"- {bool(Daf(lol=[[]]))=}\n"
    md_report += f"- {bool(Daf(lol=[[0]]))=}\n"
    md_report += f"- {bool(Daf(lol=[['']]))=}\n"
    md_report += f"- {bool(Daf(lol=[[False]]))=}\n\n"
    
    md_report += md_code_seg('daf attributes') + "\n" 
    md_report += f"- {len(my_daf)=}\n"
    md_report += f"- {my_daf.len()=}\n"
    md_report += f"- {my_daf.shape()=}\n"
    md_report += f"- {my_daf.columns()=}\n"
    md_report += f"- {my_daf.keys()=}\n"
    
    md_report += md_code_seg('get_existing_keys') + "\n"
    """ check a list of keys to see if they are defined in the daf instance 
    """
    existing_keys_ls = my_daf.get_existing_keys(['house', 'boat', 'RV'])
    md_report += f"- {existing_keys_ls=}\n"
    
    md_report += md_code_seg('select_records_daf') + "\n"
    """ select multiple records using a list of keys and create a new daf instance. 
        Also orders the records according to the list provided.
    """
    wheels_daf = my_daf.select_records_daf(['van', 'car'])
    md_report += f"wheels_daf:\n{wheels_daf}\n"

    md_report += md_code_seg('select_by_dict') + "\n"
    """ select_by_dict offers a way to select for all exact matches to dict,
        or if inverse is set, the set that does not match.
    """
    vehicles_daf  = my_daf.select_by_dict({'is_vehicle':1})
    buildings_daf = my_daf.select_by_dict({'is_vehicle':0})
    # or
    buildings_daf = my_daf.select_by_dict({'is_vehicle':1}, inverse=True)

    md_report += f"vehicles_daf:\n{vehicles_daf}\nbuildings_daf:\n{buildings_daf}\n"

    md_report += md_code_seg("use `select_where` to select rows where column 'C' is over 20") + "\n" 
    high_c_daf = my_daf.select_where(lambda row: bool(row['C'] > 20))

    md_report += f"high_c_daf:\n{high_c_daf}\n"

    md_report += md_code_seg("convert to pandas DataFrame") + "\n" 
    my_df = my_daf.to_pandas_df()

    md_report += f"\nConverted DataFrame:\n```\n{my_df}\n```\n"

    md_report += md_code_seg("Add index column 'idx' to the dataframe at the left, starting at 0.") + "\n" 
    my_daf.insert_idx_col(colname='idx') #, icol=0, startat=0)

    md_report += f"\nModified daf:\n{my_daf}\n\n"

    # md_report += md.md_code_seg("convert from pandas DataFrame") + "\n" 
    # recovered_daf = Daf.from_pandas_df(my_df, keyfield='Category')

    # md_report += f"\nConvert Back:\n```{recovered_daf}```\n"

    md_report += md_code_seg("Create a table of file information") + "\n"
    """This example demonstrates how easy it is to create a Daf structure instead of a 
        more conventional list-of-lists structure. As a result, it is 1/3 the size, and
        offers more processing capabilities. Here, we create a handy file list including
        all relevant information by incrementally appending to the daf structure.

Please note that if you try to append dictionaries to a Pandas df, you will currently get this warning:

        FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    
To use this with Pandas, it is necessary to first build the array using another structure, such as a list of dictionaries,
    (lod), and then convert to df in one blow.

    """

    #import os
    import datetime
    import platform
    #from pympler import asizeof
    import objsize    
    
    def get_one_file_infodict(dirpath: str, filename: str, timespec='seconds') -> dict:
        path = os.path.join(dirpath, filename)
        stat = os.stat(path)
        file_info_dict = {
            #'filename': filename,
            'filepath': path,
            'size': stat.st_size,
            'modified_timestamp': datetime.datetime.fromtimestamp(stat.st_mtime
                                    ).isoformat(sep='T', timespec=timespec),
            'is_dir': os.path.isdir(path),
            }
        return file_info_dict
    
    
    def get_file_list_daf(dirpath: str, recursive:bool=False, timespec='seconds'):
        
        # create an empty daf.
        file_list_daf = Daf()
        
        # we may encounter permission errors.
        try:
            filelist = os.listdir(dirpath)
        except Exception:
            return file_list_daf
        
        for filename in filelist:
            file_info_dict = get_one_file_infodict(dirpath, filename, timespec=timespec)
            if file_info_dict['is_dir'] and recursive:
                subdir_path = file_info_dict['filepath']
                subdir_daf = get_file_list_daf(subdir_path, timespec=timespec)
                file_list_daf.append(subdir_daf)
            else:    
                file_list_daf.append(file_info_dict)
            
        return file_list_daf

    system = platform.system()

    if system == 'Darwin':
        dirpath = '/System'
    elif system == 'Windows':
        dirpath = 'C:\\Windows\\System32'
    elif system == 'Linux':
        dirpath = '/usr/bin'
    else:
        dirpath = ''

    if dirpath:
        os_system_file_list_daf = get_file_list_daf(dirpath, recursive=True)
        
        # shorten down the 'filepath' column to remove the redundant prefix.
        os_system_file_list_daf.retmode = 'val'
        
        os_system_file_list_daf.apply_to_col(col='filepath', func=lambda x: x.removeprefix(dirpath))

        md_report += f"\nContents of {dirpath}:\n" + \
                     os_system_file_list_daf.to_md(
                            just='<><<', 
                            max_text_len=80,
                            max_rows=30,
                            include_summary=True,
                            ) + "\n\n"
        
        md_report += f"\nContents of {dirpath} (in raw text format):\n```\n" + \
                     os_system_file_list_daf.to_md(
                            just='<><<', 
                            max_text_len=80,
                            max_rows=30,
                            include_summary=True,
                            ) + "\n```\n\n"
            
                            
                            
        md_report += f"- daf size in memory: {objsize.get_deep_size(os_system_file_list_daf):,} bytes\n"                    
        md_report += f"- pandas df size in memory: {objsize.get_deep_size(os_system_file_list_daf.to_pandas_df()):,} bytes\n"

    md_report += md_code_seg("Limit this list to just the files") + "\n"
    """ Now what we will do is first limit the listing only to files.
        Notice also that we are using custom justification, limiting text length to 50 chars (while keeping the ends)
        including 30 rows, consisting of the first 15 and the last 15, and including the summary at the end.
    """
    files_only_daf = os_system_file_list_daf.select_by_dict({'is_dir': False})

    md_report += f"\nFiles only in {dirpath}:\n" + \
                 files_only_daf.to_md(
                        just='<><<', 
                        max_text_len=80,
                        max_rows=30,
                        include_summary=True,
                        ) + "\n\n"
    
    md_report += md_code_seg("Demonstration of groupby_cols_reduce") + "\n"
    """ Given a daf, break into a number of daf's based on values in groupby_colnames. 
        For each group, apply func. to data in reduce_cols.
        returns daf with one row per group, and keyfield not set.
        
This can be commonly used when some colnames are important for grouping, while others
        contain values or numeric data that can be reduced.
        
For example, consider the data table with the following columns:
        
            gender, religion, zipcode, cancer, covid19, gun, auto
            
The data can be first grouped by the attribute columns gender, religion, zipcode, and then
        then prevalence of difference modes of death can be summed. The result is a daf with one
        row per unique combination of gender, religion, zipcode. Say we consider just M/F, C/J/I, 
        and two zipcodes 90001, and 90002, this would result in the following rows, where the 
        values in paranthesis are the reduced values for each of the numeric columns, such as the sum.
        
In general, the number of rows is reduced to no more than the product of number of unique values in each column
        grouped. In this case, there are 2 genders, 3 religions, and 2 zipcodes, resulting in
        2 * 3 * 2 = 12 rows.
        
        """
        
    groupby_colnames = ['gender', 'religion', 'zipcode']
    reduce_colnames  = ['cancer', 'covid19', 'gun', 'auto']
        
    cols = ['gender', 'religion', 'zipcode', 'cancer', 'covid19', 'gun', 'auto']
    lol = [
        ['M', 'C', 90001,  1,  2,  3,  4],
        ['M', 'C', 90001,  5,  6,  7,  8],
        ['M', 'C', 90002,  9, 10, 11, 12],
        ['M', 'C', 90002, 13, 14, 15, 16],
        ['M', 'J', 90001,  1,  2,  3,  4],
        ['M', 'J', 90001, 13, 14, 15, 16],
        ['M', 'J', 90002,  5,  6,  7,  8],
        ['M', 'J', 90002,  9, 10, 11, 12],
        ['M', 'I', 90001, 13, 14, 15, 16],
        ['M', 'I', 90001,  1,  2,  3,  4],
        ['M', 'I', 90002,  4,  3,  2,  1],
        ['M', 'I', 90002,  9, 10, 11, 12],
        ['F', 'C', 90001,  4,  3,  2,  1],
        ['F', 'C', 90001,  5,  6,  7,  8],
        ['F', 'C', 90002,  4,  3,  2,  1],
        ['F', 'C', 90002, 13, 14, 15, 16],
        ['F', 'J', 90001,  4,  3,  2,  1],
        ['F', 'J', 90001,  1,  2,  3,  4],
        ['F', 'J', 90002,  8,  7,  6,  5],
        ['F', 'J', 90002,  1,  2,  3,  4],
        ['F', 'I', 90001,  8,  7,  6,  5],
        ['F', 'I', 90001,  5,  6,  7,  8],
        ['F', 'I', 90002,  8,  7,  6,  5],
        ['F', 'I', 90002, 13, 14, 15, 16],
        ]
        
    data_table_daf = Daf(cols=cols, lol=lol)
        
    md_report += f"\nOriginal data_table_daf:\n{data_table_daf.to_md(include_summary=True)}\n\n"
    
    md_report += md_code_seg("Now reduce the data using groupby_cols_reduce") + "\n"
        
    grouped_and_summed_daf = data_table_daf.groupby_cols_reduce(
        groupby_colnames    = groupby_colnames,     # columns used for groups
        reduce_cols         = reduce_colnames,      # columns included in the reduce operation.
        func                = Daf.sum_da,
        by                  = 'row',                # determines how the func is applied. sum_da is by row.
        )

    expected_lol = [
        ['M', 'C', 90001,  6,  8, 10, 12],
        ['M', 'C', 90002, 22, 24, 26, 28],
        ['M', 'J', 90001, 14, 16, 18, 20],
        ['M', 'J', 90002, 14, 16, 18, 20],
        ['M', 'I', 90001, 14, 16, 18, 20],
        ['M', 'I', 90002, 13, 13, 13, 13],
        ['F', 'C', 90001,  9,  9,  9,  9],
        ['F', 'C', 90002, 17, 17, 17, 17],
        ['F', 'J', 90001,  5,  5,  5,  5],
        ['F', 'J', 90002,  9,  9,  9,  9],
        ['F', 'I', 90001, 13, 13, 13, 13],
        ['F', 'I', 90002, 21, 21, 21, 21],
        ]

    md_report += f"\nResulting Reduction:\n{grouped_and_summed_daf.to_md(include_summary=True)}\n\n"
    
    md_report += f"\nCheck the result against manually generated:\n{bool(grouped_and_summed_daf.lol==expected_lol)=}\n"
    
    md_report += md_code_seg("Further group to just zipcodes") + "\n"
    """ Now further reduce the grouped and summed table to provide the sum for just zipcodes, for all
    genders and religions. By producing the initial table with all combinations reduced, further grouping
    can be done without processing the entire table again.
    In this example, we also demonstrate using NumPy to sum the columns.
    
We find in this case that sum_np is not as efficient as just summing the rows directly.
    This is because it is a lot of work just to prepare the data for NumPy to read it.
    Summing by rows is about 22x faster (takes about 4% of the time) than using sum_np
    when tested on 40,288 records grouped into 820 groups. Grouping took 4 seconds,
    and summing took 8 seconds using row-based summation vs. using NumPy, which took 180 seconds.
    In other cases, using sum_np can be 3 times faster, and is particularly attractive if there 
    is a lot of column-based calculations involved.
    """
    
    zipcodes_daf = grouped_and_summed_daf.groupby_cols_reduce(
        groupby_colnames    = ['zipcode'], 
        func                = Daf.sum_np,
        by                  = 'table',                          # determines how the func is applied.
        reduce_cols         = reduce_colnames,                  # columns included in the reduce operation.
        )

    md_report += f"\nResults for zipcode Reduction:\n{zipcodes_daf.to_md(include_summary=True)}\n\n"
    
    
        

    md_code_seg()    # end marker
    #===================================

    print(md_report)
    
    sep = os.sep
    
    md_path = 'docs/daf_demo.md'

    with open(md_path, 'w') as file:
        file.write(md_report)


if __name__ == '__main__':
    main()    