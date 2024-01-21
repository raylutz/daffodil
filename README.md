# Pydf -- Python Dataframes

The Pydf package provides a lightweight, simple and fast 2-d dataframes, similar to Pandas.

Pydf is particularly well suited to applications for data munging, incremental appending, data pipelines,
and chunked large data sets where column and array based number crunching does not dominate.

It excels when the data array needs to be heavily manipulated, particularly by rows or individual data items.
Spreadsheet-like operations are also provided, which are useful for processing the entire array with the same formula template,
and can avoid glue code for many transformations. Python equations in the formula pane operate on the data
pane and calculations from spreadsheet programs can be easily ported in, to avoid random glue code.
    
## Fast for general data operations

We were surprised to find that Pandas is very slow in converting data to a Pandas DataFrame.
Pandas uses a numpy array for each column which must be allocated in memory as one contiguous block,
and apparently there is overhead to coerce the data types. The delay may also be to provide a great deal of
user and security protections.

Pydf datatype is based on list-of-list array, and uses a dictionary for column names and for row keys, 
making it extremely fast for column and row indexing by names, while avoiding the requirement for 
contiguous data allocation.

Pydf is not as performant as Pandas or numpy for numerical operations such as sums, max, min, stdev, etc. 
when the data is uniform across columns or the entire array. However, converting to Pandas dataframe is 
extremely slow, taking about 350x longer than conversions to Pydf.

Pandas df can be more performant if data is fixed but then manipulated by column at least ~30x, 
otherwise pydf will be faster. Unlike pandas df, pydf supports fast appends, extends and concats of rows 
and columns without producing an entire new pandas dataframe object.

Appending rows in Pandas is slow because Pandas DataFrames are designed to be column-oriented, meaning that 
operations involving columns are more efficient than operations involving rows. Each column is stored as a 
separate NumPy array, and appending a row involves creating a new array for each column with the added row. 
This process can lead to significant overhead, especially when dealing with large DataFrames.

Pydf can work well with Pandas and Numpy when number crunching and array-based operations are needed.
Use Pydf to build the array incrementally using row-based operations, then read the data by Numpy or
Pandas if the data is not uniform enough. Using Numpy for data crunching is preferred because getting
the data into Numpy is much faster.

Pydf is pure python and can be run with no other packages installed. If Pandas is not used, start up time is 
improved dramatically. This can be very important in cloud-based parallel processing where every millsecond
is billed. If conversions to or from Pandas is not required, then that package is not needed.

## Memory Footprint

A Pydf object is about 1/3 the size of a lod (list of dicts) because the column names are not repeated.
However it takes about 4x more memory than a minimal Pandas dataframe and 10x more memory than single numpy array.
Yet, sometimes Pandas will be much larger when strings are included in the data.

Thus, Pydf is a compromise. It is not as wasteful as commonly used list-of-dict for such tables, and 
is a good choice when rapid appends, inserts, row-based operations, and other mutation is required.
        
## Supports incremental appends

Pydf can append one or multiple rows, cols or another Pydf object extremely quickly, because it leverages Python's
data referencing approach. If 'use_copy' is True, then a new deep copy is also made of the data, but this can be
largely avoided in most data manipulation pipelines.

A list of dictionaries is an easy to use data structure that mimics a 2-D array but it is
expensive in space (3x larger than Pydf object) because the field names are repeated for each row. Also, it does not
offer array manipulation primitives offered by Pydf.

## Can work with Pandas, Numpy, Sqlite
    
Sometimes, Pandas or Numpy will be a good choice for array processing. In such cases, Pydf can be used to
build the data structure and then port it to Pandas or Numpy. We must warn the user, however, that converting
the data into Pandas has substantial overhead of about 360x compared with creating a copy of a Pydf structure.

A Pydf object is faster to create from a csv data file, either from local disk or cloud storage.
It can be used in those cases when data is to be manipulated in record-based systems where column-based math
operations are not used very often. Simple searches, grouping, and other operations are very fast but usually
can't match the performance of Pandas or Numpy once a dataframe is constructed. Yet for most use-cases, the
benefit once in Pandas does not out-weigh the cost of getting there.
  
## Column names

Similar to Pandas and other dataframe concepts, Pydf has a separate set of column names that can be optionally
used to name the columns. This is organized internally as a Python dictionary for fast column lookups by name.
There are no restrictions on column names except to avoid double underscore "__" in the names, which is used 
to allow arbitrary names in SQLite. Also, please be aware that SQLite is case insensitive while Pydf is generally 
sensitive to case. Other than that any characters, spaces and punctuation can be used. However, avoid 
spaces and punctuation if possible and use underscore ("_") to separate characters into groups.
    
When reading CSV files, the header is normally taken from the first (non-comment) line. However, you may be 
working with CSV files without a header. If user_format is specified on reading csv files, the csv data will
be pre-processed and "comment" lines starting with # removed.

Set noheader=True to avoid creating a header line from csv input, so column names need not exist.
Otherwise, column names are taken from first (non-comment) line and must be unique (or they are amended to be so.)
If there are columns without any names provided, then these are named "UnnamedN" where N is the column number staring at 0. If there
are duplicates, then the duplicate name is named "duplicatename_N", where 'duplicatename' is the original name
and N is the column number.

Even if column names are established, Pydf still allows that columns can be indexed by number. Therefore, it is best if the column
names used are not pure numbers, to avoid confusion. Traditional column names can be added with .add_AZ_cols() function, 
similar to spreadsheet programs, `'A', 'B', ... 'Z', 'AA', 'AB'...`. The fact that columns can be both indexed by number and
by name is an improvement over Pandas which allows only that the column names can be used, although they can be numeric.
    
## Row keyfield   
    
In many cases, one of the columns can be used as a unique key for locating records. If such a keyfield exists, it 
can be adopted as the primary index of the table by specifying that column name as the keyfield.
    
If keyfield is set, then that column must be a hashable type and must have unique values. Searches of row entries based on the keyfield
will use dictionary lookups, which are highly optimized by Python. Column names are also in a dictionary structure to speed row and column lookups by name.

Creating a key index does not remove that field from the data array, but creates an additional key dictionary, kd.

When adopting a file that may have a keyfield, it will be best to follow the following steps:
1. Set keyfield='' to turn off the key indexing functionality.
2. check that the proposed keyfield column has only unique values
3. remove, delete, or otherwise deal with records with duplicate keys so the keys are unique.
4. Use method .set_keyfield(keyfield) to set the keyfield and build the lookup dictionary.

Only one keyfield is supported.
    
## Common Usage Pattern
       
One common usage pattern allows iteration over the rows and appending to another instance. For example:
    
        # read csv file into 2-d array, handling column headers and unflattening
        my_pydf = Pydf.from_csv(file_path, unflatten=True)  
    
        # create a new table to be built as we can the input.
        new_pydf = Pydf()
        
        # scan the input my_pydf row by row and construct the output. Pandas can't do this efficiently.
        for row in my_pydf:  
            new_row = transform_row(row)
            new_pydf.append(new_row)
            
        new_pdf.to_csv(file_path, flatten=True)     # create a flat csv file with any python objects flattened using JSON.    
    
## Methods and functionality
       
### print and markdown reports

produce convenient form with reduced rows and cols similar to Pandas.

    print(instance_of_pydf)

TODO: The number of rows and cols to be printed in this form default to 10 but should be changeable in the class.

#### create a Markdown table from a pydf instance that can be incorporated in reports.
    my_pydf.md_pydf_table() 

        parameters:
            max_rows:       int     = 0,         # limit the maximum number of row by keeping leading and trailing rows.
            max_cols:       int     = 0,         # limit the maximum number of cols by keeping leading and trailing cols.
            just:           str     = '',        # provide the justification for each column, using <, ^, > meaning left, center, right justified.
            shorten_text:   bool    = True,      # if the text in any field is more than the max_text_len, then shorten by keeping the ends and redacting the center text.
            max_text_len:   int     = 80,        # see above.
            smart_fmt:      bool    = False,     # if columns are numeric, then limit the number of figures right of the decimal to "smart" numbers.
            include_summary: bool   = True,      # include a pandas-like summary after the table.


### size and shape

    len(pydf)
        Provide the number of rows currently used by the data array.

    bool(my_pydf)   # True only if my_pydf exists and is not empty.
    
    (rows, cols) = pydf.shape()   # Provide the current size of the data array.
    
### data typing and conversion
        
    dtype is a dict that specifies the datatype for each column.
    if 'unflatten' is specified and dtype specifies json, then column will be unflattened when the data is read.
    Unflattening will convert JSON in the csv file to produce any arbitrary data item that can be JSON'd.

### creation and conversion

    Pydf() -- Create a new pydf instance.
        parameters:
            lol:        Optional[T_lola]        = None,     # Optional List[List[Any]] to initialize the data array. 
            cols:       Optional[T_ls]          = None,     # Optional column names to use.
            dtype:      Optional[T_dtype_dict]  = None,     # Optional dtype_dict describing the desired type of each column.
            keyfield:   str                     = '',       # A field of the columns to be used as a key.
            name:       str                     = '',       # An optional name of the Pydf array.
            use_copy:   bool                    = False,    # If True, make a deep copy of the lol data.
            disp_cols:  Optional[T_ls]          = None,     # Optional list of strings to use for display, if initialized.

    # create empty pydf with nothing specified.
    my_pydf = Pydf()

    # create empty pydf with specified cols and keyfield, and with dtypes defined.
    my_pydf = Pydf(cols=list_of_colnames, keyfield=fieldname, detype=dtype_dict) 
    
    # create empty pydf with only keyfield specified.
    my_pydf = Pydf(keyfield=fieldname)

    # create an empty pydf object with same cols and keyfield.
    new_pydf = self.clone_empty()

    # Set data table with new_lol (list of list) data item
    my_pydf.set_lol(self, new_lol)

    # create new pydf and fill with data from lod (list of dict) also optionally set keyfield and dtype
    # the cols will be defined from the keys used in the lod. The lod should have the same keys in every record.
    my_pydf = Pydf.from_lod(lod, keyfield=fieldname, dtype=dtype_dict)
    
    # convert Pandas df to Pydf
    my_pydf = Pydf.from_pandas_df(df)

    # convert to Pandas df
    my_pandas_df = my_pydf.to_pandas_df()
    
    # # produce lod (list of dictionaries) type. Generally not needed.
    lod = my_pydf.to_lod()

    
### columns and row indexing

    # return column names defined
    my_pydf.columns()
    
    # return list of row keyfield values, if keyfield is defined.
    my_pydf.keys()    
    
### appending and row/column manipulation    
    
    # append a single row provided as a dictionary.
    my_pydf.append(row)
    
    # append multiple rows as a list of dictionaries.
    my_pydf.append(lod)
    
    # concatenate another pydf as additional rows.
    my_pydf.append(pydf)    


### selecting and removing records by keys

    # select one record using keyfield.
    record_da = my_pydf.select_record_da(keyvalue)                              

    # select multiple records using the keyfield and return pydf.
    new_pydf = my_pydf.select_records_pydf(keyvalue_list)

    # remove a record using keyfield
    my_pydf.remove_key(keyval)
    
    # remove multiple records using multiple keys in a list.
    my_pydf.remove_keylist(keylist)


### selecting records without using keyfield

    # select records based on a conditional expression.
    pydf = pydf.select_where("conditional expression with 'row' like row['fieldname'] > 5")
        or 
    pydf = pydf.select_where(f"row['fieldname'] > {row_limit}")     # use f-strings to import local values.
    
    # Select one record from pydf using the idx and return as a dict.
    record_da = my_pydf.iloc(row_idx)
    
    # select records by matching fields in a dict, and return a lod. inverse means return records that do not match.
    my_lod = my_pydf.select_by_dict_to_lod(selector_da={field:fieldval,...}, expectmax: int=-1, inverse: bool=False)
    
    # select records by matching fields in a dict, and return a new_pydf. inverse means return records that do not match.
    new_pydf = my_pydf.select_by_dict(selector_da, expectmax: int=-1, inverse=False)
    
### column operations    
    
#### return a column by name as a list.
    col_list = my_pydf.col_to_la(colname, unique=False)
    
#### return a column from pydf by col idx as a list 
    col_list = my_pydf.icol_to_la(icol)

#### modify icol by index using list la, overwriting the contents. Append if icol > num cols.
    my_pydf.assign_icol_from_la(icol: int, la: list)
    
#### modify named column using list la, overwriting the contents
    my_pydf.assign_col(colname: str, la: list)
    
#### drop columns by list of colnames
    my_pydf.drop_cols(cols)
    
#### return dict of sums of columns specified or those specified in dtype as int or float if numeric_only is True.
    da = my_pydf.sum(colnames_ls, numeric_only=False)                       

#### create cross-column lookup
Given a pydf with at least two columns, create a dict of list lookup where the key are values in col1 and list of values are 
unique values in col2. Values in cols must be hashable.

    lookup_dol = my_pydf.cols_to_dol(colname1, colname2)                        

#### count discrete values in a column or columns
create dictionary of values and counts for values in colname.
sort by count if sort, from highest to lowest unless 'reverse'

    valuecounts_di = my_pydf.valuecounts_for_colname(colname, sort=False, reverse=True)
    valuecounts_dodi = my_pydf.valuecounts_for_colnames_ls(colnames_ls, sort, reverse)

#### same as above but also selected by a second column
    valuecounts_di = my_pydf.valuecounts_for_colname_selectedby_colname
    valuecounts_dodi = my_pydf.valuecounts_for_colname_groupedby_colname
    
#### group to dict of pydf based on values in colname

    my_dopydf = my_pydf.group_to_dopydf(colname)
    
    
### Indexing: reading values from pydf

Pydf offers easy-to-used indexing of rows, columns, individual cells or any ranges.
Will generally return the simplest type possible, such as cell contents, a list or pydf
- if only one cell is selected, return a single value.
- If only one row is selected, return a list.   (To return a dict, use pydf.
- if only one col is selected, return a list.
- if multiple columns are specified, they will be returned in the original orientation in a consistent pydf instance copied from the original, and with the data specified.

      my_pydf[2, 3]     -- select cell at row 2, col 3 and return value.
      my_pydf[2]        -- select row 2, including all columns, return a list.
      my_pydf[2, :]     -- same as above
      my_pydf[:, 3]     -- select only column 3, including all rows. Return a list.
      my_pydf[:, 'C']   -- select only column named 'C', including all rows, return a list.
      my_pydf[2:4]      -- select rows 2 and 3, including all columns, return as pydf.
      my_pydf[2:4, :]   -- same as above
      my_pydf[:, 3:5]   -- select columns 3 and 4, including all rows, return as pydf.
    
### Indexing: setting values in a pydf:
    my_pydf[irow] = list              -- assign the entire row at index irow to the list provided
    my_pydf[irow] = value             -- assign the entire row at index row to the value provided.
    my_pydf[irow, icol] = value       -- set cell irow, icol to value, where irow, icol are integers.
    my_pydf[irow, start:end] = value  -- set a value in cells in row irow, from columns start to end.
    my_pydf[irow, start:end] = list   -- set values from a list in cells in row irow, from columns start to end.
    my_pydf[:, icol] = list           -- assign the entire column at index icol to the list provided.
    my_pydf[start:end, icol] = list   -- assign a partial column at index icol to list provided.
    my_pydf[irow, colname] = value    -- set a value in cell irow, col, where colname is a string.
    my_pydf[:, colname] = list        -- assign the entire column colname to the list provided.
    my_pydf[start:end, colname] = list    -- assign a partial column colname to list provided from rows start to end.


### Spreadsheet-like formulas
spreadsheet-like formulas can be used to calculate values in the array.
formulas are evaluated using eval() and the data array can be modified using indexed getting and setting of values.
formula cells which are empty '' do not function.

formulas are re-evaluated until there are no further changes. A RuntimeError will result if expressions are circular.

    pydf.apply_formulas(formulas_pydf)

#### Special Notation
Use within formulas as a convenient shorthand:

- $d -- references the current pydf instance
- $c -- the current cell column index
- $r -- the current cell row index

By using the current cell references $r and $c, formulas can be "relative", similar to spreadsheet formulas.
Typical spreadsheet formulas are treated as relative, unless they are made absolute by using $.
Here, the references are absolute unless you create a relative reference by relating to the current cell row $r and/or column $c.
        
#### Example usage:

            example_pydf = Pydf(cols=['A', 'B', 'C'], 
                                lol=[ [1,  2,   0],
                                      [4,  5,   0],
                                      [7,  8,   0],
                                      [0,  0,   0]])
                                      
            formulas_pydf = Pydf(cols=['A', 'B', 'C'], 
                    lol=[['',                    '',                    "sum($d[$r,:$c])"],
                         ['',                    '',                    "sum($d[$r,:$c])"],
                         ['',                    '',                    "sum($d[$r,:$c])"],
                         ["sum($d[:-1,$c])",     "sum($d[:-1,$c])",     "sum($d[:-1, $c])"]]
                         )
                         
            example_pydf.apply_formulas(formulas_pydf)
        
            result       = Pydf(cols=['A', 'B', 'C'], 
                                lol=[ [1,  2,   3],
                                      [4,  5,   9],
                                      [7,  8,   15],
                                      [12, 15,  27]])
    
## Comparison with Pandas, Numpy SQLite

Conversion to pandas directly from arbitrary python with the possibility of mixed types is surprisingly slow.

We timed the various conversions using Pandas 1.5.3 and 2.4.1 and were surprized at the overhead for using Pandas.
(See the table below for the results of our tests)

Version 2.1.4 is only about 6% faster. This analysis was done on an array of a million random integer values.
We are certainly open to being told our manner of using Pandas is incorrect, however, we are not doing anything
fancy. For example, to convert a list of dictionaries (lod) to a Pandas dataframe, we are using `pd.DataFrame(lod)`.

The time to convert the million integer lod to Pandas is about 5.9 seconds.
Meanwhile, you can convert from lod to Numpy in only 0.063 seconds, and then from Numpy directly to Pandas
instantly, because under the hood, Pandas uses numpy arrays. The advantage of using Pydf for arrays with mixed types
gets even more pronounced when strings are mixed with numerical data. Notice that when we add a string column for
indexing the rows in Pandas, the size explodes by more than 10x, making it less efficient than Pydf.

These comparisons were run using 1000 x 1000 array with integers between 0 and 100. Note that by using Pydf, this
decreases the memory space used by 1/3 over simple lod (list of dict) data structure. Pandas can be far more
economical (25% the size of Pydf) in terms of space if the data is not mixed, otherwise, can be much larger. 
Numpy is the most efficient on space because it does not allow mixed data types. 

Manipulating data in the array, such as incrementing values, inserting cols and rows, 
generally is faster for Pydf over Pandas, while column-based operations such as summing all columns is far
faster in Pandas than in Pydf. 

For example, when summing columns, Pandas can save 0.175 seconds per operation, but 6/0.175 = 34. Thus, the 
conversion to dataframe is not worth it for that type of operation unless at least 34 similar operations
are performed. It may be feasible to leverage the low coversion time from Pydf to Numpy, and then perform
the sums in Numpy.


|             Attribute              |  Pydf  |  Pandas  |  Numpy   | Sqlite |  lod   | (loops) |
| ---------------------------------: | :----: | :------: | :------: | :----: | :----: | :------ |
|                           from_lod |  1.3   |   59.3   |   0.63   |  6.8   |        | 10      |
|                       to_pandas_df |  62.5  |          | 0.00028  |        |  59.3  | 10      |
|                     from_pandas_df |  4.1   |          | 0.000045 |        |        |         |
|                           to_numpy |  0.65  | 0.000045 |          |        |  0.63  | 10      |
|                         from_numpy | 0.078  | 0.00028  |          |        |        | 10      |
|                     increment cell | 0.012  |  0.047   |          |        |        | 1,000   |
|                        insert_irow | 0.055  |   0.84   |          |        |        | 100     |
|                        insert_icol |  0.14  |   0.18   |          |        |        | 100     |
|                           sum cols |  1.8   |  0.048   |  0.032   |  3.1   |  1.4   | 10      |
|                          transpose |  20.4  |  0.0029  |          |        |        | 10      |
|                       keyed lookup | 0.0061 |  0.061   |          |  0.36  | 0.0078 | 100     |
|       Size of 1000x1000 array (MB) |  38.3  |   9.5    |   3.9    |  4.9   |  119   |         |
| Size of keyed 1000x1000 array (MB) |  39.4  |   98.1   |          |        |  119   |         |

The numbers above are using Pandas 1.5.3. There was approximately 6% improvement when run with
Pandas 2.1.4. 

This analysis is based on using asizeof to calculate the true memory footprint, and to be fair, 
operations returning rows must return a python dict rather than a native data object, such as a
pandas series. The conversion to dict can include some significant delay.

Note: The above table was generated using Pydf.from_lod_to_cols() and my_pydf.md_pydf_table()


## Conclusion and Summary

The general conclusion is that it generally beneficial to use Pydf instead of lod or pandas df for
general-purpose table manipulation, but when number crunching is needed, prepare the data with Pydf 
and convert directly to Numpy. The fact that we can convert quickly to Numpy and from Numpy to Pandas
is instant makes the delay to load directly to Pandas dataframe seem nonsensical. 

We find that in practice, we had many cases where Pandas data frames were being used only for 
reading data to the array, and then converting to lod data type and processing by rows to
build a lod structure, only to convert to df at the end, with no data crunching where
Pandas excels. We found also that when we performed column-based calculations, that they were 
sometimes wasteful because they would routinely process all data in the column rather than terminating 
early. Complex conditionals and using '.apply' is not recommended for Pandas but work well with Pydf.

Pandas is a very large package and it takes time to load, and may be too big for embedded or 
other memory-constrained applications, whereas Pydf is a very lightweight Package.

Pandas is great for interactive operation where time constraints are of no consequence.

At this time, Pydf is new, and we are still working on the set of methods needed most often 
while recognizing that Numpy will be the best option for true number crunching. Please let
us know of your ideas and improvements.



