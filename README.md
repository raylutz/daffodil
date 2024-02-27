![daffodil_logo](https://github.com/raylutz/Pydf/assets/14955977/5e141583-0216-429d-9ba8-be938aa13017)

# Python Daffodil

(STATUS -- Note: This is an alpha release. The API is in flux. The name of the packege will be changing from Pydf to 'Daffodil' and the class will be Daf, while instance will end with the type _daf. This change is largely reflected in the text below.)

The Python Daffodil (DAtaFrames For Optimized Data Inspection and Logical processing) package provides
lightweight, simple and flexible 2-d dataframes built on 
python data types, including a list-of-list array as the core datatype. Daffodil is similar to other data frame
packages, such as Pandas, Numpy, Polars, Swift, Vaex, Dask, PyArrow, SQLite, PySpark, etc. but is simpler and may be faster 
because it does not have conversion overhead.
Daffodil provides basic dataframe functionality which is not available in core python, but should be. 
Daffodil uses standard python data types, and can mix data types in rows and columns, and can store any type 
within a cell, even another Daffodil instance. 

It works well in traditional Pythonic processing paradigms, such as in loops, allowing fast row appends, 
insertions and other operations that column-oriented packages like Pandas handle poorly or don't offer at all.
Daffodil offers zero-copy manipulations -- selecting, inserting, appending rows does not make a copy of the data but uses references to the original data and works the way Python normally does, leveraging the inherent power of Python without replacing it.

Daffodil is particularly well suited to applications for data munging, incremental appending, data pipelines, 
row-based apply and reduce functions, including support for chunked large data sets that can be described 
by a Daffodil table which operates as a manifest to chunks, and useful for delegations for parallel processing, where each delegation can handle a number of chunks.

Daffodil is not necessarily slower than trying to use Pandas (or other packages) within your python program, 
because moving data in and out of Pandas and those other packages can have very high overhead. Daffodil is a 
very simple 'bare metal' class that is well suited for those situations where pure number crunching is not 
the main objective. But it is also very compatible with other dataframe packages and can provide great way 
to build and clean the data before providing the data to other packages for number crunching.

It excels when the data array needs to be heavily manipulated, particularly by rows or individual data items, 
particularly when rows are to be inserted, removed, or appended. The fact is that data is commonly built
record-by-record, while popular analysis and manipulation tools are oriented to work on data columns once
it is fully assembled. If only a very few data operations are performed on columns (such as a sums, stdev, etc.)
then it is frequently more performant to leave it in row format rather than reforming it into columns and using
those other packages.

Spreadsheet-like operations are also provided, which are useful for processing the entire array with the same formula template,
and can avoid glue code for many transformations. Python equations in the formula pane operate on the data
pane and calculations from spreadsheet programs can be easily ported in, to avoid random glue code.

## Visualization of the data model

![pydf_table](https://github.com/raylutz/Pydf/assets/14955977/011b0bf9-5461-4b0a-af45-5f2bf523417c)

    
## Good for general data operations

We were surprised to find that Pandas is very slow in importing python data to a Pandas DataFrame.
Pandas uses a numpy array for each column which must be allocated in memory as one contiguous block,
resulting in substantial overhead. Converting a list-of_dict (lod) array to Pandas DataFrame using the 
simple `pd.DataFrame(lod)` method takes about 350x longer than converting the same data to a Daffodil instance.

The Daffodil class is based on list-of-list array (lol), and uses a dictionary for column names (hd -- header dict) and for 
row keys (kd -- key dict), making it extremely fast for column and row indexing, while avoiding the requirement for 
contiguous data allocation. Python uses dynamic arrays to store references to each data item in the lol
structure. 

Daffodil is not as performant as Pandas or NumPy for numerical operations such as sums, max, min, stdev, etc. 
when the data is uniform within columns or the entire array. Daffodil does not offer array operations like
C = A + B, where A and B are both large arrays with the same shape producing array C, which is the sum 
of each cell in the same grid location. This type of functionality is already available in NumPy, and 
NumPy can fill that role. Also, it is not a replacement for 
performing matrix operations which is already available in NumPy.

Appending rows in Pandas is slow because each column is stored as a 
separate NumPy array, and appending a row involves creating a new array for each column with the added row. 
This process can lead to significant overhead, especially when dealing with large DataFrames. In fact, Pandas
is so bad that the append operation is now deprecated.

Pandas can be more performant than Daffodil if column-oriented manipulations are repeated on the same data at least ~30x, 
otherwise Daffodil will probably be faster due to the overhead of reading a table from a data array into Pandas. In other words,
if you have an array and you need to do just a few column-based operations (fewer than 30) 
then it will be faster to just do them in Daffodil using a row-oriented apply or reduce operation, rather than
exporting the array to Pandas, performing the calcs and the transferring it back in. (You can see our benchmarks
and other tests linked below.)

Daffodil can work well with Pandas and NumPy when number crunching and array-based operations are needed.
Use Daffodil to build the array incrementally using row-based operations, then export the data to NumPy or
Pandas. Using NumPy is recommended if the data is uniform enough because it is faster and has a smaller
memory footprint than Pandas.

Daffodil is pure python and can be run with no (or few) other packages installed. If Pandas is not used, start up time is 
improved dramatically. This can be very important in cloud-based parallel processing where every millsecond
is billed or in embedded systems that want to utilize tables but can't suffer the overhead.
If conversions to or from Pandas is not required, then that package is not needed.

## Memory Footprint

A Daffodil object (usually `daf`) is about 1/3 the size of a Python lod (list-of-dict) structure because the column names are not repeated,
and dicts are allocated sparsely and consume more space than list-of-list (lol).
However it takes about 4x more memory than a minimal Pandas dataframe and 10x more memory than single NumPy array.
Yet, sometimes Pandas will be much larger when strings are included in the data. The inclusion of one string column
to be used for indexed selections in Pandas consumes 10x more memory than the same data without that column. 
Daffodil does not expand appreciably and will be 1/3 the size of Pandas in that case, and offers searches that are
10x faster.

Thus, Daffodil is a compromise. It is not as wasteful as commonly used lod for such tables, and 
is a good choice when rapid appends, inserts, row-based operations, and other mutation is required. It also
provides row and column operations using \[row, col] indexes, where each can be slices or column names.
This type of indexing is syntactically similar to what is offered by Pandas and Polars, but Daffodil has almost
no constraints on the data in the array, including mixed types in columns, other objects, and even entire
Daffodil arrays in one cell.
        
## Supports incremental appends

Daffodil can append one or more rows or concatenate with another Daffodil object extremely quickly, because it leverages Python's
data referencing approach. Although the 'use_copy' option is provided where a new deep copy is made, this can be
largely avoided in most data manipulation pipelines. When the data can be used without copying, then this will 
minimize overhead. Concatenating, dropping and inserting columns is functionality that is provided, but is not
recommennded. Normally, columns need not be manipulated. Just leave any columns that are not needed out of any calculations or other uses.

A list-of-dict (lod) is an easy to use data structure that mimics a 2-D array and is commonly found in Python code, but it is
expensive in space (3x larger than Daffodil object) because the field names are repeated for each row. Also, it does not offer array manipulation primitives, convenient markdown reporting tools, and other features offered by Daffodil.

If a keyfield is specified, a Daffodil array is quite similar in row-col selection functionality to a dict-of-dict Python structure, but is 1/3 the size, and Daffodil offers many other conveniences.

## Column names

Similar to Pandas and other dataframe concepts, Daffodil has a separate set of column names that can be optionally
used to name the columns. This is organized internally as a Python dictionary (hd -- header dict) for fast column lookups by name.
Column names must be hashable, and other than that, there are no firm restrictions.  
(However, to use the interface with SQLite, avoid using the double underscore "__" in the names, which is used to 
allow arbitrary names in SQLite.)
    
When reading CSV files, the header is normally taken from the first (non-comment) line. If "user_format" is 
specified on reading csv files, the csv data will be pre-processed and "comment" lines starting with # are removed.

Daffodil supports CSVJ, which is a mix of CSV with JSON metadata in comment fields in the first few lines of the file, to provide data type, formatting, and other information. Using CSVJ speeds importing CSV data into a Daffodil instance because the data can be converted to the appropriate time as it is read, and therefore avoids a second pass to convert data from str type, which is the default. This also may unflatten objects. (This feature not supported yet).

In some cases, you may be working with CSV files without a header of column names. Setting noheader=True avoids 
capturing the column names from the header line from csv input, and then column names will not be defined.

If columns are not appropriate for immediate use, such as if they are sometimes repeated or are missing, then
the array can be read with noheader=True, and then poping the first row and applying it using set_cols()
If any column name is blank, then these are named "colN" (or any other prefix you may prefer) 
where N is the column number staring at 0. If there
are duplicates, then the duplicate name is named "duplicatename_N", where 'duplicatename' is the original name
and N is the column number. If you don't want this behavior, then use noheader=True and handle definition of the 
'cols' parameter yourself.

Even if column names are established, Daffodil still allows that columns (and rows) can be indexed by number. 
Therefore, it is best if the column
names used are not pure numbers to avoid confusion. Traditional column names can be added with .set_cols() method, with
the cols parameter set to None. This results in column names similar to spreadsheet programs, 
`'A', 'B', ... 'Z', 'AA', 'AB'...`. 

The column names can be passed in the 'cols' parameter as a list, or if the dtypes dict is provided and cols are not,
then the column names are defined from dtypes dict, and the datatypes are simultaneously defined. The dtypes_dict 
can be used to define datatypes for each column, which is similar behavior to other dataframe packages, but this
is optional. Any cell can be any data type.
    
## Row keyfield   
    
In many cases, one of the columns can be used as a unique key for locating records. If such a column exists, it 
can be adopted as the primary index of the table by specifying that column name as the `keyfield`. When this is done,
then a kd (key dictionary) is built and maintained from that column. This is similar behavior to the Polars
package and differs from Pandas, which has an index that sticks with each row. The row indexes do not stick to the rows in Daffodil, and are always with respect to the frame. This is more like how Polars works.
    
If keyfield is set, then that column must be a hashable type and must have unique values. Searches of row entries based on the keyfield use dictionary lookups, which are highly optimized for speed by Python.

Creating a key index does not remove that field from the data array, but creates an additional key dictionary, kd.

When adopting a file that may have a column that is tainted, it will be best to follow the following steps:
1. Set keyfield='' to turn off the key indexing functionality.
2. Read in the data, and it will not be limited by the keyfield definition.
3. Use method .set_keyfield(keyfield) to set the keyfield and build the lookup dictionary.
4. Check that they are all unique by comparing the number of keys vs. the number of records.
5. if the lengths are different, remove, delete, or otherwise deal with records with duplicate keys so the keys are unique.
6. And then use .set_keyfield(keyfield) again.

Only one keyfield is supported.
    
## Column vs. Row Operations
Daffodil is a row-oriented package, rather than being column oriented, as are other popular packages, like Pandas, Polars, etc, 
Thus, it is easy to manipulate rows (appending, inserting, deleting, selecting, etc) while it is relatively much more difficult to manipulate
columns (appending, inserting, deleting, etc.)  Rows are very easy to handle because the list-of-list underlying structure
re-uses any lists selected in any selection operation. A new Daffodil instance which might be a subset of the original 
does not consume much additional space because the contents of those rows is not copied. Instead, Python
copies only the references to the rows. If only a few rows are used from the original, the the remaining rows will 
be garbage collected by the normal Python mechanisms and the rows that are still active are the same rows that existed in the original array without copying.

This use-without-copying pattern means that Daffodil can perform quite well when compared with other packages when doing this type of manipulation, both in terms of space and also time.

In contrast, operations that add, drop, or insert columns are relatively slow, 
but it turns out that actually these operations are not normally that 
necessary. Reducing the number of columns only is important in a few cases:

1. When converting from/to other forms. Extraneous columns may exist or may be of the wrong type.
2. When performing .apply() or .reduce() operations to avoid processing extraneous columns.
4. When creating a report and only including some columns in the report

In these cases, the columns to be included can be expressed explicitly. When selecting columns, a transposition can be performed for free, if `flip=True` is indicated.

Other column operations such as statistics are not as performant but in those cases when
many operations are required, the appropriate portion of the array can be ported to NumPy, Pandas, or any other dataframe package.

## Common Usage Pattern
       
One common usage pattern allows iteration over the rows and appending to another instance. For example:
    
        # read csv file into 2-d array, handling column headers and unflattening
        my_daf = Pydf.from_csv(file_path, unflatten=True)  
    
        # create a new table to be built as we scan the input.
        new_daf = Pydf()
        
        # scan the input my_daf row by row and construct the output. Pandas can't do this efficiently.
        for original_row in my_daf:  
            new_row = transform_row(original_row)
            new_daf.append(new_row)                
            # appending is no problem in Daffodil. Pandas will emit a future warning that appending is deprecated.
            
        # create a flat csv file with any python objects flattened using JSON.
        new_daf.to_csv(file_path, flatten=True)        

This common pattern can be abbreviated using the apply() method:

        my_daf = Pydf.from_csv(file_path, unflatten=True)
        
        new_daf = my_daf.apply(transform_row)
        
        new_daf.to_csv(file_path, flatten=True)

Or

        Pydf.from_csv(file_path).apply(transform_row).to_csv(file_path)

And further extension of this pattern can apply the transformation to a set of csv files described by a chunk_manifest.
The chunk manifest essentially provides metadata and instructions for accessing the source data.

        chunk_manifest_daf = Pydf.from_csv(file_path)  
        result_manifest_daf = chunk_manifest_daf.manifest_apply(transform_row)

Similarly, a set of csv_files can be reduced to a single record using a reduction method. For example, 
for determining valuecounts of columns in a set of files:

        chunk_manifest_daf = Pydf.from_csv(file_path, unflatten=True)
        result_record = chunk_manifest_daf.manifest_reduce(count_values)
    
## Methods and functionality
       
### print and markdown reports

Daffodil can produce convenient form for interactive inspection similar to the abbreviated form similar to Pandas,
but unlike Pandas, Markdown is the primary format for all reports.

    print(instance_of_daf)

For example, the print statement above will produce markdown text that can be directly viewed interactively and
can also be included in markdown reports. The following is random data in a 1000 x 1000 array.

| Col0 | Col1 | Col2 | Col3 | Col4 | ... | Col995 | Col996 | Col997 | Col998 | Col999 |
| ---: | ---: | ---: | ---: | ---: | --: | -----: | -----: | -----: | -----: | -----: |
|   51 |   92 |   14 |   71 |   60 | ... |      9 |     66 |     17 |     99 |     85 |
|   33 |    7 |   39 |   82 |   41 | ... |     85 |     50 |     87 |     40 |     16 |
|   75 |   45 |   31 |   78 |   79 | ... |     23 |     98 |     25 |     36 |     84 |
|   53 |   20 |   73 |   37 |   45 | ... |     16 |     33 |     15 |     59 |     65 |
|   65 |   89 |   12 |   55 |   30 | ... |     48 |     57 |     38 |     79 |     96 |
|  ... |  ... |  ... |  ... |  ... | ... |    ... |    ... |    ... |    ... |    ... |
|   47 |   57 |   85 |   63 |   23 | ... |     27 |     71 |     55 |     97 |     56 |
|   71 |   48 |   29 |   19 |   43 | ... |     70 |     76 |     80 |     64 |      8 |
|   37 |    4 |   96 |   39 |   82 | ... |     21 |     17 |     31 |     32 |     20 |
|   23 |   39 |   77 |    9 |   21 | ... |      0 |     63 |     22 |     81 |     97 |
|   86 |    9 |   27 |    2 |   40 | ... |     86 |     34 |     61 |     77 |     52 |

\[1000 rows x 1000 cols; keyfield=; 0 keys ] (Pydf)

#### create a Markdown table from a Daffodil instance that can be incorporated in reports.

The method 'to_md()' can be used for more flexible reporting.

    my_daf.to_md() 

        parameters:
            max_rows:       int     = 0,         # limit the maximum number of row by keeping leading and trailing rows.
            max_cols:       int     = 0,         # limit the maximum number of cols by keeping leading and trailing cols.
            just:           str     = '',        # provide the justification for each column, using <, ^, > meaning left, center, right justified.
            shorten_text:   bool    = True,      # if the text in any field is more than the max_text_len, then shorten by keeping the ends and redacting the center text.
            max_text_len:   int     = 80,        # see above.
            smart_fmt:      bool    = False,     # if columns are numeric, then limit the number of figures right of the decimal to "smart" numbers.
            include_summary: bool   = True,      # include a pandas-like summary after the table.


### size and shape

    len(daf)
        Provide the number of rows currently used by the data array.

    bool(my_daf)   # True only if my_daf exists and is not empty.
    
    (rows, cols) = daf.shape()   # Provide the current size of the data array.
    
### data typing and conversion
        
Since we are using Python data objects, each cell in the array can have a different data type. However, it is useful to 
convert data when it is first read from a csv file to make sure it is handled correctly. CSV files, by default, provide
character data and therefore, without conversion, will provide str data. 
        
dtypes is a dict that specifies the datatype for each column, and if provided, will convert the data as it is initially read
from the source. Other sources of data will normally provide the data type when it is imported.
    
When reading flat csv files, if 'unflatten' is specified and dtypes specifies a list or a dict, then the data in those columns 
will be converted from JSON to produce the list or dict object.

Data which is missing is provided as null strings, which will be ignored in apply or reduce operations, and when converted to 
other forms, like NumPy, will be expressed as missing data using NAN or other indicators.

Daffodil supports the CSVJ file format, which includes a set of initial comments that are valid JSON to describe metdata and 
data types. A CSVJ file is generally also a valid CSV file with # comment lines.


### creation and conversion

    Pydf() -- Create a new daffodil instance.
        parameters:
            lol:        Optional[T_lola]        = None,     # Optional List[List[Any]] to initialize the data array. 
            cols:       Optional[T_ls]          = None,     # Optional column names to use.
            dtypes:     Optional[T_dtype_dict]  = None,     # Optional dtype_dict describing the desired type of each column.
            keyfield:   str                     = '',       # A field of the columns to be used as a key.
            name:       str                     = '',       # An optional name of the Daffodil array.
            use_copy:   bool                    = False,    # If True, make a deep copy of the lol data.
            disp_cols:  Optional[T_ls]          = None,     # Optional list of strings to use for display, if initialized.

#### create empty daf with nothing specified.
    
    my_daf = Pydf()

#### create empty daf with specified cols and keyfield, and with dtypes defined.
    
    my_daf = Pydf(cols=list_of_colnames, keyfield=fieldname, dtypes=dtype_dict) 
    
#### create empty daf with only keyfield specified.
    
    my_daf = Pydf(keyfield=fieldname)

#### create an empty daf object with same cols and keyfield.
    
    new_daf = old_daf.clone_empty()

#### Set data table with new_lol (list of list) data item
    
    my_daf.set_lol(new_lol)

#### create new daf with additional parameters
Fill with data from lod (list of dict) also optionally set keyfield and dtypes.
The cols will be defined from the keys used in the lod. The lod should have the same keys in every record.

    my_daf = Pydf.from_lod(lod, keyfield=fieldname, dtypes=dtype_dict)
    
#### convert Pandas df to Daffodil daf
    
    my_daf = Pydf.from_pandas_df(df)

#### convert to Pandas df
    
    my_pandas_df = my_daf.to_pandas_df()
    
#### produce lod (list of dictionaries) type.

Generally not needed as any actions that can be performed on lod can be done with Daffodil.

    lod = my_daf.to_lod()
    
### columns and row indexing
The base array in a Daffodil instance can be indexed with row and column index numbers, starting at 0.
In addition, columns and rows can be named. Column names are generally established from the header
line in the source csv file. Column and row names are optional. If column names exist, they must
be unique, exist, and be hashable. For ease of use, and to separate them from numerical indices,
they should be strings.

Rows are also indexed numerically or optionally with a indexing column. This indexing column, 
unlike the column names, is included in the array. Again, it is most convenient for it to be str type,
and must be hashable. To estabish a column for indexing, the 'keyfield' is set to the column name.

For fast lookups of rows and columns, dictionaries are used to look up the row and column indices.

#### return column names defined

    my_daf.columns()
    
#### return list of row keyfield values, if keyfield is defined.

    my_daf.keys()    
    
### appending and row/column manipulation    
    
#### append a single row provided as a dictionary.
Please note that if a keyfield is set, and if the key already exists in the array, then
it will be overwritten with new data rather than adding a new row. This behavior is consistent
with all types of appending.

    my_daf.append(row)
    
#### append multiple rows as a list of dictionaries.

    my_daf.append(lod)
    
#### concatenate other_daf as additional rows.

    my_daf.append(other_daf)    


### selecting and removing records by keys

#### select one record using keyfield.

    record_da = my_daf.select_record_da(keyvalue)
    
or

   record_list = my_daf[keyvalue].to_dict()
   
Note that this syntax differs from Pandas, which normally references a column if square brackets are used with no other
syntax.

#### select multiple records using the keyfield and return a daf.

    new_daf = my_daf[keyvalue_list]
    

#### drop a record using keyfield

    new_daf = select_krows(krows=keyval, invert=True)
    
    
#### remove multiple records using multiple keys in a list.
    
    new_daf = select_krows(krows=keylist, invert=True)

### selecting records without using keyfield

#### select records based on a conditional expression.

    new_daf = daf.select_where(lambda row: row['fieldname'] > 5)

or

    new_daf = daf.select_where(lambda row: row['fieldname'] > row_limit")
    
#### Select one record from daf using the idx and return as a dict.
    
    record_da = my_daf.iloc(row_idx)    # deprecate
or
    record_da = my_daf[row_idx].to_dict()
or
    record_da = my_daf.select_irows(irows=[row_idx]).to_dict()
    
    
#### select records by matching fields in a dict, and return a lod. inverse means return records that do not match.
    
    my_lod = my_daf.select_by_dict(selector_da={field:fieldval,...}, expectmax: int=-1, inverse: bool=False).to_lod()
    
#### select records by matching fields in a dict, and return a new_daf. inverse means return records that do not match.

    new_daf = my_daf.select_by_dict(selector_da, expectmax: int=-1, inverse=False)
    
### column operations    
    
#### return a column by name as a list.

    col_la = my_daf.select_kcol(kcol=colname).to_list()       # works even if colname is an integer.
    
or

    col_list = my_daf[:, colname].to_list()                   # if the colname is a string
    
#### return a column from daf by col idx as a list 

    col_list = my_daf[:, icol].to_list()

or

    col_list = my_daf.select_icol(icol).to_list()

#### modify icol by index using list la, overwriting the contents. Append if icol > num cols.

    my_daf[:, icol] = la    

    
#### modify named column using list la, overwriting the contents

    my_daf[:, colname] = la
    
    
#### modify named column using single value, overwriting the contents

    my_daf[:, colname] = 0        # write 0's to all cells in the column
    
    
#### drop columns by list of colnames
This operation is not efficient and should be avoided.

    my_daf.drop_cols(colnames_ls)

    my_daf.select_kcols(colnames_ls, invert=True)
    
#### return dict of sums of columns specified or those specified in dtypes as int or float if numeric_only is True.

    sum_da = my_daf.sum(colnames_ls, numeric_only=False)                       

#### create cross-column lookup

Given a daf with at least two columns, create a dict of list lookup where the key are values in col1 and list of values are 
unique values in col2. Values in cols must be hashable.

    lookup_dol = my_daf.cols_to_dol(colname1, colname2)                        

#### count discrete values in a column or columns

create dictionary of values and counts for values in colname.
sort by count if sort, from highest to lowest unless 'reverse'

    valuecounts_di = my_daf.valuecounts_for_colname(colname, sort=False, reverse=True)
    valuecounts_dodi = my_daf.valuecounts_for_colnames_ls(colnames_ls, sort, reverse)

#### same as above but also selected by a second column

    valuecounts_di = my_daf.valuecounts_for_colname_selectedby_colname
    valuecounts_dodi = my_daf.valuecounts_for_colname_groupedby_colname
    
#### group to dict of daf based on values in colname

    my_dodaf = my_daf.group_to_dodaf(colname)    
    
### Indexing: inspecting values in a daf array

Daffodil offers easy-to-used indexing of rows, columns, individual cells or any ranges.
Will generally return the simplest type possible, such as cell contents, a list or daf if retmode == 'val'
otherwise, if retmode == 'obj', then a full daf is returned.

if retmode is 'val':
- if only one cell is selected, return a single value.
- If only one row is selected, return a list.
- if only one col is selected, return a list.
- if multiple columns are specified, they will be returned in the original orientation in a consistent pydf instance copied from the original, and with the data specified.

Please note: operations on columns is relatively inefficient. Try to avoid working on one column at a time.
Instead, use .apply() or .reduce() and handle any manipulations of all columns at the same time. 

Each instance has the my_daf.retmode attribute, which can be 'obj' or 'val'.
If it is 'obj', then using indexing below will always return a daf object. If .retmode is 'val', then it will return either
a single value, list, or daf object, if it is possible to simplify.

      my_daf[2, 3]     -- select cell at row 2, col 3 and return value.
      my_daf[2]        -- select row 2, including all columns, return a list.
      my_daf[2, :]     -- same as above
      my_daf[:, 3]     -- select only column 3, including all rows. Return a list.
      my_daf[:, 'C']   -- select only column named 'C', including all rows, return a list.
      my_daf[2:4]      -- select rows 2 and 3, including all columns, return as daf.
      my_daf[2:4, :]   -- same as above
      my_daf[:, 3:5]   -- select columns 3 and 4, including all rows, return as daf.
      my_daf[[2,4,6]]  -- return rows with indices 2,4,6 as daf array.
      my_daf[:, [1,3,5]] -- return columns with indices 1,3,5 as daf array.
      my_daf[['row5','row6','row7']] -- return rows with keyfield values 'row5','row6','row7'
      my_daf[:, ['col1', 'col3', 'col5']] -- return columns with column names 'col1', 'col3', 'col5'
      my_daf[('row5','row49'), :]] -- return rows with keyfield values 'row5' through 'row49' inclusive (note need for column idx)
      my_daf[('row5',), :]] -- return rows with keyfield values 'row5' through the end (note need for column idx)
      my_daf[(,'row49'), :]] -- return rows with keyfield values from the first row through 'row49' inclusive (note need for column idx)
      my_daf[:, ('col5', 'col23')]] -- return columns with column names from 'col5', through 'col23' inclusive
      my_daf[:, (, 'col23')]] -- return columns with column names from the first column through 'col23' inclusive
      my_daf[:, ('col23',)]] -- return columns with column names from 'col23', through the end

Please note that if you want to index rows by a keyfield or index columns using column names that are integers, 
then you must use method calls. The square-bracket indexing will assume any integers are indices, not names.
The integer values shown in these examples do not index the array directly, but choose the row or columns by name.
To choose by row keys (krows), then keyfield must be set. To choose by column keys (kcols), cols must be set.

      my_daf.select_krows(krows = 123, inverse=False)  -- return daf with one row with integer 123 in keyfield column.
      my_daf.select_krows(krows = [123, 456], inverse=False)  -- return daf with two rows selected by with integers in the keyfield column.
      my_daf.select_krows(krows = [123, 456], inverse=True)   -- return daf dropping two rows selected by with integers in the keyfield column.
      my_daf.select_krows(krows = (123, ), inverse=True)   -- drop all rows starting with row named 123 to the end.
      my_daf.select_krows(krows = (, 123), inverse=True)   -- drop all rows from the first through row named 123.
     
      my_daf.select_kcols(kcols = 123, inverse=False)  -- return daf with one column with integer 123 colname.
      my_daf.select_kcols(kcols = 123, inverse=True)   -- drop column with name 123
      my_daf.select_kcols(kcols = 123, inverse=True, flip=True)   -- drop column with name 123 and transpose columns to rows.

There are also similar methods for selecting by indexes. If flip=True, a transposition is performed and when selecting columns,
it can be done for free (but realize that selecting columns is relatively slow).

    my_daf.select_irows(irows=[1,2,3,45])    -- select rows 1,2,3, and 45 using indices.

    my_daf.select_icols(icols=slice(4,10))   -- select columns 4 thorugh 9 (inclusive) 
    my_daf.select_icols(icols=slice(4,10), flip=True)   -- select columns 4 thorugh 9 (inclusive) and transpose columns to rows.
    my_daf.select_icols(flip=True)   -- select all columns and transpose columns to rows.
          
### Indexing: setting values in a daf:
     my_daf[irow] = list              -- assign the entire row at index irow to the list provided
     my_daf[irow] = value             -- assign the entire row at index row to the value provided.
     my_daf[irow, icol] = value       -- set cell irow, icol to value, where irow, icol are integers.
     my_daf[irow, start:end] = value  -- set a value in cells in row irow, from columns start to end.
     my_daf[irow, start:end] = list   -- set values from a list in cells in row irow, from columns start to end.
     my_daf[:, icol] = list           -- assign the entire column at index icol to the list provided.
     my_daf[start:end, icol] = list   -- assign a partial column at index icol to list provided.
     my_daf[irow, colname] = value    -- set a value in cell irow, col, where colname is a string.
     my_daf[:, colname] = list        -- assign the entire column colname to the list provided.
     my_daf[start:end, colname] = list    -- assign a partial column colname to list provided from rows start to end.


### Spreadsheet-like formulas

spreadsheet-like formulas can be used to calculate values in the array.
formulas are evaluated using eval() and the data array can be modified using indexed getting and setting of values.
formula cells which are empty '' do not function.

formulas are re-evaluated until there are no further changes. A RuntimeError will result if expressions are circular.

    daf.apply_formulas(formulas_daf)

#### Special Notation
Use within formulas as a convenient shorthand:

- `$d` -- references the current daf instance
- `$c` -- the current cell column index
- `$r` -- the current cell row index

By using the current cell references `$r` and `$c`, formulas can be "relative", similar to spreadsheet formulas.
In contrast with typical spreadsheet formulas which are treated as relative, unless they are made absolute by using `$`,
here, the references are absolute unless you create a relative reference by relating to the current cell row `$r` and/or column `$c`.
        
#### Example usage:
In this example, we have an 4 x 3 array and we will sum the rows and columns to the right and bottom col and row, respectively.

            example_daf = Pydf(cols=['A', 'B', 'C'], 
                                lol=[ [1,  2,   0],
                                      [4,  5,   0],
                                      [7,  8,   0],
                                      [0,  0,   0]])
                                      
            formulas_daf = Pydf(cols=['A', 'B', 'C'], 
                    lol=[['',                    '',                    "sum($d[$r,:$c])"],
                         ['',                    '',                    "sum($d[$r,:$c])"],
                         ['',                    '',                    "sum($d[$r,:$c])"],
                         ["sum($d[:$r,$c])",     "sum($d[:$r,$c])",     "sum($d[:$r, $c])"]]
                         )
                         
            example_daf.apply_formulas(formulas_daf)
        
            result       = Pydf(cols=['A', 'B', 'C'], 
                                lol=[ [1,  2,   3],
                                      [4,  5,   9],
                                      [7,  8,   15],
                                      [12, 15,  27]])
    
## Comparison with Pandas, Numpy SQLite

One of the primary reasons for developing Daffodil is that 
conversion to Pandas directly from arbitrary Python list-of-dicts, for example, is surprisingly slow.

We timed the various conversions using Pandas 1.5.3 and 2.4.1.
(See the table below for the results of our tests). 

Also see: https://github.com/raylutz/Pydf/blob/main/docs/test_df_vs_pydf.md

Version 2.1.4 is only about 6% faster. This analysis was done on an array of a million random integer values.
We are certainly open to being told our manner of using Pandas is incorrect, however, we are not doing anything
fancy. For example, to convert a list of dictionaries (lod) to a Pandas dataframe, we are using `pd.DataFrame(lod)`.

The time to convert the million integer lod to Pandas is about 5.9 seconds.
Meanwhile, you can convert from lod to Numpy in only 0.063 seconds, and then from Numpy directly to Pandas
instantly, because under the hood, Pandas uses numpy arrays. The advantage of using Daffodil for arrays with mixed types
gets even more pronounced when strings are mixed with numerical data. Notice that when we add a string column for
indexing the rows in Pandas, the size explodes by more than 10x, making Pandas less space efficient than Daffodil.

These comparisons were run using 1000 x 1000 array with integers between 0 and 100. Note that by using Daffodil, this
decreases the memory space used by 1/3 over simple lod (list of dict) data structure. Pandas can be far more
economical (25% the size of daf) in terms of space if the data is not mixed, otherwise, can be much larger. 
Numpy is the most efficient on space because it does not allow mixed data types. 

Manipulating data in the array, such as incrementing values, inserting cols and rows, 
generally is faster for Daffodil over Pandas, while column-based operations such as summing all columns is far
faster in Pandas than in Daffodil. Row-based operations in Pandas, like summing values in rows may be best 
implemented by transposing the array and then summing columns, because of the column-based data organization.

For example, when summing columns, Pandas can save 0.175 seconds per operation, but 5.9/0.175 = 34. Thus, the 
conversion to dataframe is not worth it for that type of operation unless at least 34 similar operations
are performed. It may be feasible to leverage the low coversion time from Daffodil to Numpy, and then perform
the sums in Numpy.


|             Attribute              |  Daffodil  |  Pandas  |  Numpy   | Sqlite |  lod   | (loops) |
| ---------------------------------: | :--------: | :------: | :------: | :----: | :----: | :------ |
|                           from_lod |      1.3   |   59.3   |   0.63   |  6.8   |        | 10      |
|                       to_pandas_df |      62.5  |          | 0.00028  |        |  59.3  | 10      |
|                     from_pandas_df |      4.1   |          | 0.000045 |        |        |         |
|                           to_numpy |      0.65  | 0.000045 |          |        |  0.63  | 10      |
|                         from_numpy |     0.078  | 0.00028  |          |        |        | 10      |
|                     increment cell |     0.012  |  0.047   |          |        |        | 1,000   |
|                        insert_irow |     0.055  |   0.84   |          |        |        | 100     |
|                        insert_icol |      0.14  |   0.18   |          |        |        | 100     |
|                           sum cols |      1.8   |  0.048   |  0.032   |  3.1   |  1.4   | 10      |
|                          transpose |      20.4  |  0.0029  |          |        |        | 10      |
|                       keyed lookup |     0.0061 |  0.061   |          |  0.36  | 0.0078 | 100     |
|       Size of 1000x1000 array (MB) |      38.3  |   9.5    |   3.9    |  4.9   |  119   |         |
| Size of keyed 1000x1000 array (MB) |      39.4  |   98.1   |          |        |  119   |         |

The numbers above are using Pandas 1.5.3. There was approximately 6% improvement when run with
Pandas 2.1.4. 

This analysis is based on using asizeof to calculate the true memory footprint, and to be fair, 
operations returning rows must return a python dict rather than a native data object, such as a
pandas series. The conversion to dict can include some significant delay.

Note: The above table was generated using Pydf.from_lod_to_cols() and my_daf.to_md() and
demonstrates how Daffodil can be helpful in creating markdown reports.

## Demo

See this demo of Daffodil functionality.

https://github.com/raylutz/Pydf/blob/main/docs/pydf_demo.md



## Conclusion and Summary

The general conclusion is that it generally beneficial to use Daffodil instead of lod or pandas df for
general-purpose table manipulation, but when number crunching is needed, prepare the data with Daffodil 
and convert directly to Numpy. The fact that we can convert quickly to Numpy and from Numpy to Pandas
is instant makes the delay to load directly to Pandas dataframe seem nonsensical. 

We find that in practice, we had many cases where Pandas data frames were being used only for 
reading data to the array, and then converting to lod data type and processing by rows to
build a lod structure, only to convert to df at the end, with no data crunching where
Pandas excels. We found also that when we performed column-based calculations, that they were 
sometimes wasteful because they would routinely process all data in the column rather than terminating 
early. Complex conditionals and using '.apply' is not recommended for Pandas but work well with Daffodil.

Pandas is a very large package and it takes time to load, and may be too big for embedded or 
other memory-constrained applications, whereas Daffodil is a very lightweight Package. Daffodil
can be very lightweight because you can use regular Python expressions for all apply and reduce
operations. Thus, no need for a slew of new method names, lazy evaluation, and processing concepts.

At this time, Daffodil is new, and we are still working on the set of methods needed most often 
while recognizing that Numpy will be the best option for true number crunching. Please let
us know of your ideas and improvements.



