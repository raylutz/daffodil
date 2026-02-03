![daffodil_logo](https://github.com/raylutz/daffodil/assets/14955977/5e141583-0216-429d-9ba8-be938aa13017)

# Python Daffodil

The Python Daffodil (DAtaFrames For Optimized Data Inspection and Logical processing) package provides
lightweight, simple and flexible 2-d dataframes built on 
python data types, including a list-of-list array as the core datatype. Daffodil is similar to other data frame
packages, such as Pandas, Numpy, Polars, Swift, Vaex, Dask, PyArrow, SQLite, PySpark, etc. but is simpler and may be faster 
because it does not have conversion overhead. Daffodil excels in row-based appends and apply operations, complex embedded types, etc.

STATUS: Daffodil is largely operating quite well, but there are still design tradeoffs that are being investigated.
Some of the methods of the class and assumptions that can be made about the state of the data may change slightly
as these design tradeoffs are being evaluated and the final initial design resolved. Please see GitHub issues to
weigh in on the design.

Also planning extensions to support treating SQL tables as daffodil data tables, improved support for AI and ML
embeddings, rapid conversion of specified cols to dict-of-numpy array .to_dnpa() which is a lightweight pandas-like form
that supports column-oriented array operations.

## Data Model
The Daffodil data model is really very simple. The core array is a list-of-lists (lol), optionally with one or two associated
dictionaries, one for the column names and one for row keys.

![image](https://github.com/raylutz/daffodil/assets/14955977/fa33237c-2075-4bbe-81e1-a6c1e324f46a)


Daffodil uses standard python data types, and can mix data types in rows and columns, and can store any type 
within a cell, even another Daffodil instance. 

It works well in traditional Pythonic processing paradigms, such as in loops, allowing fast row appends, 
insertions and other operations that column-oriented packages like Pandas handle poorly or don't offer at all.
Selecting, inserting, appending rows does not make a copy of the data but uses references the way 
Python normally does, leveraging the inherent power of Python without replacing it.

Daffodil offers row-based apply and reduce functions, including support for chunked large data sets that can be described 
by a Daffodil table which operates as a manifest to chunks, and useful for delegations for parallel processing, 
where each delegation can handle a number of chunks.

Daffodil is a very simple 'bare metal' class that is well suited for those situations where pure number crunching is not 
the main objective. But it is also very compatible with other dataframe packages and can provide great way 
to build and clean the data before providing the data to other packages for number crunching.

Tabular data is commonly built
record-by-record, while popular analysis and manipulation tools are oriented to work on data columns once
it is fully assembled. If only a very few data operations are performed (say < 30) on columns (such as a sums, stdev, etc.)
then it is frequently more performant to leave it in row format rather than reforming it into columns and enduring
the delays of porting and converting the data to those other packages.

Spreadsheet-like operations are also provided, which are useful for processing the entire array with the same formula template,
and can avoid glue code for many transformations. Python equations in the formula pane operate on the data
pane and calculations from spreadsheet programs can be easily ported in, to avoid random glue code.

## Good for general data operations

We were surprised to find that Pandas is **very slow** in importing Python data.
Pandas uses a numpy array for each column which must be allocated in memory as one contiguous block. 
Converting a row-oriented list-of_dict (lod) array to Pandas DataFrame using the 
simple `pd.DataFrame(lod)` method takes about 45x longer than converting the same data to a Daffodil instance.

The Daffodil class is simple, with the data array a list-of-list (lol), and uses a dictionary for column names (hd -- header dict) and for 
row keys (kd -- key dict), making it extremely fast for column and row indexing, while avoiding the requirement for 
contiguous data allocation. Python uses dynamic arrays to store references to each data item in the lol
structure. But it also provides a simple UI to make it easy to select rows, columns, slices by number or names.

For numerical operations such as sums, max, min, stdev, etc., Daffodil is not as performant as Pandas or NumPy 
when the data is uniform within columns or the entire array. Daffodil does not offer additional array operations like
C = A + B, where A and B are both large arrays with the same shape producing array C, which is the sum 
of each cell in the same grid location. This type of functionality, as well as matrix operations is already available in NumPy, and 
NumPy can fill that role.

Appending rows in Pandas is slow because each column is stored as a 
separate NumPy array, and appending a row involves creating a new array for each column with the added row. 
This process can lead to significant overhead, especially when dealing with large DataFrames. In fact, it
is so bad that the append operation is now deprecated in Pandas. That means you have to turn to some other
method of building your data, and then the question comes up: Should I export the data to Pandas or just
work on it in Python. 

When is it worth porting to Pandas? That will depend on the size of the array and what the calculations are.
But a convenient rule of thumb from our testing is that Pandas can be more performant than Daffodil if 
column-oriented manipulations (similar to summing) are repeated on the same data at least ~30 times.

In other words, if you have an array and you need to do just a few column-based operations (fewer than 30), 
then it will be probably be faster to just do them using Daffodil `.apply()` or `.reduce()` operations, rather than
exporting the array to Pandas, performing the calcs and the transferring it back in. (You can see our benchmarks
and other tests linked below.)

In addition, Daffodil is a good partner with Pandas and NumPy when only some number crunching and array-based operations are needed.
Use Daffodil to build the array incrementally using row-based operations, then export the data to NumPy or
Pandas. NumPy is recommended if the data is uniform enough because it is faster and has a smaller
memory footprint than Pandas.

Daffodil is pure python and can be run with no (or few) other packages installed. It is relatively tiny. 
If Pandas is not used, start up time is improved dramatically. This can be very important in cloud-based parallel processing where every millsecond
is billed or in embedded systems that want to utilize tables but can't suffer the overhead.
If conversions to or from Pandas is not required, then that package is not needed.

Daffodil record lookups by key are extremely fast because they use the Python dictionary for looking up rows. It is about 10x faster than
Pandas in this regard. It will be hard to beat this, as long as the data table can fit in memory. And with any string data, Daffodil tables
are smaller than Pandas tables.

## Memory Footprint

A Daffodil object (usually `daf`) is about 1/3 the size of a Python lod (list-of-dict) structure where each record has
repeated column names. The Daffodil array has only one (optional) dictionary for column keys and one (optional) dictionary
for row keys.

With only numeric data, it takes about 4x more memory than a minimal Pandas dataframe and 10x more memory than single NumPy array.
Yet, sometimes Pandas will be much larger when strings are included in the data. The inclusion of one string column
to be used for indexed selections in Pandas consumes 10x more memory than the same data without that column. 
Daffodil does not expand appreciably and will be 1/3 the size of Pandas in that case.

Thus, Daffodil is a compromise. It is not as wasteful as commonly used lod for such tables, and 
is a good choice when rapid appends, inserts, row-based operations, and other mutation is required. It also
provides row and column operations using \[row, col] indexes, where each can be slices, or lists of indices or names.
This type of indexing is syntactically similar to what is offered by Pandas and Polars, but Daffodil has almost
no constraints on the data in the array, including mixed types in columns, other objects, and even entire
Daffodil arrays in one cell. We believe the UI is simpler than what is offered by Pandas.
        
## Supports incremental appends

Daffodil can append or insert one or more rows or concatenate with another Daffodil object extremely quickly, because it leverages Python's
data referencing approach. When the data can be used without copying, then this will 
minimize overhead. Concatenating, dropping and inserting columns is functionality that is provided, but is not
recommennded. Avoid explicitly dropping columns and simply provide a list of columns included in calculations.

## Column names

Similar to Pandas and other dataframe concepts, Daffodil has a separate set of column names that can be optionally
used to name the columns. This is organized internally as a Python dictionary (hd -- header dict) for fast column lookups by name.
Column names must be hashable and unique, and other than that, there are no firm restrictions.  
(However, to use the interface with SQLite, avoid using the double underscore "__" in the names, which is used to 
allow arbitrary names in SQLite.)
    
When reading CSV files, the header is normally taken from the first (non-comment) line. If "user_format" is 
specified on reading csv files, the csv data will be pre-processed and "comment" lines starting with # are removed.

<!--
Daffodil supports CSVJ, which is a mix of CSV with JSON metadata in comment fields in the first few lines of the file, 
to provide data type, formatting, and other information. Using CSVJ speeds importing CSV data into a Daffodil instance 
because the data can be converted to the appropriate type as it is read, and therefore avoids a second pass to convert 
data from str type, which is the default. This also may unflatten objects. (CSVJ not supported yet).
-->

In some cases, you may be working with CSV files without a header line providing of column names. This is common
in xlsx spreadsheets which have column names set by position. Setting noheader=True avoids 
capturing the column names from the header line from csv input, and then column names will not be defined, but can
be defined in code.

If columns repeated or are missing, this will be detected when first read, and the header row will be adjusted
so it can be used. If any column name is blank, then these are named "colN" (or any other prefix you may prefer) 
where N is the column number staring at 0. If there
are duplicates, then the duplicate name is named "duplicatename_N", where 'duplicatename' is the original name
and N is the column number. If you don't want this behavior, then use noheader=True and handle definition of the 
'cols' parameter yourself.

Even if column names are established, Daffodil still allows that columns (and rows) can be indexed by number. 
Daffodil can support numeric row or column names that are different from the inherent lol array indices, but
methods must be used to disambiguate from the numeric indices which will default when using square bracket 
indexing. Column names can be added with .set_cols() method. If the cols parameter is set to None, 'A1' naming is used,
similar to spreadsheet programs, `'A', 'B', ... 'Z', 'AA', 'AB'...`. 

The column names can be passed in the 'cols' parameter as a list, or if the dtypes dict is provided and cols are not,
then the column names are defined from dtypes dict, and the datatypes are simultaneously defined. The dtypes_dict 
can be optionally used to define datatypes for each column, which is similar behavior to other dataframe packages, but this
is optional. Any cell can be any data type. the `dtypes` dict is useful when reading data from csv files because the
default type is `str`.

### Schema-based column definition (optional)

In addition to defining column names via cols or dtypes, a Daffodil instance may optionally be initialized with a schema class. 
A schema class uses standard Python type-annotation syntax to declare column names, their intended types, and default values 
in a single, readable location.

Example:

    class BallotSchema:
        __keyfield__ = "ballot_id"
        ballot_id: str = ""
        contest: str = ""
        page: int = 0


Passing this class to the constructor:

    daf = Daf(schema=BallotSchema)

has the following effects:

- Column names are defined from the annotated attribute names.
- The dtypes dictionary is automatically derived from the annotations.
- Default values are remembered and can be used to create new records.
- if __keyfield__ is specified, then self.keyfield will be set to this value by the constructor, if keyfield is not otherwise specified.
 

The schema is used only as a declarative source of column metadata. It does not impose validation or restrict what values 
may be stored in the table unless additional checking is explicitly enabled by the user.

When a schema is attached, a default record can be created using:

    record = daf.default_record()

which returns a new dictionary populated with the schema’s default values.

Schema usage is entirely optional. Explicit dtypes or cols arguments continue to work as before, and if both a 
schema and dtypes are provided, the explicitly supplied dtypes take precedence.
    
## Row keyfield   
    
In many cases, one of the columns can be used as a unique key for locating records. If such a column exists, it 
can be adopted as the primary index of the table by specifying that column name as the `keyfield`. When this is done,
then the `kd` (key dictionary) is built and maintained from that column. 
Creating a key index does not remove that field from the data array. `kd` is an additional structure created internally.

If keyfield is set, then that column must be a hashable type and must have unique values. Searches of row entries use 
dictionary lookups, which are highly optimized for speed by Python.

The kd can also be set without the existence of a column to adopt from the array. This is useful particularly when 
transposing the dataframe so that the column names can be adopted as the row keys, and vice versa.

Unlike the keyfield oriented lookup functionality, row indices do not stick to the rows and are always with respect to the frame. 
This is similar behavior to the Polars package and differs from Pandas, which has an index that sticks with each row, and is more
like the kd approach used by Daffodil.

When adopting a file that may have a column that is tainted, it will be best to follow the following steps:
1. Set `keyfield=''` to turn off the key indexing functionality.
2. Read in the data, and it will not be limited by the keyfield definition.
3. Use method `my_daf.set_keyfield(keyfield_name)` to set the keyfield to the column `keyfield_name` and build the lookup dictionary.
4. Check that they are all unique by comparing the number of keys vs. the number of records.
5. if the lengths are different, remove, delete, or otherwise deal with records with duplicate keys so the keys are unique.
6. And then use `.set_keyfield(keyfield)` again.

The convenience method `my_daf.add_idx()` can be used to add a column with indices that can be used as a keyfield.

Only one keyfield is supported, but additional keying can be built by the users by creating dicts of any column or set of columns.

Joins require a common keyfield among daf arrays being joined.
    
## Column vs. Row Operations
Daffodil is a row-oriented package. Other popular packages, like Pandas, Polars, etc, are column oriented because it can be very 
beneficial for calculations that can be performed on a column basis.

Thus, in Daffodil, it is easy to manipulate rows (appending, inserting, deleting, etc) while it is relatively much more difficult to manipulate
columns.  Rows are very easy to handle because the list-of-list underlying structure
re-uses any lists selected in any selection operation. A new Daffodil instance which might include be a subset of the rows in the original 
does not consume much additional space because the contents of those rows is not copied. Instead, Python
copies only the _references_ to the rows. If only a few rows are used from the original, the the remaining rows will 
be garbage collected by the normal Python mechanisms.  The rows that are still active are the same rows that existed in the original array without copying.

This use-without-copying pattern means that Daffodil can perform quite well when compared with other packages when doing this type of manipulation, both in terms of space and also time.

In contrast, operations that add, drop, or insert columns are relatively slow, 
but it turns out that actually these operations are not normally that 
necessary. Reducing the number of columns only is important in a few cases:

1. When converting from/to other forms. Extraneous columns may exist or may be of the wrong type.
2. When performing `.apply()` or `.reduce()` operations to avoid processing extraneous columns.
4. When creating a report and only including some columns in the report

In these cases, the columns to be included can be expressed explicitly, rather then modifying the array by dropping them. 

Nevertheless, these operations are provided. When selecting/dropping columns, a transposition can be performed for free, if `flip=True` is indicated.

Other column operations such as statistics are not as performant as column-based packages but in those cases when
many operations are required, the appropriate portion of the array can be ported to NumPy, Pandas, or any other dataframe package.

Also, rows can be conceptually treated as if they are columns, because the structure of a Daffodil array is transposition symmetrical. Simply
place column data in each row and name the row with the column name. To sum values in a column, then the values in each row, which is a 
list, can be summed with the sum() operator.

## Datatypes and conversion

### data typing and conversion
        
Daffodil stores data as native Python objects, and each cell in the array may hold a value of any type. As a result, 
data typing is flexible by design and not enforced unless explicitly requested.

Type conversion is most relevant when reading data from CSV files. CSV input is inherently text-based, so values are 
initially read as str unless converted. In many workflows, converting the entire dataset eagerly is unnecessary or 
undesirable, so Daffodil allows type conversion to be applied explicitly and selectively.

Missing data in CSV input is represented internally as null strings. These values are ignored in apply and reduce 
operations. When data is converted to other representations, such as NumPy arrays, missing values may be expressed 
using NaN or other sentinel values. Null strings are used internally because they display cleanly when printed, 
avoiding the visual clutter associated with NaN.

Column type hints may be provided using the optional dtypes dictionary, which maps column names to Python types (for 
example, str, int, float, dict, or list). These types are advisory and are primarily used to guide conversion from 
string values when reading CSV data. Type conversion is applied only when explicitly requested, preserving lazy evaluation 
and minimizing unnecessary data transformation.

As an alternative to specifying dtypes directly, a schema class may be provided when constructing a Daffodil instance. 
When used, the schema defines column names, intended types, and default values in a single location using standard Python 
type-annotation syntax. The dtypes dictionary is derived automatically from the schema annotations, and conversion behavior 
remains unchanged.


### `.apply_dtypes(dtypes, unflatten, from_str)`

When data is read from a `.csv` file, it is parsed into `str` objects, as this is the fastest possible way to load data from such a file.
The `.apply_dtypes()` method is used to convert all or some of the columns to the appropriate type. This conversion is done "in place"
and a new array is not created. Thus, if only a limited number of columns is converted, it will not disturb the other columns for 
best performance.

If `from_str` is True (the default), only non-`str` columns are converted. Otherwise, columns specified as `str` will 
also be scanned to ensure that they are expressed as `str` types.

If `unflatten` is True (the default), columns with `list` or `dict` types will be unflattened from PYON or JSON to create 
accurate internal object types. Use PYON to be able to create sets, tuples and dicts with non-string keys. These are not
directly supported in JSON.

If the `.apply_dtypes()` method is called with a `dtypes` argument, then if the Daf object does not have any dtypes defined, 
the `dtypes` parameter will be used to initiallize the internal `dtypes` attribute and 
the columns will be defined accordingly. However, if the `dtypes` attribute is already defined as non-empty, 
then the `dtypes` dictionary argument is used to define which columns will be included in the operation.

To apply dtypes to a .csv file, then the following syntax can be used:

    my_daf = Daf.from_csv('filename.csv').apply_dtypes(dtypes=my_daf_dtypes)
    
which will convert all non-`str` types and unflatten JSON encoded `dict` and `list` values. Here, any `str` data is left alone.

The explicit nature of the `.apply_dtypes()` method makes it feasible to avoid any conversion if say only `str` formatted columns
are needed, and improve performance, and the operation of conversion of types is easy to understand.

### `.flatten()`

PYON is Python Object Notation, like JSON, but uses single quotes when writing but can read either single or double
quotes, and can represent any core python object, such as lists, dicts, sets, and tuples. It also can handle non-string keys
(None, integers and tuples, for example) in dicts. All objects found are converted to PYON when written automatially. To save space
in the .csv file, use 0 an 1 to represent booleans.  See https://github.com/raylutz/README.md for more information on PYON.

It is no longer necessary to manually flatten  `list` or `dict` objects using the `.flatten()` method.

    my_daf.flatten().to_csv('filename.csv')
    
or alternatively:

    my_daf.to_csv('filename.csv', flatten=True)
    
### `.to_list()`, `.to_dict()`, `.to_value()`

When selecting a single row or column in a Daf array, it will be returned normally as another Daf object.
However, you can use `.to_list()` to convert it to a single list of values, or `.to_dict()` to get a dictionary,
with keys set as the column names. If a single cell is selected, use .to_value() to obtain that single value.

### `.retmode`

A Daf object also has the attribute `.retmode` which can be either 'obj' (default) or 'val'
If set to 'obj', then Daffodil objects are always produced from a selection operation. If set to 'val',
then it will return a single value if a single cell is selected, or a list if a single row or column
is selected.

### example1

For example, to sum all the values in a specific column, converting to a list will allow the python sum()
operator to correctly sum the values. Caution: if the values must be numeric types.

    total = sum(my_daf[:, 'this_column'].to_list())
    
Note: for performance, use `reduce()` and process all columns at the same time if multiple columns are to be
summed, for example, as this is much more peformant and is scalable to multiple `.csv` files.

### example2

In this example, we set the retmode to 'val' so individual or list values will result

    my_daf.retmode = 'val'
    
    total_one = my_daf[3,4] + my_daf[5,6] * 10

    total_two = sum(my_daf[:, 'this_column'])

       
## Common Usage Pattern
       
One common usage pattern allows iteration over the rows and appending to another Daf instance. For example:
    
        # read csv file into 2-D array, handling column headers, respecting data types and unflattening
        my_daf = Daf.from_csv(file_path).apply_dtypes(dtypes=my_daf_dtypes)
    
        # create a new (empty) table to be built as we scan the input.
        new_daf = Daf()
        
        # scan the input my_daf row by row and construct the output. Pandas can't do this efficiently.
        
        for original_row in my_daf:  
            new_row = transform_row(original_row)
            new_daf.append(new_row)                
            # appending is no problem in Daffodil. Pandas will emit a future warning that appending is deprecated.
            # here the column names are initialized as the first dictionary is appended.
            
        # create a flat csv file with any python objects flattened using JSON.
        new_daf.flatten().to_csv(file_path)
        

This common pattern can be abbreviated using the apply() method:

        my_daf = Daf.from_csv(file_path).apply_dtypes(dtypes=my_daf_dtypes)
        
        new_daf = my_daf.apply(transform_row)
        
        new_daf.flatten().to_csv(file_path)

Or

        Daf.from_csv(file_path).apply_dtypes(dtypes=my_daf_dtypes).apply(transform_row).flatten().to_csv(file_path)

And further extension of this pattern can apply the transformation to a set of csv files described by a chunk_manifest.
The chunk manifest essentially provides metadata and instructions for accessing the source data, which may be many 1000s
of chunks, each of which will fit in memory.

        chunk_manifest_daf = Daf.from_csv(file_path)  
        result_manifest_daf = chunk_manifest_daf.manifest_apply(transform_row)

Similarly, a set of csv_files can be reduced to a single record using a reduction method. For example, 
for determining valuecounts of columns in a set of files:

        chunk_manifest_daf = Daf.from_csv(file_path)
        result_record = chunk_manifest_daf.manifest_reduce(count_values)
        
Daffodil actually extends to chunks very elegantly because the apply() or reduce() operators can be applied to the
rows and to chunks just as well. In contrast, column-based schemes require many passes through the data or delayed
"lazy" operations that are difficult to comprehend.        
        
    
## Methods and functionality
       
### print and markdown reports

Daffodil can produce convenient form for interactive inspection similar to Pandas,
but unlike Pandas, Markdown is the primary format for all reports. Markdown provided
by Daffodil are also convieniently formatted for fixed-font displays, such as a monitor.

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

\[1000 rows x 1000 cols; keyfield=; 0 keys ] (Daf)

#### create a Markdown table from a Daffodil instance that can be incorporated in reports.

The method `to_md()` can be used for more flexible reporting.

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
    


### creation and conversion

    Daf() -- Create a new daffodil instance.
        parameters:
            lol:        Optional[T_lola]        = None,     # Optional List[List[Any]] to initialize the data array. 
            cols:       Optional[T_ls]          = None,     # Optional column names to use.
            dtypes:     Optional[T_dtype_dict]  = None,     # Optional dtype_dict describing the desired type of each column.
            keyfield:   str                     = '',       # A field of the columns to be used as a key.
            name:       str                     = '',       # An optional name of the Daffodil array.
            use_copy:   bool                    = False,    # If True, make a deep copy of the lol data.
            disp_cols:  Optional[T_ls]          = None,     # Optional list of strings to use for display, if initialized.
            retmode:    str                     = 'obj',    # determines how data will be returned, either always as an daf array ('obj')
                                                            # or as single values, list or Daf instance ('val')

#### create empty daf with nothing specified.
    
    my_daf = Daf()

#### create empty daf with specified cols and keyfield, and with dtypes defined.
    
    my_daf = Daf(cols=list_of_colnames, keyfield=fieldname, dtypes=dtype_dict)
    
Note that although dtypes may be defined, conversion of types can be an expensive
operation and so it is done explicitly, using the `apply_dtypes()` method.    
    
#### create empty daf with only keyfield specified.
    
    my_daf = Daf(keyfield=fieldname)

#### create an empty daf object with same cols and keyfield.
    
    new_daf = old_daf.clone_empty()

#### Set data table with new_lol (list of list) data item
    
    my_daf.set_lol(new_lol)

#### create new daf with additional parameters

Fill with data from lod (list of dict) also optionally set keyfield and dtypes.
The cols will be defined from the keys used in the lod. The lod should have the same keys in every record.

    my_daf = Daf.from_lod(lod, keyfield=fieldname, dtypes=dtype_dict)
    
#### convert Pandas df to Daffodil daf type
When converting from or to a Pandas dataframe, datatypes are converted using reasonable conversion lists.
    
    my_daf = Daf.from_pandas_df(df)

#### convert to Pandas df
    
    my_pandas_df = my_daf.to_pandas_df()
    
#### produce alternative Python forms.

##### to_lod()
Converts the Daffodil array to a list of dictionaries. The lod data type is ~3x larger than a Daf array. 
(Generally not needed as any actions that can be performed on lod can be done with a Daffodil instance.)

    lod = my_daf.to_lod()

#### to_dict(irow=0)
Converts a single row to a dictionary, defaulting to row 0. This is convenient if the array has been 
reduced to a single row, and it can be used conveniently as a single dictionary.

#### to_list(irow=None, icol=None, unique=False)
Convert a single row or column to a list. If both irow and icol are None, then return either row 0
or col 0 based on what is found in the array. Otherwise, return a single row or column based on 
the irow or icol parameter.
    
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
    
### Indexing: inspecting values in a daf array

Daffodil offers easy-to-used indexing of rows, columns, individual cells or any ranges.
if `retmode == 'val'`, then it will generally return the simplest type possible, such as cell contents, a list or daf 
otherwise, if `retmode == 'obj'`, then a full daf object is returned. If you desired a list or dict, then it is 
convenient to just use the `.to_list()` or `.to_dict()` methods.

if retmode is 'val':
- if only one cell is selected, return a single value.
- If only one row is selected, return a list.
- if only one col is selected, return a list.
- if multiple columns are specified, they will be returned in the original orientation in a consistent daf instance copied from the original, and with the data specified.

Please note: operations on columns is relatively inefficient. Try to avoid working on one column at a time.
Instead, use .apply() or .reduce() and handle any manipulations without dropping columns, and select them with 
the cols parameter at that time.

|  Expression                               | Operation                                                                 |
|:------------------------------------------|:--------------------------------------------------------------------------|
|`my_daf[2, 3]`                             | select cell at row 2, col 3 and return a daf array with one value         |
|`my_daf[2, 3].to_value()`                  | select cell at row 2, col 3 and return a the value                        |
|`my_daf[2]`                                | select row 2, including all columns, return a daf array with one column   |
|`my_daf[2, :]`                             | same as above                                                             |
|`my_daf[2].to_list()`                      | select row 2, including all columns, return a list                        |
|`my_daf[-1, :]`                            | select the last row                                                       |
|`my_daf[:5]`                               | select first 5 rows; like `head()` in other dataframe packages.           |
|`my_daf[:-5]`                              | select last 5 rows; like `tail()` in other dataframe packages.            |
|`my_daf[:, 3]`                             | select only column 3, including all rows. use .to_list() to return a list |
|`my_daf[:, 'C']`                           | return only column named 'C', including all rows.                         |
|`my_daf[2:4]`                              | return rows 2 and 3, including all columns as daf array                   | 
|`my_daf[2:4, :]`                           | same as above                                                             |
|`my_daf[:, 3:5]`                           | return columns 3 and 4, including all rows as daf array                   |
|`my_daf[:, range(2,6)]`                    | return columns 2,3,4,5, including all rows as daf array                   |
|`my_daf[[2,4,6]]`                          | return rows with indices 2,4,6 as daf array.                              |
|`my_daf[range(2,6)]`                       | return rows with indices 2,3,4,5 as daf array.                            |
|`my_daf[[range(1,10), range(46,50)]]`      | return rows with indices in list of ranges provided                       |
|`my_daf[:, [1,3,5]]`                       | return columns with indices 1,3,5 as daf array.                           |
|`my_daf[:, [range(1,10), range(46,50)]]`   | return columns with indices in list of ranges provided                    |
|`my_daf[['row5','row6','row7']]`           | return rows with keyfield values 'row5','row6','row7'                     |
|`my_daf[:, ['col1', 'col3', 'col5']]`      | return columns with column names 'col1', 'col3', 'col5'                   |
|`my_daf[('row5','row49'), :]]`             | return rows with keyfield values 'row5' through 'row49' inclusive (note: column idx ':' is required)  |
|`my_daf[('row5',), :]]`                    | return rows with keyfield values 'row5' through the end (note: column idx ':' is required)   |
|`my_daf[(,'row49'), :]]`                   | return rows with keyfield values from the first row through 'row49' inclusive (note: column idx is required)
|`my_daf[:, ('col5', 'col23')]]`            | return columns with column names from 'col5', through 'col23' inclusive   |
|`my_daf[:, (, 'col23')]]`                  | return columns with column names from the first column through 'col23' inclusive  |
|`my_daf[:, ('col23',)]]`                   | return columns with column names from 'col23', through the end            |


Please note that if you want to index rows by a keyfield or index columns using column names that are integers, 
then you must use method calls. The square-bracket indexing will assume any integers are indices, not names.
The integer values shown in the examples below do not index the array directly, but choose the row or columns by name.
To choose by row keys (krows), then keyfield must be set. To choose by column keys (kcols), cols must be set.

The attribute `keyfield` will be propagated to a returned daf instance if the original `keyfield` column still exists in the selected column set.

|  Expression                               | Operation                                                                     |
|:------------------------------------------|:------------------------------------------------------------------------------|
|`my_daf.select_records_daf(123)`           | return daf with one row with integer 123 in keyfield column.                  |
|`my_daf.select_krows(krows = 123)`         | same as above.                                                                |
|`my_daf.select_krows(krows = [123, 456])`  | return daf with two rows selected by with integers in the keyfield column.    |
|`my_daf.select_krows(krows = [123, 456], inverse=True)`    | return daf dropping two rows selected by with integers in the keyfield column.  |
|`my_daf.select_krows(krows = (123, ), inverse=True)`       | drop all rows starting with row named 123 to the end.         |
|`my_daf.select_krows(krows = (, 123), inverse=True)`       | drop all rows from the first through row named 123.           |
|          ------                                           |          ------                                               |
|`my_daf.select_kcols(kcols = 123)`                         | return daf of column named 123 (integer), placed in col 0     |
|`my_daf.select_kcols(kcols = 123).to_list()`               | return list of column named 123 (integer).                    |
|`my_daf.select_kcols(kcols = 123).to_list(unique=True)`    | return list with one column with integer 123 colname, and remove duplicates.  |
|`my_daf.select_kcols(kcols = 123, inverse=True)`           | drop column with name 123                                     |
|`my_daf.select_kcols(kcols = 123, inverse=True, flip=True)`    |drop column with name 123 and transpose columns to rows.   |

There are also similar methods for selecting rows and cols by indexes. Selecting rows using select_irows(rows_spec) is the same as my_daf[row_spec],
except the parameter inverse is available to drop rows rather than keeping them.

|  Expression                                               | Operation                                                                     |
|:----------------------------------------------------------|:------------------------------------------------------------------------------|
|`my_daf.select_irows(irows=10)`                            | select single row 10. Same as `my_daf[10]`.                                   |
|`my_daf.select_irows(irows=10, inverse=True)`              | drop single row 10. Same as `my_daf[10]`.                                     |
|`my_daf.select_irows(irows=[1,2,3,45])`                    | select rows 1,2,3, and 45 using indices. Same as `my_daf[[1,2,3,45]]`.        |
|`my_daf.select_irows(irows=slice(20,,2))`                  | select rows starting at row 20 through the end and skip every other row. Same as `my_daf[20::2]`  |
|          ------                                           |          ------                                                   |
|`my_daf.select_icols(icols=slice(4,10))`                   | select columns 4 thorugh 9 (inclusive). Same as `my_daf[:, 4:10]`             |
|`my_daf.select_icols(icols=slice(4,10), flip=True)`        | select columns 4 thorugh 9 (inclusive) and transpose columns to rows.         |
|`my_daf.select_icols(flip=True)`                           | select all columns and transpose columns to rows.                             |         
          
### Indexing: setting values in a daf:
Similar indexing is used when setting values in the array. 

- If the value is a single value, it is repeated in all cells of the selection.
- If the value is a list, then it is applied to the selection in the order provided.
- If the value is a dict, and a row is selected, the column names will be respected.

Here are some examples.

|  Expression                                               | Operation                                                                     |
|:----------------------------------------------------------|:------------------------------------------------------------------------------|
|`my_daf[irow] = list`                                      | assign the entire row at index irow to the list provided                      |
|`my_daf[irow] = value`                                     | assign the entire row at index row to the single value provided.              |
|`my_daf[irow, icol] = value`                               | set cell irow, icol to value, where irow, icol are integers.                  |
|`my_daf[irow, start:end] = value`                          | set a value in cells in row irow, from columns start to end.                  |
|`my_daf[irow, start:end] = list`                           | set values from a list in cells in row irow, from columns start to end.       |
|`my_daf[irow, range] = list`                               | set values from a list in cells in row irow, from columns in range.           |
|`my_daf[:, icol] = list`                                   | assign the entire column at index icol to the list provided.                  |
|`my_daf[start:end, icol] = list`                           | assign a partial column at index icol to list provided.                       |
|`my_daf[irow, colname] = value`                            | set a value in cell irow, col, where colname is a string.                     |
|`my_daf[:, colname] = list`                                | assign the entire column colname to the list provided.                        |
|`my_daf[start:end, colname] = list`                        | assign a partial column colname to list provided from rows start to end.      |


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

    record = my_daf.select_record(keyvalue)
    
or

    record_list = my_daf[keyvalue].to_list()
   
Note that this syntax differs from Pandas, which normally references a column if square brackets are used with no other
syntax.

#### select multiple records using the keyfield and return a daf.

    new_daf = my_daf[keyvalue_list]
    

#### drop a record using keyfield

    new_daf = select_krows(krows=keyval, invert=True)
    
    
#### remove multiple records using multiple keys in a list.
    
    new_daf = my_daf.select_krows(krows=keylist, invert=True)

### selecting records without using keyfield

#### select records based on a conditional expression.

    new_daf = my_daf.select_where(lambda row: row['fieldname'] > 5)

or

    new_daf = my_daf.select_where(lambda row: row['fieldname'] > row_limit)
    
#### Select one record from daf using the idx and return as a dict.
    
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

    my_daf.select_kcols(colnames_ls, invert=True)
    
#### return dict of sums of columns specified or those specified in dtypes as int or float if numeric_only is True.

    sum_da = my_daf.sum(colnames_ls, numeric_only=False)

#### create cross-column lookup

Given a daf with at least two columns, create a dict of list lookup where the key are values in col1 and list of values are 
unique values in col2. Values in cols must be hashable.

    lookup_dol = my_daf.cols_to_dol(colname1, colname2)                        

#### count discrete values in a column or columns

create dictionary of values and counts for values in colname or colnames.
sort by count if sort, from highest to lowest unless 'reverse'

    valuecounts_di = my_daf.valuecounts_for_colname(colname, sort=False, reverse=True)
    valuecounts_dodi = my_daf.valuecounts_for_colnames_ls(colnames_ls, sort=False, reverse=False)

#### same as above but also selected by a second column

    valuecounts_di = my_daf.valuecounts_for_colname_selectedby_colname(colname)
    valuecounts_dodi = my_daf.valuecounts_for_colname_groupedby_colname(colname)
    
#### group to dict of daf based on values in colname

    my_dodaf = my_daf.group_to_dodaf(colname)    
    
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

```
    example_daf = Daf(cols=['A', 'B', 'C'], 
                        lol=[ [1,  2,   0],
                              [4,  5,   0],
                              [7,  8,   0],
                              [0,  0,   0]])
                              
    formulas_daf = Daf(cols=['A', 'B', 'C'], 
            lol=[['',                    '',                    "sum($d[$r,:$c])"],
                 ['',                    '',                    "sum($d[$r,:$c])"],
                 ['',                    '',                    "sum($d[$r,:$c])"],
                 ["sum($d[:$r,$c])",     "sum($d[:$r,$c])",     "sum($d[:$r, $c])"]]
                 )
                 
    example_daf.apply_formulas(formulas_daf)

    result       = Daf(cols=['A', 'B', 'C'], 
                        lol=[ [1,  2,   3],
                              [4,  5,   9],
                              [7,  8,   15],
                              [12, 15,  27]])
                              
## Join Overview
The `join` method in Daf enables combining data from two tables (Daf instances) based on a shared keyfield. 
This functionality is inspired by SQL joins, supporting `inner`, `left`, `right`, and `outer` join types, 
but with additional flexibility for customization.

### Join Types
- **Inner Join**: Includes only rows with matching keys in both tables.
- **Left Join**: Includes all rows from the left table and matching rows from the right table. Missing values in the right table are filled with `None`.
- **Right Join**: Includes all rows from the right table and matching rows from the left table. Missing values in the left table are filled with `None`.
- **Outer Join**: Includes all rows from both tables, filling missing values with `None`.

### Daf Join API
#### Usage
```python
result_daf = daf1.join(
    other_daf=daf2,          # The other Daf instance
    how="inner",             # Join type: 'inner', 'left', 'right', 'outer'
    shared_fields=["id"],    # Optional: List of fields to exclude from conflict resolution
    custom_translator_daf=None,  # Optional: Custom mapping for column resolution
    diagnose=True            # Enable diagnostics for debugging
)```

#### Operation
Daf instances joined must have keyfields defined and they must have values that can be used to join the records.

If there are any fields that exist with the same names between the two tables, then they are suffixed with the
name of the source tables, or _0 and _1 if no names are defined.

If fields are shared between the two tables and should not be differentiated, then those can be listed as 'shared fields'
and they will occur only once in the joined table.

Using 'join' with memory-based tables will create a new table instance. If used with SQL tables, then it creates a view.
    
## Comparison with Pandas, Numpy SQLite

One of the primary reasons for developing Daffodil is that 
conversion to Pandas directly from arbitrary Python list-of-dicts, for example, is surprisingly slow.

We timed the various conversions using Pandas 1.5.3 and 2.4.1.

See: https://github.com/raylutz/daffodil/blob/main/docs/daf_benchmarks.md

Daffodil is faster than Pandas for array manipulation (inserting rows (300x faster) and cols (1.4x faster)), 
performing actions on individual cells (5x faster), appending rows (which Pandas essentially outlaws), 
and performing keyed lookups (8.4x faster). Daffodil arrays are smaller whenever any strings are included 
in the array by about 3x over Pandas. While Pandas and Numpy are faster for columnar calculations, 
Numpy is always faster if columns are all numeric data. Daffodil is larger than purely numeric arrays than
Numpy and Pandas but if string columns are included, Pandas will likely be 3x larger. Numpy does not handle
string columns mixed with numeric data.

Therefore it is a good strategy to use Daffodil for all operations except for pure data manipulation, 
and then port the appropriate columns to NumPy.


## Demo

See this demo of Daffodil functionality.

https://github.com/raylutz/daffodil/blob/main/docs/daf_demo.md

## Syntax Comparison with Pandas

Below is a sample of equivalent functions between Pandas and Daffodil. Please note that Daffodil does not attempt to create column-oriented functions such as
.add, .sub, .mul, .div, .mod, .pow, etc which are either available on row basis using Python apply() or reduce() or by porting the array to NumPy.

|  Pandas                                           |   Daffodil                               |Description     |
|:--------------------------------------------------|:-----------------------------------------|:---------------|
|`df = pd.DataFrame()`                              |`daf = Daf()`                             |Create empty dataframe  |
|`df.index`                                         |`daf.keys()`                              |row labels of the Dataframe that "stick" to the data  |
|`df.columns`                                       |`daf.columns()`                           |list of column names      |
|`df.dtypes`                                        |`daf.dtypes`                              |data types dictionary     |
|`df.select_dtypes()`                               |`cols = daf.calc_icols()`                 |calculate the columns to be included based on data types.  |
|`df.to_numpy()`                                    |`daf.to_numpy(cols)`                      |convert selected columns to NumPy array  |
|`df.shape`                                         |`daf.shape()`                             |return (rows, cols) dimension  |
|`df.empty`                                         |`bool(daf)`                               |`bool(daf)` also will allow None to be detected  |
|`df.convert_dtypes()`                              |`daf.apply_dtypes()`                      |convert columns to the datatypes specified in self.dtypes dict  |
|`df.head(n)`                                       |`daf[:n]`                                 |return first n rows, default is 5             |
|`df.insert()`                                      |`daf.insert_icol(), .insert_col()`        |insert a column at a specified location in the array  |
| -- (not available)                                |`daf.insert_irow()`                       |insert a row at specified row location (not supported by Pandas)  |
|`df.concat()`                                      |`daf.append()`                            |add one or many rows/cols   |
| -- (deprecated)                                   |`daf.append()`                            |add one row    |
|`df[colname]`                                      |`daf[:, colname]`                         |select one column   |
|`df.apply()`  (not recommended)                    |`daf.apply()`, `.manifest_apply()`        |apply function to a row at a time.    |
|`df.map()`                                         |`daf.apply()`                             |any arbitrary python function can be applied  |
|`df.agg()`                                         |`daf.reduce()`                            |reduce array to a record using arbitrary function  |
|`df.transform()`                                   |`daf.apply(by='table')`                   |transform a table using a function producing a table   |
|`df.groupby()`                                     |`daf.groupby(); groupby_cols(); groupby_cols_reduce; groupby_reduce()`  |group rows by values in a column and poss. apply a reduction  |
|`df.value_counts()`                                |`daf.value_counts()`                      |reduce a column by counting values.  |
|`df.tail(n)`                                       |`daf[-n:]`                                |return last n rows  |
|`df.sort_values()`                                 |`daf.sort_by_colname()`                   |Daf only sorts rows   |
|`df.transpose()`                                   |`daf.select_kcols(flip=True)`             |Not recommended in Daffodil but free if columns are dropped   |
|`df.drop()`                                        |`daf.select_kcols(kcols=[colnames], inverse=True)`         |Drop columns by name. Transpose is free in Daf if done during a drop  |
|`df[~df[keyfield].isin(list of keys)]`             |`daf.select_krows(krows=[list of keys], inverse=True)`     |Drop rows by keys in the keyfield.   |
|`df.to_records()`                                  |`daf.to_lod(); .to_dod()`                  |convert from array to records in list-of-dict (or dict-of-dict) format.  |
|`df.to_markdown()`                                 |`daf.to_md()`                              |convert to markdown representation. default presentation in Daf  |
|`df.assign()`                                      |`daf[:, n] = new_col`                      |assign new values to a column  |
| -- (not available?)                               |`daf[rowname or idx] = dict`               |assign new values to a row and respect column names as dict keys  |
|`df[df[colname] > 5]`                              |`daf.select_where(lambda row: row[colname] > 5)`           |select rows where the value in colname > 5   |
|`df.rename(renaming dict)`                         |`daf.rename_cols(); daf.set_cols(); daf.set_rowkeys()`     |Daf allows renaming rows when keyfield=''  |
|`df.reset_index`                                   |`daf.set_keyfield(''); daf.set_rowkeys()`  |similar in operation.   |
|`df.set_index`                                     |`daf.set_keyfield(keyfieldname)`           |Daf can use an existing column for the keyfield or can set the rowkeys independently  |
|`df.truncate()`                                    |`daf[:n]; daf[n:]`                         |Truncate before or after some index n.  |
|`df.replace()`                                     |`daf.find_replace(find_pat, replace_val)`  |Replace values found in-place.   |
|`df.merge()`                                       |`daf.join(other_daf, how, shared_cols)`    |Join two tables using inner, outer, right, left  |



## Conclusion and Summary

The general conclusion is that it generally beneficial to use Daffodil instead of lod or pandas df for
general-purpose table manipulation, but when number crunching is needed, prepare the data with Daffodil 
and convert directly to Numpy or Pandas. 

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



