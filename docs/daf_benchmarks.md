# Evaluate conversion and calculation time tradeoffs between Pandas, Daffodil, Numpy, etc.
    
The Daffodil package was originally developed because we found we were using list-of-dict (lod)
    structure to build a table suitable for porting into Pandas, because Pandas is slow in performing
    row-by-row appends. The question was originally when was it beneficial to convert to Pandas to
    take advantage of its faster C implementation vs. continuing to use Python structures. What we
    found was that converting from Python lod to Pandas DataFrame was very slow and unless we needed
    to perform many repeated column-based calculations, we may as well leave it as Python and not
    suffer the conversion time. The Daffodil Package (DAta Frames For Optimized Data Inspection and Logical operations)
    was created to allow for data structures that are 1/3 the size of a lod, allowing for fast appends,
    while also providing for a rich indexing syntax to reference rows and columns, and also support 
    column and row keys using fast lookups provided by Python dictionaries. 
    
This series of tests compares the speed of converting Python lod structure to Pandas and Numpy,
    and also compares a few other operations. Certainly, Pandas is a good choice for anyone doing
    interactive data exploration and where the user is already familiar with Pandas syntax. The 
    benefit of Daffodil is for those programmers who are using Pandas as a convenient tabular 
    representation in repetitive data conversion programs used in pipelines, ETL applications, etc.
    and even in big-data applications where the data can be operated on in chunks, but where the
    entirety of the data cannot fit in memory at the same time. Daffodil is competitive with mixed
    data types, particularly string data. Daffodil can be used to prep the data for Pandas or Numpy
    using fast row appends while leveraging the faster column processing of Pandas or Numpy
    for numeric columns that will benefit.
    
    
## Create sample_lod

Here we generate a table using a python list-of-dict structure,
        with 1000 columns and 1000 rows, consisting of 1 million
        integers from 0 to 100. Set the seed to an arbitrary value for
        reproducibility. Also, create other forms similarly or by converting
        the sample_lod. We will show the table once it is converted to other forms.

```python
    np.random.seed(42)  # For reproducibility
    sample_lod = [dict(zip([f'Col{i}' 
                    for i in range(num_columns)], 
                        np.random.randint(0, 100, num_columns))) 
                            for _ in range(num_rows)]
                            
    # build a sizeof_di dict                        
    sizeof_di = {}
    sizeof_di['lod'] = safe_sizeof(sample_lod)
    md_report += pr(f"\n\nGenerated sample_lod with {len(sample_lod)} records\n"
                    f"- {sizeof_di['lod']=:,} bytes\n\n")
```




Generated sample_lod with 1000 records
- sizeof_di['lod']=54,040,856 bytes


## Create sample_klod

sample_klod is similar to sample_lod but it has a first column 
        'rowkey' is a string key that can be used to look up a row. Each rowkey is
        simply 'ColN', where N is the row number. Please note that this is different
        from indexing the rows (which is always tied to the position) as the rowkeys 
        are tied to the row, even if the order is changed.

```python
    sample_klod = [dict(zip(['rowkey']+[f'Col{i}' 
                    for i in range(num_columns)], 
                        [str(i)] + list(np.random.randint(1, 100, num_columns)))) 
                            for i in range(num_rows)]

    sizeof_di['klod'] = safe_sizeof(sample_klod)
    md_report += pr(f"\n\nGenerated sample_klod with {len(sample_klod)} records\n"
                 f"- {sizeof_di['klod']=:,} bytes\n\n")
```




Generated sample_klod with 1000 records
- sizeof_di['klod']=54,092,746 bytes


## Create daf from sample_lod

Here we simply create a Daffodil DataFrame 'sample_daf' of the same random data.
        The Daffodil DataFrame core datatype is List[List[Any]], i.e. a list of lists of anything, 
        and iwll be about 1/3 the size of an equivalent list-of-dict structure because the
        keys for each dictionary are not repeated.

```python
    sample_daf = Daf.from_lod(sample_lod)
    sizeof_di['daf'] = safe_sizeof(sample_daf)
    md_report += pr(f"daf:\n{sample_daf}\n\n"
                    f"{sizeof_di['daf']=:,} bytes\n\n")
```


daf:
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

\[1,000 rows x 1,000 cols; keyfield=''; 0 keys ] (Daf)


sizeof_di['daf']=36,119,290 bytes


## Create kdaf from sample_klod

Similarly, we create the keyed daf table by converting the sample_klod structure

```python
    sample_kdaf = Daf.from_lod(sample_klod, keyfield='rowkey')
    sizeof_di['kdaf'] = safe_sizeof(sample_kdaf)
    md_report += pr(f"kdaf:\n{sample_kdaf}\n\n"
                    f"{sizeof_di['kdaf']=:,} bytes\n\n")
```


kdaf:
| rowkey | Col0 | Col1 | Col2 | Col3 | ... | Col995 | Col996 | Col997 | Col998 | Col999 |
| -----: | ---: | ---: | ---: | ---: | --: | -----: | -----: | -----: | -----: | -----: |
|      0 |   26 |   56 |   78 |   29 | ... |     62 |     74 |     69 |     75 |     20 |
|      1 |   46 |   72 |   39 |   48 | ... |     29 |     68 |     90 |     43 |      2 |
|      2 |   27 |    4 |   19 |   82 | ... |     17 |     31 |     28 |     13 |     79 |
|      3 |   18 |    2 |   49 |   30 | ... |      3 |     83 |     12 |     97 |     13 |
|      4 |   84 |   69 |    9 |   18 | ... |     53 |      2 |     19 |     34 |     41 |
|    ... |  ... |  ... |  ... |  ... | ... |    ... |    ... |    ... |    ... |    ... |
|    995 |   11 |   75 |   33 |   98 | ... |     85 |     15 |     13 |      5 |     28 |
|    996 |   73 |    2 |   59 |   13 | ... |     29 |     24 |     63 |     71 |     27 |
|    997 |   81 |   74 |   63 |   79 | ... |      3 |     41 |     33 |     74 |     62 |
|    998 |   98 |   76 |   29 |   51 | ... |     55 |     59 |     30 |      3 |     47 |
|    999 |   52 |   59 |   80 |   13 | ... |     58 |     33 |     33 |     90 |     89 |

\[1,000 rows x 1,001 cols; keyfield='rowkey'; 1,000 keys ] (Daf)


sizeof_di['kdaf']=36,234,035 bytes


## Create Pandas df

Here we use an unadorned basic pre-canned Pandas function to construct the dataframe,
        but to make sure it may take advantage of the fact that all data is integers, we provide
        also the dtype=int parameter. As it turns out, the performance does not change either way.
        We can note here that this DataFrame is more efficient in terms of space than the Daffodil
        object by a factor of about 4. But as we will see once we start timing these, the conversion
        is quite slow.

```python
    df = pd.DataFrame(sample_lod, dtype=int)
    sizeof_di['df'] = safe_sizeof(df)

    md_report += pr(f"\n\n```{df}\n```\n\n"
                    f"{sizeof_di['df']=:,} bytes\n\n")
```




```     Col0  Col1  Col2  Col3  Col4  Col5  Col6  Col7  Col8  Col9  Col10  ...  Col989  Col990  Col991  Col992  Col993  Col994  Col995  Col996  Col997  Col998  Col999
0      51    92    14    71    60    20    82    86    74    74     87  ...      24      16      12      83      24      67       9      66      17      99      85
1      33     7    39    82    41    40     5    51    25    63     97  ...      44      88      32      40       7      10      85      50      87      40      16
2      75    45    31    78    79    53    85    91    19    32     73  ...      82      57       3       3      19       9      23      98      25      36      84
3      53    20    73    37    45     3    59    56    44    19     16  ...      44      60      82      15      65      39      16      33      15      59      65
4      65    89    12    55    30    33    38    66     7    86     77  ...      45      31      54      95       4      35      48      57      38      79      96
..    ...   ...   ...   ...   ...   ...   ...   ...   ...   ...    ...  ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...
995    47    57    85    63    23    69     1    88    15    50     95  ...      35      44      65      18      80      70      27      71      55      97      56
996    71    48    29    19    43    52    13     3    34    40      3  ...      21      18      22      45      39      17      70      76      80      64       8
997    37     4    96    39    82    47    53    83    49    64     72  ...      59       9      77       4      26      86      21      17      31      32      20
998    23    39    77     9    21    61     2    43    55    59      3  ...       3      19      71      86      76      16       0      63      22      81      97
999    86     9    27     2    40     3    66    51    94    90     23  ...      37      60      85       7       9       5      86      34      61      77      52

[1000 rows x 1000 columns]
```

sizeof_di['df']=14,021,624 bytes


## Create Pandas csv_df from Daf thru csv

Create a Pandas DataFrame by convering Daf through a csv buffer.

    We found tha twe could save a lot of time by converting the data to a csv buffer and then 
    importing that buffer into Pandas. This does not make a lot of sense, but it is true.
    But it is slightly more wasteful in terms of space than the direct conversion.

```python
    csv_df = sample_daf.to_pandas_df(use_csv=True)
    sizeof_di['csv_df'] = safe_sizeof(csv_df)

    md_report += pr(f"\n\n```{csv_df}\n```\n\n"
                    f"{sizeof_di['csv_df']=:,} bytes\n\n")
```




```     Col0  Col1  Col2  Col3  Col4  Col5  Col6  Col7  Col8  Col9  Col10  ...  Col989  Col990  Col991  Col992  Col993  Col994  Col995  Col996  Col997  Col998  Col999
0      51    92    14    71    60    20    82    86    74    74     87  ...      24      16      12      83      24      67       9      66      17      99      85
1      33     7    39    82    41    40     5    51    25    63     97  ...      44      88      32      40       7      10      85      50      87      40      16
2      75    45    31    78    79    53    85    91    19    32     73  ...      82      57       3       3      19       9      23      98      25      36      84
3      53    20    73    37    45     3    59    56    44    19     16  ...      44      60      82      15      65      39      16      33      15      59      65
4      65    89    12    55    30    33    38    66     7    86     77  ...      45      31      54      95       4      35      48      57      38      79      96
..    ...   ...   ...   ...   ...   ...   ...   ...   ...   ...    ...  ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...
995    47    57    85    63    23    69     1    88    15    50     95  ...      35      44      65      18      80      70      27      71      55      97      56
996    71    48    29    19    43    52    13     3    34    40      3  ...      21      18      22      45      39      17      70      76      80      64       8
997    37     4    96    39    82    47    53    83    49    64     72  ...      59       9      77       4      26      86      21      17      31      32      20
998    23    39    77     9    21    61     2    43    55    59      3  ...       3      19      71      86      76      16       0      63      22      81      97
999    86     9    27     2    40     3    66    51    94    90     23  ...      37      60      85       7       9       5      86      34      61      77      52

[1000 rows x 1000 columns]
```

sizeof_di['csv_df']=26,021,624 bytes


## Create keyed Pandas df

Create a keyed Pandas df based on the sample_klod generated.
        This object has one column which provides a string row key for looking up a row.
        Please note this takes far more memory than a Pandas df without this
        str column, almost 3x the size of Daf instance with the same data. To test fast lookups,
        we also use set_index to get ready for fast lookups so we can compare with Daffodil lookups.
        Daffodil uses a very fast dictionary lookup, and is faster than Pandas.

```python
    kdf = pd.DataFrame(sample_klod)
    
    # also set the rowkey as the index for fast lookups.
    kdf.set_index('rowkey', inplace=True)
    
    sizeof_di['kdf'] = safe_sizeof(kdf)
        
    md_report += pr(f"\n\n```{kdf}\n```\n\n")
    md_report += pr(f"- {sizeof_di['kdf']=:,} bytes\n\n")
```




```        Col0  Col1  Col2  Col3  Col4  Col5  Col6  Col7  Col8  Col9  Col10  ...  Col989  Col990  Col991  Col992  Col993  Col994  Col995  Col996  Col997  Col998  Col999
rowkey                                                                     ...                                                                                        
0         26    56    78    29    56    83    63    16    79    65     14  ...      20      40      67      65      17      57      62      74      69      75      20
1         46    72    39    48     3    13    65    92    83    73      5  ...       6       3      55      69      58      53      29      68      90      43       2
2         27     4    19    82    87    85    15    49     6    41     59  ...       6      40      98      59       8      83      17      31      28      13      79
3         18     2    49    30     1    74    39    78    17     2     13  ...      27      77      36      65      99      47       3      83      12      97      13
4         84    69     9    18    18    47    60    57    48     9     41  ...      74      90      84      21      23      63      53       2      19      34      41
...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...    ...  ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...
995       11    75    33    98     9    21    84    42    75     3     54  ...      58      63      63      86      52       4      85      15      13       5      28
996       73     2    59    13    29     5    24     1    30    98     34  ...      92      42      46      10      96      75      29      24      63      71      27
997       81    74    63    79    90    19    43    73    73    19     57  ...      23      72      74      16      67      81       3      41      33      74      62
998       98    76    29    51    99    49    99    56    12    86     49  ...      24      86      51      83       3      16      55      59      30       3      47
999       52    59    80    13    36    26    74    71     2    69     87  ...      41      35      20      63       1      25      58      33      33      90      89

[1000 rows x 1000 columns]
```

- sizeof_di['kdf']=69,891,679 bytes


## create hdnpa from lod

A hdnpa is a Numpy array with a header dictionary. The overall size is about the same as just the NumPy array,
        but it provides column names to be comparable with the DataFrame form. However, we must remind the reader that
        the numpy array must be a uniform data type.

```python
    hdnpa = lod_to_hdnpa(sample_lod)
    sizeof_di['hdnpa'] = safe_sizeof(hdnpa)
    md_report += pr(f"{sizeof_di['hdnpa']=:,} bytes\n\n")
```


sizeof_di['hdnpa']=4,054,216 bytes


## Create lont from lod

We also tried a structure based on a list of named tuples. This is very slow and does not provide any savings.

```python
    lont = lod_to_lont(sample_lod)
    sizeof_di['lont'] = safe_sizeof(lont)
    md_report += pr(f"{sizeof_di['lont']=:,} bytes\n\n")
```


sizeof_di['lont']=36,048,856 bytes


## Create hdlot from lod

Another option is a list of tuples with a header dictionary. This is also slow and no space savings..

```python
    hdlot = lod_to_hdlot(sample_lod)
    sizeof_di['hdlot'] = safe_sizeof(hdlot)
    md_report += pr(f"{sizeof_di['hdlot']=:,} bytes\n\n")
```


sizeof_di['hdlot']=36,102,944 bytes


## Create sqlite_table from klod

Converting to a sqlite table is surprisingly fast as it beats creating a Pandas dataframe with this data.
    The space taken in memory is hard to calculate and the method we used to calculate it would produce 0.

```python
    daf_sql.lod_to_sqlite_table(sample_klod, table_name='tempdata1')

    datatable1 = 'tempdata1'
    datatable2 = 'tempdata2'
    sizeof_di['sqlite'] = os.path.getsize(datatable1+'.db')
    md_report += pr(f"{sizeof_di['sqlite']=:,} bytes\n\n")
```


sizeof_di['sqlite']=5,148,672 bytes


## Create table of estimated memory usage for all types

use Daf.from_lod_to_cols to create a table with first colunm key names, and second column values. 
        We will update this table using Daffodil indexing to provide the timing for all tested combinations.

```python
    all_sizes_daf = Daf.from_lod_to_cols([sizeof_di], cols=['Datatype', 'Size in Memory (bytes)'])
    md_report += all_sizes_daf.to_md(smart_fmt=True)
```


| Datatype | Size in Memory (bytes) |
| -------: | ---------------------: |
|      lod |             54,040,856 |
|     klod |             54,092,746 |
|      daf |             36,119,290 |
|     kdaf |             36,234,035 |
|       df |             14,021,624 |
|   csv_df |             26,021,624 |
|      kdf |             69,891,679 |
|    hdnpa |              4,054,216 |
|     lont |             36,048,856 |
|    hdlot |             36,102,944 |
|   sqlite |              5,148,672 |

## Times for conversions and operations

This secion uses the timeit() function to time conversions.
        For each conversion, the time wil be added to the (datatype)_times dicts.

```python
    setup_code =
```

import pandas as pd
import numpy as np
from collections import namedtuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'src'))
from daffodil.daf import Daf
from daffodil.lib import daf_sql
import gc
gc.disable()

'''

    loops = 10
    report_cols = [ 'Attribute',            'daf', 'pandas', 'numpy', 'sqlite', 'lod'] #, 'note']
    report_attrs = [ 
                    'from_lod', 
                    'to_pandas_df', 
                    'to_pandas_df_thru_csv', 
                    'from_pandas_df', 
                    'to_numpy',
                    'from_numpy',
                    'increment cell',
                    'insert_irow',
                    'insert_icol',
                    'sum cols',
                    'sample_daf.daf_sum()',
                    'sample_daf.daf_sum2()',
                    'sample_daf.daf_sum3()',
                    'sum_np',
                    'transpose',
                    #'transpose_keyed',
                    'keyed lookup',
                    '=====',
                    'Size of 1000x1000 array (MB)',
                    'Size of keyed 1000x1000 array (MB)',
                  ]
    report_daf = Daf(cols=report_cols, keyfield='Attribute')
    for attr in report_attrs:
        report_daf.append({'Attribute': attr})
    

    #report_daf['from_lod', 'loops']    = loops
    report_daf['from_lod', 'daf']           = ms = timeit.timeit('Daf.from_lod(sample_lod)',  setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"Daf.from_lod()                  {ms:.4f} ms")

    report_daf['from_lod', 'pandas']        = ms = timeit.timeit('pd.DataFrame(sample_lod)',   setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    report_daf['to_pandas_df', 'lod']       = ms
    print(f"lod_to_df() plain               {ms:.4f} ms")

    report_daf['from_lod', 'numpy']         = ms = timeit.timeit('lod_to_hdnpa(sample_lod)', setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    report_daf['to_numpy', 'lod']           = ms
    print(f"lod_to_numpy()                  {ms:.4f} ms")

    report_daf['from_lod', 'sqlite']        = ms = timeit.timeit('daf_sql.lod_to_sqlite_table(sample_klod, table_name=datatable2)', setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"lod_to_sqlite_table()           {ms:.4f} ms")

    #-------------------------------

    #report_daf['to_pandas_df', 'loops']    = loops
    report_daf['to_pandas_df', 'daf']       = ms = timeit.timeit('sample_daf.to_pandas_df()', setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"daf.to_pandas_df()              {ms:.4f} ms")

    #report_daf['to_pandas_df_thru_csv', 'loops']   = loops
    report_daf['to_pandas_df_thru_csv', 'daf']    = ms = timeit.timeit('sample_daf.to_pandas_df(use_csv=True)', setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"daf.to_pandas_df(use_csv=True)  {ms:.4f} ms")

    #report_daf['from_pandas_df', 'loops']  = loops
    report_daf['from_pandas_df', 'daf']     = ms = timeit.timeit('Daf.from_pandas_df(df)', setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"Daf.from_pandas_df()            {ms:.4f} ms")

    #report_daf['to_numpy', 'loops']        = loops
    report_daf['to_numpy', 'daf']           = ms = timeit.timeit('sample_daf.to_numpy()',            setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"daf.to_numpy()                  {ms:.4f} ms")

    #report_daf['from_numpy', 'loops']      = loops 
    report_daf['from_numpy', 'daf']         = ms = timeit.timeit('Daf.from_numpy(hdnpa[1])',  setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"Daf.from_numpy()                {ms:.4f} secs")

    report_daf['to_pandas_df', 'numpy']     = ms = timeit.timeit('pd.DataFrame(hdnpa[1])',  setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    report_daf['from_numpy', 'pandas']      = ms
    print(f"numpy to pandas df              {ms:.4f} ms")

    report_daf['from_pandas_df', 'numpy']   = ms = timeit.timeit('df.values',  setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    report_daf['to_numpy', 'pandas']        = ms
    print(f"numpy from pandas df            {ms:.4f} ms")


    print("\nMutations:")

    sample_daf.retmode = 'val'

    #report_daf['increment cell', 'loops']   = 
    increment_loops = loops * 100
    report_daf['increment cell', 'daf']     = ms = timeit.timeit('sample_daf[500, 500] += 1', setup=setup_code, globals=globals(), number=increment_loops) * 1000 / (increment_loops)
    gc.enable()
    print(f"daf[500, 500] += 1              {ms:.4f} ms")

    report_daf['increment cell', 'pandas']  = ms = timeit.timeit('df.at[500, "Col500"] += 1', setup=setup_code, globals=globals(), number=increment_loops) * 1000 / (increment_loops)
    gc.enable()
    print(f"df.at[500, 'Col500'] += 1       {ms:.4f} ms")

    #report_daf['insert_irow', 'loops']     = 
    insert_loops = loops * 10
    report_daf['insert_irow', 'daf']        = ms = timeit.timeit('sample_daf.insert_irow(irow=400, row=sample_daf[600, :].copy())', setup=setup_code, globals=globals(), number=insert_loops) * 1000 / (insert_loops)
    gc.enable()
    print(f"daf.insert_irow                 {ms:.4f} ms")

    report_daf['insert_irow', 'pandas']     = ms = timeit.timeit('pd.concat([df.iloc[: 400], pd.DataFrame([df.loc[600].copy()]), df.iloc[400:]], ignore_index=True)', setup=setup_code, globals=globals(), number=insert_loops)  * 1000/ (insert_loops)
    gc.enable()
    print(f"df insert row                   {ms:.4f} ms")

    #report_daf['insert_icol', 'loops']     = 
    insert_loops = loops * 10
    report_daf['insert_icol', 'daf']        = ms = timeit.timeit('sample_daf.insert_icol(icol=400, col_la=sample_daf[:, 600].copy())', setup=setup_code, globals=globals(), number=insert_loops)  * 1000/ (insert_loops)
    gc.enable()
    print(f"daf.insert_icol                 {ms:.4f} ms")

    report_daf['insert_icol', 'pandas']     = ms = timeit.timeit("pd.concat([df.iloc[:, :400], pd.DataFrame({'Col600_Copy': df['Col600'].copy()}), df.iloc[:, 400:]], axis=1)", setup=setup_code, globals=globals(), number=insert_loops) * 1000 / (insert_loops)
    gc.enable()
    print(f"df insert col                   {ms:.4f} ms")

    print("\nTime for sums:")

    #report_daf['sum cols', 'loops']        = loops
    report_daf['sum cols', 'pandas']        = ms = timeit.timeit('cols=list[df.columns]; df[cols].sum().to_dict()', setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"df_sum_cols()                   {ms:.4f} ms")

    report_daf['sum cols', 'daf']           = ms = timeit.timeit('sample_daf.sum()', setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"daf.sum()                       {ms:.4f} ms")

    report_daf['sample_daf.daf_sum()', 'daf'] = ms = timeit.timeit('sample_daf.daf_sum()', setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    print(f"sample_daf.daf_sum()            {ms:.4f} ms")

    report_daf['sample_daf.daf_sum2()', 'daf'] = ms = timeit.timeit('sample_daf.daf_sum2()', setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"sample_daf.daf_sum2()           {ms:.4f} ms")

    report_daf['sample_daf.daf_sum3()', 'daf'] = ms = timeit.timeit('sample_daf.daf_sum3()', setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"sample_daf.daf_sum3()           {ms:.4f} ms")

    #report_daf['sum_np', 'loops']          = loops
    report_daf['sum_np', 'daf']             = ms = timeit.timeit('sample_daf.sum_np()',    setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"daf.sum_np()                    {ms:.4f} ms")

    report_daf['sum cols', 'numpy']         = ms = timeit.timeit('hdnpa_dotsum_cols(hdnpa)',  setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"hdnpa_dotsum_cols()             {ms:.4f} ms")

    report_daf['sum cols', 'sqlite']        = ms = timeit.timeit('daf_sql.sum_columns_in_sqlite_table(table_name=datatable1)', setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"sqlite_sum_cols()               {ms:.4f} ms")

    report_daf['sum cols', 'lod']           = ms = timeit.timeit('lod_sum_cols(sample_lod)',  setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"lod_sum_cols()                  {ms:.4f} ms")

    #report_daf['transpose', 'loops']        = loops
    report_daf['transpose', 'pandas']       = ms = timeit.timeit('df.transpose()',            setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"df.transpose()                  {ms:.4f} ms")

    report_daf['transpose', 'daf']          = ms = timeit.timeit('sample_daf.transpose()',          setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"daf.transpose()                 {ms:.4f} ms")

    report_daf['transpose', 'numpy']        = ms = timeit.timeit('np.transpose(hdnpa[1])',    setup=setup_code, globals=globals(), number=loops) * 1000 / (loops)
    gc.enable()
    print(f"daf.transpose()                 {ms:.4f} ms")

    print("\nTime for lookups:")

    #report_daf['keyed lookup', 'loops']    = 
    keyed_lookup_loops = loops*10
    report_daf['keyed lookup', 'daf']       = ms = timeit.timeit("sample_kdaf.select_record('500')", setup=setup_code, globals=globals(), number=keyed_lookup_loops) * 1000 / (keyed_lookup_loops)
    gc.enable()
    print(f"kdaf row lookup                 {ms:.4f} ms")

    report_daf['keyed lookup', 'pandas']    = ms = timeit.timeit("kdf.loc['500'].to_dict()",      setup=setup_code, globals=globals(), number=keyed_lookup_loops) * 1000 / (keyed_lookup_loops)
    gc.enable()
    print(f"kdf row lookup (indexed)        {ms:.4f} ms")

    report_daf['keyed lookup', 'lod']       = ms = timeit.timeit('klod_row_lookup(sample_klod)',  setup=setup_code, globals=globals(), number=keyed_lookup_loops) * 1000 / (keyed_lookup_loops)
    gc.enable()
    print(f"klod_row_lookup()               {ms:.4f} ms")

    report_daf['keyed lookup', 'sqlite']    = ms = timeit.timeit('daf_sql.sqlite_selectrow(table_name=datatable1)', setup=setup_code, globals=globals(), number=keyed_lookup_loops) * 1000 / (keyed_lookup_loops)
    gc.enable()
    print(f"sqlite_row_lookup()             {ms:.4f} ms")

    MB = 1024 * 1024

    report_daf.append({'Attribute': '=====', 
                        'daf':      '=====', 
                        'pandas':   '=====', 
                        'numpy':    '=====', 
                        'sqlite':   '=====', 
                        'lod':      '=====',
                        })

    report_daf.append({'Attribute': 'Size of 1000x1000 array (MB)', 
                        'daf':      sizeof_di['daf']    / MB, 
                        'pandas':   sizeof_di['df']     / MB, 
                        'numpy':    sizeof_di['hdnpa']  / MB, 
                        'sqlite':   sizeof_di['sqlite'] / MB, 
                        'lod':      sizeof_di['lod']    / MB,
                        })

    report_daf.append({'Attribute': 'Size of keyed 1000x1000 array (MB)', 
                        'daf':      sizeof_di['kdaf']   / MB, 
                        'pandas':   sizeof_di['kdf']    / MB, 
                        'numpy':    '--' , 
                        'sqlite':   sizeof_di['sqlite'] / MB, 
                        'lod':      sizeof_di['klod']   / MB,
                        })

    md_report += f"\n### Summary of times (ms) and Sizes (MB)\n\n{report_daf.to_md(smart_fmt = True, just = '>^^^^^')}\n\n"
    
    #md_report += f"\n### Raw data for purposes of obtaining a plot using AI:\n\n{report_daf.to_json()}\n\n"
    
    """
    
Notes:

1. `to_pandas_df()` -- this is a critical operation where Pandas has a severe problem, as it takes about 34x
    longer to load an array vs. Daffodil directly. Since Pandas is very slow appending rows, it is a common pattern to
    first build a table to list of dictionaries (lod), and then port to a pandas df. But
    the overhead is so severe that it will take at least 30 column operations across all columns to make
    up the difference, and so it is commonly better to avoid Pandas altogether, particular in repetitive operations.<br>
2. `to_pandas_df_thru_csv()` -- This exports the array to csv in a buffer, then reads the buffer into Pandas,
    and can improve the porting to pandas df by nearly 10x.<br>
3. `sum_cols()` uses best python summing of all columns with one pass through the data, while `sum_np` first
    exports the columns to NumPy, performs the sum of all columns there, and then reads it back to Daf. In this case,
    it takes about 1/3 the time to use NumPy for summing. This demonstrates that using NumPy for 
    strictly numeric operations on columns may be a good idea if the number of columns and rows being summed is
    sufficient. Otherwise, using Daffodil to sum the columns may still be substantially faster.<br>
    Note: daf_sum2() has an interesting simple change in the code and it results in over 10x increase in time (under study!)<br>
4. Transpose: Numpy performs a transposition without creating a separate copy of the array in memory. Instead, it returns a 
    view of the original array with the dimensions rearranged. It is a constant-time operation, as it simply 
    involves changing the shape and strides of the array metadata without moving any of the actual data in 
    memory. It is extremely efficient and does not consume additional memory. There may be a way to 
    accelerate transposition within Daffodil using python and of non-numeric objects by using a similar strategy
    with the references to objects that Python uses. Transpose with Daffodil is slow right now but there may be
    a way to more quickly provide the transpose operation if coded at a lower level. If columns are selected or dropped,
    a transposition (set flip=True) is 'free' because any column manipulation in Daffodil is relatively difficult.<br>
5. In general, we note that Daffodil is faster than Pandas for array manipulation (inserting rows (94x faster) 
    and cols (20% faster)), appending rows (which Pandas essentially outlaws), 
    and performing keyed lookups (10x faster). Daffodil arrays are smaller 
    whenever any strings are included in the array to about half that of Pandas, as a Pandas array will explode by 5x when a 
    single string column is added for indexing.  While Pandas and Numpy
    are far faster for columnar calculations, Numpy is always faster on all numeric data. Therefore
    it is a good strategy to use Daffodil for all operations except for pure data manipulation, and then
    port the appropriate columns to NumPy. <br>
6. The stragegy of Daffodil compared with Pandas is that Daffodil leaves data in Python native form, which requires no 
    conversion, and is smaller if there are any strings involved. This is a benefit for all cases except when rapid 
    and repeated numeric processing is required, and then the user should export only those
    columns to Numpy. In contrast, Pandas converts all columns to Numpy, and then has to repair the columns that
    have strings or other types. Daffodil is not a direct replacement for Pandas which is still going to be a good choice
    when data already exists and is not being built by any Python code, and there is no need to change world view.
7. Programmers should avoid unnecessary transitions between Python and Pandas if time is of concern.

### Summary of times (ms) and Sizes (MB)

|             Attribute              |  daf  | pandas | numpy  | sqlite |  lod  |
| ---------------------------------: | :---: | :----: | :----: | :----: | :---: |
|                           from_lod |  110  | 4,587  |  63.4  |  661   |       |
|                       to_pandas_df | 4,780 |        | 0.026  |        | 4,587 |
|              to_pandas_df_thru_csv |  227  |        |        |        |       |
|                     from_pandas_df | 11.5  |        | 0.0023 |        |       |
|                           to_numpy | 46.6  | 0.0023 |        |        | 63.4  |
|                         from_numpy |  5.9  | 0.026  |        |        |       |
|                     increment cell | 0.091 | 0.038  |        |        |       |
|                        insert_irow | 0.076 |  6.6   |        |        |       |
|                        insert_icol |  1.2  |  1.6   |        |        |       |
|                           sum cols |  170  |  3.9   |  2.5   | 2,311  |  138  |
|               sample_daf.daf_sum() |  260  |        |        |        |       |
|              sample_daf.daf_sum2() | 2,716 |        |        |        |       |
|              sample_daf.daf_sum3() |  279  |        |        |        |       |
|                             sum_np | 59.0  |        |        |        |       |
|                          transpose | 1,851 |  0.15  | 0.0024 |        |       |
|                       keyed lookup | 0.076 |  0.29  |        |  5.5   | 0.075 |
|                              ===== | ===== | =====  | =====  | =====  | ===== |
|       Size of 1000x1000 array (MB) | 34.4  |  13.4  |  3.9   |  4.9   | 51.5  |
| Size of keyed 1000x1000 array (MB) | 34.6  |  66.7  |   --   |  4.9   | 51.6  |


