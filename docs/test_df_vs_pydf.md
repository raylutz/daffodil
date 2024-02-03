# Evaluate conversion and calculation time tradeoffs between Pandas, Pydf, Numpy, etc.

## Create sample_lod

Here we generate a table using a python list-of-dict structure,
        with {num_columns} columns and {num_rows} rows, consisting of
        integers from 0 to 100. Set the seed to an arbitrary value for
        reproducibility. Also, create other forms similarly or by converting
        the sample_lod.

```python
    sizeof_dict = {}
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
```




Generated sample_lod with 1000 records
- sizeof_di['lod']=124,968,872 bytes

## Create sample_klod

sample_klod is similar to sample_lod but it has a first column 
        'rowkey' is a string key that can be used to look up a row.

```python
    sample_klod = [dict(zip(['rowkey']+[f'Col{i}' 
                    for i in range(num_columns)], 
                        [str(i)] + list(np.random.randint(1, 100, num_columns)))) 
                            for i in range(num_rows)]

    sizeof_di['klod'] = asizeof.asizeof(sample_klod)
    md_report += pr(f"\n\nGenerated sample_klod with {len(sample_klod)} records\n"
                 f"- {sizeof_di['klod']=:,} bytes\n\n")
```




Generated sample_klod with 1000 records
- sizeof_di['klod']=125,024,928 bytes

## Create pydf from sample_lod



```python
    pydf = Pydf.from_lod(sample_lod)
    sizeof_di['pydf'] = asizeof.asizeof(pydf)
    md_report += pr(f"pydf:\n{pydf}\n\n"
                    f"{sizeof_di['pydf']=:,} bytes\n\n")
```


pydf:
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


sizeof_di['pydf']=40,190,736 bytes

## Create kpydf from sample_klod



```python
    kpydf = Pydf.from_lod(sample_klod, keyfield='rowkey')
    sizeof_di['kpydf'] = asizeof.asizeof(kpydf)
    md_report += pr(f"kpydf:\n{kpydf}\n\n"
                    f"{sizeof_di['kpydf']=:,} bytes\n\n")
```


kpydf:
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

\[1000 rows x 1001 cols; keyfield=rowkey; 1000 keys ] (Pydf)


sizeof_di['kpydf']=40,323,496 bytes

## Create Pandas df



```python
    # Create a Pandas DataFrame
    # here we used an unadorned basic pre-canned method.
    df = pd.DataFrame(sample_lod, dtype=int)
    sizeof_di['df'] = asizeof.asizeof(df)

    md_report += pr(f"\n\n```{df}\n```\n\n"
                    f"{sizeof_di['df']=:,} bytes\n\n")
```




```     Col0  Col1  Col2  Col3  Col4  Col5  Col6  Col7  Col8  Col9  Col10  Col11  ...  Col988  Col989  Col990  Col991  Col992  Col993  Col994  Col995  Col996  Col997  Col998  Col999
0      51    92    14    71    60    20    82    86    74    74     87     99  ...      43      24      16      12      83      24      67       9      66      17      99      85
1      33     7    39    82    41    40     5    51    25    63     97     58  ...      10      44      88      32      40       7      10      85      50      87      40      16
2      75    45    31    78    79    53    85    91    19    32     73     39  ...      55      82      57       3       3      19       9      23      98      25      36      84
3      53    20    73    37    45     3    59    56    44    19     16     70  ...      33      44      60      82      15      65      39      16      33      15      59      65
4      65    89    12    55    30    33    38    66     7    86     77     54  ...      54      45      31      54      95       4      35      48      57      38      79      96
..    ...   ...   ...   ...   ...   ...   ...   ...   ...   ...    ...    ...  ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...
995    47    57    85    63    23    69     1    88    15    50     95     29  ...      71      35      44      65      18      80      70      27      71      55      97      56
996    71    48    29    19    43    52    13     3    34    40      3     38  ...      68      21      18      22      45      39      17      70      76      80      64       8
997    37     4    96    39    82    47    53    83    49    64     72     12  ...      50      59       9      77       4      26      86      21      17      31      32      20
998    23    39    77     9    21    61     2    43    55    59      3     92  ...      13       3      19      71      86      76      16       0      63      22      81      97
999    86     9    27     2    40     3    66    51    94    90     23     29  ...      98      37      60      85       7       9       5      86      34      61      77      52

[1000 rows x 1000 columns]
```

sizeof_di['df']=9,767,776 bytes

## Create Pandas csv_df from Pydf thru csv



```python
    # Create a Pandas DataFrame by convering Pydf through a csv buffer.
    csv_df = pydf.to_pandas_df(use_csv=True)
    sizeof_di['csv_df'] = asizeof.asizeof(csv_df)

    md_report += pr(f"\n\n```{csv_df}\n```\n\n"
                    f"{sizeof_di['csv_df']=:,} bytes\n\n")
```




```     Col0  Col1  Col2  Col3  Col4  Col5  Col6  Col7  Col8  Col9  Col10  Col11  ...  Col988  Col989  Col990  Col991  Col992  Col993  Col994  Col995  Col996  Col997  Col998  Col999
0      51    92    14    71    60    20    82    86    74    74     87     99  ...      43      24      16      12      83      24      67       9      66      17      99      85
1      33     7    39    82    41    40     5    51    25    63     97     58  ...      10      44      88      32      40       7      10      85      50      87      40      16
2      75    45    31    78    79    53    85    91    19    32     73     39  ...      55      82      57       3       3      19       9      23      98      25      36      84
3      53    20    73    37    45     3    59    56    44    19     16     70  ...      33      44      60      82      15      65      39      16      33      15      59      65
4      65    89    12    55    30    33    38    66     7    86     77     54  ...      54      45      31      54      95       4      35      48      57      38      79      96
..    ...   ...   ...   ...   ...   ...   ...   ...   ...   ...    ...    ...  ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...
995    47    57    85    63    23    69     1    88    15    50     95     29  ...      71      35      44      65      18      80      70      27      71      55      97      56
996    71    48    29    19    43    52    13     3    34    40      3     38  ...      68      21      18      22      45      39      17      70      76      80      64       8
997    37     4    96    39    82    47    53    83    49    64     72     12  ...      50      59       9      77       4      26      86      21      17      31      32      20
998    23    39    77     9    21    61     2    43    55    59      3     92  ...      13       3      19      71      86      76      16       0      63      22      81      97
999    86     9    27     2    40     3    66    51    94    90     23     29  ...      98      37      60      85       7       9       5      86      34      61      77      52

[1000 rows x 1000 columns]
```

sizeof_di['csv_df']=17,983,776 bytes

## Create keyed Pandas df

Create a keyed Pandas df based on the sample_klod generated.
        Please note this takes far more memory than a Pandas df without this
        str column, almost 3x the size of Pydf instance. To test fast lookups,
        we also use set_index to get ready for fast lookups.

```python
    kdf = pd.DataFrame(sample_klod)
    
    # also set the rowkey as the index for fast lookups.
    kdf.set_index('rowkey', inplace=True)
    sizeof_di['kdf'] = asizeof.asizeof(kdf)
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

- sizeof_di['kdf']=102,910,896 bytes

## create hdnpa from lod



```python
    hdnpa = lod_to_hdnpa(sample_lod)
    sizeof_di['hdnpa'] = asizeof.asizeof(hdnpa)
    md_report += pr(f"{sizeof_di['hdnpa']=:,} bytes\n\n")
```


sizeof_di['hdnpa']=4,125,152 bytes

## Create lont from lod



```python
    lont = lod_to_lont(sample_lod)
    sizeof_di['lont'] = asizeof.asizeof(lont)
    md_report += pr(f"{sizeof_di['lont']=:,} bytes\n\n")
```


sizeof_di['lont']=40,048,872 bytes

## Create hdlot from lod



```python
    hdlot = lod_to_hdlot(sample_lod)
    sizeof_di['hdlot'] = asizeof.asizeof(hdlot)
    md_report += pr(f"{sizeof_di['hdlot']=:,} bytes\n\n")

    # create sqlite_table
```


sizeof_di['hdlot']=40,173,880 bytes

## Create sqlite_table from klod



```python
    lod_to_sqlite_table(sample_klod, table_name='tempdata1')

    datatable1 = 'tempdata1'
    datatable2 = 'tempdata2'
    sizeof_di['sqlite'] = os.path.getsize(datatable1+'.db')
    md_report += pr(f"{sizeof_di['sqlite']=:,} bytes\n\n")
```


sizeof_di['sqlite']=5,148,672 bytes

## Create table of estimated memory usage for all types

use Pydf.from_lod_to_cols to create a table with first colunm key names, and second column values.

```python
    all_sizes_pydf = Pydf.from_lod_to_cols([sizeof_di], cols=['Datatype' 'Size in Memory (bytes)'])
    md_report += all_sizes_pydf.to_md(smart_fmt=True)
```


| DatatypeSize in Memory (bytes) |             |
| -----------------------------: | :---------- |
|                            lod | 124,968,872 |
|                           klod | 125,024,928 |
|                           pydf | 40,190,736  |
|                          kpydf | 40,323,496  |
|                             df | 9,767,776   |
|                         csv_df | 17,983,776  |
|                            kdf | 102,910,896 |
|                          hdnpa | 4,125,152   |
|                           lont | 40,048,872  |
|                          hdlot | 40,173,880  |
|                         sqlite | 5,148,672   |
## Time conversions and operations

This secion uses the timeit() function to time conversions.
        For each convertion, the time wil be added to the (datatype)_times dicts.

```python
    setup_code = f'''
import pandas as pd
import numpy as np
from collections import namedtuple
import sys
sys.path.append('..')
from Pydf.Pydf import Pydf, T_pydf

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
```


|             Attribute              |  Pydf  |  Pandas  |  Numpy   | Sqlite |  lod   | (loops) |
| ---------------------------------: | :----: | :------: | :------: | :----: | :----: | :------ |
|                           from_lod |  1.4   |   45.5   |   0.61   |  6.8   |        | 10      |
|                       to_pandas_df |  47.4  |          | 0.00029  |        |  45.5  | 10      |
|              to_pandas_df_thru_csv |  5.0   |          |          |        |        | 10      |
|                     from_pandas_df |  3.9   |          | 0.000043 |        |        | 10      |
|                           to_numpy |  0.45  | 0.000043 |          |        |  0.61  | 10      |
|                         from_numpy | 0.077  | 0.00029  |          |        |        | 10      |
|                     increment cell | 0.011  |  0.039   |          |        |        | 1,000   |
|                        insert_irow | 0.0017 |   0.80   |          |        |        | 100     |
|                        insert_icol |  0.11  |   0.18   |          |        |        | 100     |
|                           sum cols |  1.5   |  0.042   |  0.022   |  2.6   |  1.3   | 10      |
|                          transpose |  19.5  |  0.0021  |          |        |        | 10      |
|                       keyed lookup | 0.0077 |  0.054   |          |  0.34  | 0.0067 | 100     |
|       Size of 1000x1000 array (MB) |  38.3  |   9.3    |   3.9    |  4.9   |  119   |         |
| Size of keyed 1000x1000 array (MB) |  38.5  |   98.1   |          |        |  119   |         |
