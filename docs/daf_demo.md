# Daffodil Demo

 Daffodil is a simple and flexible dataframe package for use with Python.
This page will demonstrate the functionality of Daffodil by showing actual code and results of 
running that code. Daffodil is a good match for many common algorithms in data pipelines and other conversion use cases.
    
For more information about Daffodil, see [https://github.com/raylutz/Daffodil/blob/main/README.md]()

This page is the result of using simple "notebook" functionality (md_demo.py)
which will create a markdown "notebook" report by printing a code block and then run and capture the result. The report can
be viewed directly or converted to HTML for use on a static website.



Code segment with label 'Create a new empty table' not found. Remove parenthesis in label and install null call at the end.

The empty my_daf:

\[0 rows x 0 cols; keyfield=; 0 keys ] (Daf)

bool(my_daf)=False


Code segment with label 'Append some rows to the daf object' not found. Remove parenthesis in label and install null call at the end.


The appended my_daf:
| A | B | C  |
| -: | -: | -: |
| 1 | 2 |  3 |
| 5 | 6 |  7 |
| 8 | 9 | 10 |

\[3 rows x 3 cols; keyfield=; 0 keys ] (Daf)



Code segment with label 'Subset rows, cols using row and cols using indices:' not found. Remove parenthesis in label and install null call at the end.


First 2 rows (and all columns) by slicing:
my_daf[0:2] = 
| A | B | C |
| -: | -: | -: |
| 1 | 2 | 3 |
| 5 | 6 | 7 |

\[2 rows x 3 cols; keyfield=; 0 keys ] (Daf)

Row 0 and 2 (and all columns) using a list:
my_daf[[0,2]] = 
| A | B | C  |
| -: | -: | -: |
| 1 | 2 |  3 |
| 8 | 9 | 10 |

\[2 rows x 3 cols; keyfield=; 0 keys ] (Daf)

Just row 1:
my_daf[1] = 
my_daf[1] = 
| A | B | C |
| -: | -: | -: |
| 5 | 6 | 7 |

\[1 rows x 3 cols; keyfield=; 0 keys ] (Daf)

First 2 cols by slicing:
my_daf[:, 0:2] = 
| A | B |
| -: | -: |
| 1 | 2 |
| 5 | 6 |
| 8 | 9 |

\[3 rows x 2 cols; keyfield=; 0 keys ] (Daf)

Columns 0 and 2 using a list:
my_daf[:, [0,2]] = 
| A | C  |
| -: | -: |
| 1 |  3 |
| 5 |  7 |
| 8 | 10 |

\[3 rows x 2 cols; keyfield=; 0 keys ] (Daf)

Just col 1:
my_daf[:, 1] = 
| B |
| -: |
| 2 |
| 6 |
| 9 |

\[3 rows x 1 cols; keyfield=; 0 keys ] (Daf)



Code segment with label 'Read and write individual cells by row,col indices' not found. Remove parenthesis in label and install null call at the end.


The modified my_daf:
| A | B  | C  |
| -: | -: | -: |
| 1 |  2 |  3 |
| 5 |  6 |  7 |
| 8 | 50 | 10 |

\[3 rows x 3 cols; keyfield=; 0 keys ] (Daf)



Code segment with label 'Read columns and rows' not found. Remove parenthesis in label and install null call at the end.


- col_2=[3, 7, 10]
- row_1=[5, 6, 7]
- col_B=[2, 6, 50]


Code segment with label 'Read rows and columns using methods' not found. Remove parenthesis in label and install null call at the end.


- col_2=[3, 7, 10]
- row_1={'A': 5, 'B': 6, 'C': 7}
- col_B=[2, 6, 50]


Code segment with label 'Insert a new column "Category" on left, and make it the keyfield' not found. Remove parenthesis in label and install null call at the end.


my_daf:
| Category | A | B  | C  |
| -------: | -: | -: | -: |
|    house | 1 |  2 |  3 |
|      car | 5 |  6 |  7 |
|     boat | 8 | 50 | 10 |

\[3 rows x 4 cols; keyfield=Category; 3 keys ] (Daf)



Code segment with label 'Select a record by the key:' not found. Remove parenthesis in label and install null call at the end.


Result:

- da={'Category': 'car', 'A': 5, 'B': 6, 'C': 7}


Code segment with label 'Append more records from a lod' not found. Remove parenthesis in label and install null call at the end.


The appended my_daf:
| Category | A  | B  | C  |
| -------: | -: | -: | -: |
|    house |  1 |  2 |  3 |
|      car |  5 |  6 |  7 |
|     boat |  8 | 50 | 10 |
|     mall | 11 | 12 | 13 |
|      van | 14 | 15 | 16 |
|    condo | 17 | 18 | 19 |

\[6 rows x 4 cols; keyfield=Category; 6 keys ] (Daf)



Code segment with label 'Update records' not found. Remove parenthesis in label and install null call at the end.


The updated my_daf:
| Category | A  | B  | C  |
| -------: | -: | -: | -: |
|    house | 31 | 32 | 33 |
|      car | 25 | 26 | 27 |
|     boat |  8 | 50 | 10 |
|     mall | 11 | 12 | 13 |
|      van | 14 | 15 | 16 |
|    condo | 17 | 18 | 19 |

\[6 rows x 4 cols; keyfield=Category; 6 keys ] (Daf)



Code segment with label 'Add a column "is_vehicle"' not found. Remove parenthesis in label and install null call at the end.


The updated my_daf:
| Category | is_vehicle | A  | B  | C  |
| -------: | ---------: | -: | -: | -: |
|    house |          0 | 31 | 32 | 33 |
|      car |          1 | 25 | 26 | 27 |
|     boat |          1 |  8 | 50 | 10 |
|     mall |          0 | 11 | 12 | 13 |
|      van |          1 | 14 | 15 | 16 |
|    condo |          0 | 17 | 18 | 19 |

\[6 rows x 5 cols; keyfield=Category; 6 keys ] (Daf)



Code segment with label 'daf bool' not found. Remove parenthesis in label and install null call at the end.


- bool(my_daf)=True
- bool(Daf(lol=[]))=False
- bool(Daf(lol=[[]]))=False
- bool(Daf(lol=[[0]]))=True
- bool(Daf(lol=[['']]))=True
- bool(Daf(lol=[[False]]))=True



Code segment with label 'daf attributes' not found. Remove parenthesis in label and install null call at the end.


- len(my_daf)=6
- my_daf.len()=6
- my_daf.shape()=(6, 5)
- my_daf.columns()=['Category', 'is_vehicle', 'A', 'B', 'C']
- my_daf.keys()=['house', 'car', 'boat', 'mall', 'van', 'condo']


Code segment with label 'get_existing_keys' not found. Remove parenthesis in label and install null call at the end.


- existing_keys_ls=['house', 'boat']


Code segment with label 'select_records_daf' not found. Remove parenthesis in label and install null call at the end.


wheels_daf:
| Category | is_vehicle | A  | B  | C  |
| -------: | ---------: | -: | -: | -: |
|      van |          1 | 14 | 15 | 16 |
|      car |          1 | 25 | 26 | 27 |

\[2 rows x 5 cols; keyfield=Category; 2 keys ] (Daf)



Code segment with label 'select_by_dict' not found. Remove parenthesis in label and install null call at the end.


vehicles_daf:
| Category | is_vehicle | A  | B  | C  |
| -------: | ---------: | -: | -: | -: |
|      car |          1 | 25 | 26 | 27 |
|     boat |          1 |  8 | 50 | 10 |
|      van |          1 | 14 | 15 | 16 |

\[3 rows x 5 cols; keyfield=Category; 3 keys ] (Daf)

buildings_daf:
| Category | is_vehicle | A  | B  | C  |
| -------: | ---------: | -: | -: | -: |
|    house |          0 | 31 | 32 | 33 |
|     mall |          0 | 11 | 12 | 13 |
|    condo |          0 | 17 | 18 | 19 |

\[3 rows x 5 cols; keyfield=Category; 3 keys ] (Daf)



Code segment with label 'use `select_where` to select rows where column 'C' is over 20' not found. Remove parenthesis in label and install null call at the end.


high_c_daf:
| Category | is_vehicle | A  | B  | C  |
| -------: | ---------: | -: | -: | -: |
|    house |          0 | 31 | 32 | 33 |
|      car |          1 | 25 | 26 | 27 |

\[2 rows x 5 cols; keyfield=Category; 2 keys ] (Daf)



Code segment with label 'convert to pandas DataFrame' not found. Remove parenthesis in label and install null call at the end.



Converted DataFrame:
```
  Category  is_vehicle   A   B   C
0    house           0  31  32  33
1      car           1  25  26  27
2     boat           1   8  50  10
3     mall           0  11  12  13
4      van           1  14  15  16
5    condo           0  17  18  19
```


Code segment with label 'Add index column 'idx' to the dataframe at the left, starting at 0.' not found. Remove parenthesis in label and install null call at the end.



Modified daf:
| idx | Category | is_vehicle | A  | B  | C  |
| --: | -------: | ---------: | -: | -: | -: |
|   0 |    house |          0 | 31 | 32 | 33 |
|   1 |      car |          1 | 25 | 26 | 27 |
|   2 |     boat |          1 |  8 | 50 | 10 |
|   3 |     mall |          0 | 11 | 12 | 13 |
|   4 |      van |          1 | 14 | 15 | 16 |
|   5 |    condo |          0 | 17 | 18 | 19 |

\[6 rows x 6 cols; keyfield=Category; 6 keys ] (Daf)




Code segment with label 'Create a table of file information' not found. Remove parenthesis in label and install null call at the end.



Contents of C:\Windows\System32:
|                            filepath                             |  size  | modified_timestamp  | is_dir |
| :-------------------------------------------------------------- | -----: | :------------------ | :----- |
| \07409496-a423-4a3e-b620-2cfb01a9318d_HyperV-ComputeNetwork.dll |  12304 | 2019-12-07T03:19:14 | False  |
| \1028\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1028\vsjitdebuggerui.dll                                       |  18848 | 2022-06-01T02:06:34 | False  |
| \1029\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1029\vsjitdebuggerui.dll                                       |  23456 | 2022-06-01T02:06:34 | False  |
| \1031\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1031\vsjitdebuggerui.dll                                       |  24976 | 2022-06-01T02:06:34 | False  |
| \1033\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1033\vsjitdebuggerui.dll                                       |  23456 | 2022-06-01T02:05:54 | False  |
| \1036\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1036\vsjitdebuggerui.dll                                       |  24480 | 2022-06-01T02:05:56 | False  |
| \1040\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1040\vsjitdebuggerui.dll                                       |  23968 | 2022-06-01T02:05:36 | False  |
| \1041\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1041\vsjitdebuggerui.dll                                       |  20368 | 2022-06-01T02:06:34 | False  |
| ...                                                             |    ... | ...                 | ...    |
| \zh-TW\comctl32.dll.mui                                         |   5120 | 2019-12-07T01:09:02 | False  |
| \zh-TW\comdlg32.dll.mui                                         |  45056 | 2023-12-13T18:14:08 | False  |
| \zh-TW\fms.dll.mui                                              |   9216 | 2023-12-13T18:14:04 | False  |
| \zh-TW\mlang.dll.mui                                            |  12288 | 2019-12-07T01:09:04 | False  |
| \zh-TW\msimsg.dll.mui                                           |  37376 | 2023-12-13T18:13:59 | False  |
| \zh-TW\msprivs.dll.mui                                          |   3584 | 2019-12-07T01:09:04 | False  |
| \zh-TW\quickassist.exe.mui                                      |   3072 | 2023-12-13T18:17:15 | False  |
| \zh-TW\SyncRes.dll.mui                                          |  13824 | 2019-12-06T08:58:00 | False  |
| \zh-TW\Windows.Management.SecureAssessment.Diagnostics.dll.mui  |   3584 | 2019-12-07T01:10:17 | False  |
| \zh-TW\Windows.Media.Speech.UXRes.dll.mui                       |   6656 | 2023-12-13T18:14:04 | False  |
| \zh-TW\windows.ui.xaml.dll.mui                                  |  11776 | 2023-12-13T18:14:00 | False  |
| \zh-TW\WWAHost.exe.mui                                          |  11264 | 2023-12-13T18:14:04 | False  |
| \zipcontainer.dll                                               |  79872 | 2019-12-07T01:08:33 | False  |
| \zipfldr.dll                                                    | 285696 | 2024-04-24T05:28:01 | False  |
| \ztrace_maps.dll                                                |  30720 | 2019-12-07T01:08:28 | False  |

\[8802 rows x 4 cols; keyfield=; 0 keys ] (Daf)



Contents of C:\Windows\System32 (in raw text format):
```
|                            filepath                             |  size  | modified_timestamp  | is_dir |
| :-------------------------------------------------------------- | -----: | :------------------ | :----- |
| \07409496-a423-4a3e-b620-2cfb01a9318d_HyperV-ComputeNetwork.dll |  12304 | 2019-12-07T03:19:14 | False  |
| \1028\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1028\vsjitdebuggerui.dll                                       |  18848 | 2022-06-01T02:06:34 | False  |
| \1029\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1029\vsjitdebuggerui.dll                                       |  23456 | 2022-06-01T02:06:34 | False  |
| \1031\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1031\vsjitdebuggerui.dll                                       |  24976 | 2022-06-01T02:06:34 | False  |
| \1033\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1033\vsjitdebuggerui.dll                                       |  23456 | 2022-06-01T02:05:54 | False  |
| \1036\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1036\vsjitdebuggerui.dll                                       |  24480 | 2022-06-01T02:05:56 | False  |
| \1040\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1040\vsjitdebuggerui.dll                                       |  23968 | 2022-06-01T02:05:36 | False  |
| \1041\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1041\vsjitdebuggerui.dll                                       |  20368 | 2022-06-01T02:06:34 | False  |
| ...                                                             |    ... | ...                 | ...    |
| \zh-TW\comctl32.dll.mui                                         |   5120 | 2019-12-07T01:09:02 | False  |
| \zh-TW\comdlg32.dll.mui                                         |  45056 | 2023-12-13T18:14:08 | False  |
| \zh-TW\fms.dll.mui                                              |   9216 | 2023-12-13T18:14:04 | False  |
| \zh-TW\mlang.dll.mui                                            |  12288 | 2019-12-07T01:09:04 | False  |
| \zh-TW\msimsg.dll.mui                                           |  37376 | 2023-12-13T18:13:59 | False  |
| \zh-TW\msprivs.dll.mui                                          |   3584 | 2019-12-07T01:09:04 | False  |
| \zh-TW\quickassist.exe.mui                                      |   3072 | 2023-12-13T18:17:15 | False  |
| \zh-TW\SyncRes.dll.mui                                          |  13824 | 2019-12-06T08:58:00 | False  |
| \zh-TW\Windows.Management.SecureAssessment.Diagnostics.dll.mui  |   3584 | 2019-12-07T01:10:17 | False  |
| \zh-TW\Windows.Media.Speech.UXRes.dll.mui                       |   6656 | 2023-12-13T18:14:04 | False  |
| \zh-TW\windows.ui.xaml.dll.mui                                  |  11776 | 2023-12-13T18:14:00 | False  |
| \zh-TW\WWAHost.exe.mui                                          |  11264 | 2023-12-13T18:14:04 | False  |
| \zipcontainer.dll                                               |  79872 | 2019-12-07T01:08:33 | False  |
| \zipfldr.dll                                                    | 285696 | 2024-04-24T05:28:01 | False  |
| \ztrace_maps.dll                                                |  30720 | 2019-12-07T01:08:28 | False  |

\[8802 rows x 4 cols; keyfield=; 0 keys ] (Daf)

```

- daf size in memory: 2,404,880 bytes
- pandas df size in memory: 2,875,088 bytes


Code segment with label 'Limit this list to just the files' not found. Remove parenthesis in label and install null call at the end.



Files only in C:\Windows\System32:
|                            filepath                             |  size  | modified_timestamp  | is_dir |
| :-------------------------------------------------------------- | -----: | :------------------ | :----- |
| \07409496-a423-4a3e-b620-2cfb01a9318d_HyperV-ComputeNetwork.dll |  12304 | 2019-12-07T03:19:14 | False  |
| \1028\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1028\vsjitdebuggerui.dll                                       |  18848 | 2022-06-01T02:06:34 | False  |
| \1029\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1029\vsjitdebuggerui.dll                                       |  23456 | 2022-06-01T02:06:34 | False  |
| \1031\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1031\vsjitdebuggerui.dll                                       |  24976 | 2022-06-01T02:06:34 | False  |
| \1033\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1033\vsjitdebuggerui.dll                                       |  23456 | 2022-06-01T02:05:54 | False  |
| \1036\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1036\vsjitdebuggerui.dll                                       |  24480 | 2022-06-01T02:05:56 | False  |
| \1040\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1040\vsjitdebuggerui.dll                                       |  23968 | 2022-06-01T02:05:36 | False  |
| \1041\VsGraphicsResources.dll                                   |  77216 | 2022-10-06T17:05:12 | False  |
| \1041\vsjitdebuggerui.dll                                       |  20368 | 2022-06-01T02:06:34 | False  |
| ...                                                             |    ... | ...                 | ...    |
| \zh-TW\comctl32.dll.mui                                         |   5120 | 2019-12-07T01:09:02 | False  |
| \zh-TW\comdlg32.dll.mui                                         |  45056 | 2023-12-13T18:14:08 | False  |
| \zh-TW\fms.dll.mui                                              |   9216 | 2023-12-13T18:14:04 | False  |
| \zh-TW\mlang.dll.mui                                            |  12288 | 2019-12-07T01:09:04 | False  |
| \zh-TW\msimsg.dll.mui                                           |  37376 | 2023-12-13T18:13:59 | False  |
| \zh-TW\msprivs.dll.mui                                          |   3584 | 2019-12-07T01:09:04 | False  |
| \zh-TW\quickassist.exe.mui                                      |   3072 | 2023-12-13T18:17:15 | False  |
| \zh-TW\SyncRes.dll.mui                                          |  13824 | 2019-12-06T08:58:00 | False  |
| \zh-TW\Windows.Management.SecureAssessment.Diagnostics.dll.mui  |   3584 | 2019-12-07T01:10:17 | False  |
| \zh-TW\Windows.Media.Speech.UXRes.dll.mui                       |   6656 | 2023-12-13T18:14:04 | False  |
| \zh-TW\windows.ui.xaml.dll.mui                                  |  11776 | 2023-12-13T18:14:00 | False  |
| \zh-TW\WWAHost.exe.mui                                          |  11264 | 2023-12-13T18:14:04 | False  |
| \zipcontainer.dll                                               |  79872 | 2019-12-07T01:08:33 | False  |
| \zipfldr.dll                                                    | 285696 | 2024-04-24T05:28:01 | False  |
| \ztrace_maps.dll                                                |  30720 | 2019-12-07T01:08:28 | False  |

\[8679 rows x 4 cols; keyfield=; 0 keys ] (Daf)




Code segment with label 'Demonstration of groupby_cols_reduce' not found. Remove parenthesis in label and install null call at the end.



Original data_table_daf:
| gender | religion | zipcode | cancer | covid19 | gun | auto |
| -----: | -------: | ------: | -----: | ------: | --: | ---: |
|      M |        C |   90001 |      1 |       2 |   3 |    4 |
|      M |        C |   90001 |      5 |       6 |   7 |    8 |
|      M |        C |   90002 |      9 |      10 |  11 |   12 |
|      M |        C |   90002 |     13 |      14 |  15 |   16 |
|      M |        J |   90001 |      1 |       2 |   3 |    4 |
|      M |        J |   90001 |     13 |      14 |  15 |   16 |
|      M |        J |   90002 |      5 |       6 |   7 |    8 |
|      M |        J |   90002 |      9 |      10 |  11 |   12 |
|      M |        I |   90001 |     13 |      14 |  15 |   16 |
|      M |        I |   90001 |      1 |       2 |   3 |    4 |
|      M |        I |   90002 |      4 |       3 |   2 |    1 |
|      M |        I |   90002 |      9 |      10 |  11 |   12 |
|      F |        C |   90001 |      4 |       3 |   2 |    1 |
|      F |        C |   90001 |      5 |       6 |   7 |    8 |
|      F |        C |   90002 |      4 |       3 |   2 |    1 |
|      F |        C |   90002 |     13 |      14 |  15 |   16 |
|      F |        J |   90001 |      4 |       3 |   2 |    1 |
|      F |        J |   90001 |      1 |       2 |   3 |    4 |
|      F |        J |   90002 |      8 |       7 |   6 |    5 |
|      F |        J |   90002 |      1 |       2 |   3 |    4 |
|      F |        I |   90001 |      8 |       7 |   6 |    5 |
|      F |        I |   90001 |      5 |       6 |   7 |    8 |
|      F |        I |   90002 |      8 |       7 |   6 |    5 |
|      F |        I |   90002 |     13 |      14 |  15 |   16 |

\[24 rows x 7 cols; keyfield=; 0 keys ] (Daf)




Code segment with label 'Now reduce the data using groupby_cols_reduce' not found. Remove parenthesis in label and install null call at the end.



Resulting Reduction:
| gender | religion | zipcode | cancer | covid19 | gun | auto |
| -----: | -------: | ------: | -----: | ------: | --: | ---: |
|      M |        C |   90001 |      6 |       8 |  10 |   12 |
|      M |        C |   90002 |     22 |      24 |  26 |   28 |
|      M |        J |   90001 |     14 |      16 |  18 |   20 |
|      M |        J |   90002 |     14 |      16 |  18 |   20 |
|      M |        I |   90001 |     14 |      16 |  18 |   20 |
|      M |        I |   90002 |     13 |      13 |  13 |   13 |
|      F |        C |   90001 |      9 |       9 |   9 |    9 |
|      F |        C |   90002 |     17 |      17 |  17 |   17 |
|      F |        J |   90001 |      5 |       5 |   5 |    5 |
|      F |        J |   90002 |      9 |       9 |   9 |    9 |
|      F |        I |   90001 |     13 |      13 |  13 |   13 |
|      F |        I |   90002 |     21 |      21 |  21 |   21 |

\[12 rows x 7 cols; keyfield=; 0 keys ] (Daf)



Check the result against manually generated:
bool(grouped_and_summed_daf.lol==expected_lol)=True


Code segment with label 'Further group to just zipcodes' not found. Remove parenthesis in label and install null call at the end.



Results for zipcode Reduction:
| zipcode | cancer | covid19 | gun | auto |
| ------: | -----: | ------: | --: | ---: |
|   90001 |     61 |      67 |  73 |   79 |
|   90002 |     96 |     100 | 104 |  108 |

\[2 rows x 5 cols; keyfield=; 0 keys ] (Daf)


