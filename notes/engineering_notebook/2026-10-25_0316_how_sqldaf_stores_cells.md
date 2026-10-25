# How a SqlDaf stores cells: a draft for review

Started 2026-10-25. Ray asked for one description of how cell values go into a SqlDaf and come back,
with exact examples that print each result. This draft is for his review. It is not in the docs yet.
Related: Q293, a dict or list cell written to a SqlDaf, mail to_daffodil/2026-10-24_2348_sqldaf_dict_cells.md.

## The model, as discussed with Ray

- A SqlDaf column is like a Daf just read with `from_csv()`: every cell starts as the text Python's
  csv reader gives. A table kept in a file is converted once, in bulk, with `apply_dtypes()`, on
  the int and float columns it needs. Bools are 0 and 1.
- Loaded from CSV, an empty field is stored as SQL NULL. SQL NULL reads and prints as NULLSTR, ''.
- A dict or list cell is daffodil PYON text, `str()` of a cell whose bools are 0 and 1 and whose
  missing values are '', as `to_csv_file()` writes it. That text is valid JSON5, so SQLite's JSON
  functions read it. On a SqlDaf it stays text. A Python function in `select_where()` or `apply()`,
  or one registered with SQLite, decodes it. After `to_daf()`, the Daf's `apply_dtypes()` decodes it.
- Ray: `apply_dtypes()` with dtype dict on a SqlDaf should not raise. It should give `str(dict)`, the text.

## The example session

Run 2026-10-25 in the sql_backing worktree at 2e39719, by scratch session.py. The worktree also held the
uncommitted Q293 change, which the session does not reach, since it writes no dict from Python. `job_dir` stands for a
temporary folder.

```pycon
>>> SqlDaf.db_path = 'job_dir/job.db'
>>> dtypes = {'id': str, 'n': int, 'x': float, 'ok': bool, 'p': str, 'cv': dict}
>>> daf = Daf(cols=['id', 'n', 'x', 'ok', 'p', 'cv'], keyfield='id', dtypes=dtypes, lol=[['b1', 3, 2.5, 1, 'P1', {'c0': 1, 'c1': ''}], ['b2', '', '', 0, '', {}]])
>>> daf

| id | n  |  x  | ok | p  |         cv          |
| -: | -: | --: | -: | -: | ------------------: |
| b1 |  3 | 2.5 |  1 | P1 | {'c0': 1, 'c1': ''} |
| b2 |    |     |  0 |    |                  {} |

%% daf rows=2; cols=6; keyfield='id'; name=''
>>> daf.to_csv_file('job_dir/t.csv')
'job_dir/t.csv'
>>> print(open('job_dir/t.csv').read())
id,n,x,ok,p,cv
b1,3,2.5,1,P1,"{'c0': 1, 'c1': ''}"
b2,,,0,,{}
>>> sdaf = SqlDaf.from_csv('job_dir/t.csv', name='t', keyfield='id')
>>> sdaf

| id | n  |  x  | ok | p  |         cv          |
| -: | -: | --: | -: | -: | ------------------: |
| b1 |  3 | 2.5 |  1 | P1 | {'c0': 1, 'c1': ''} |
| b2 |    |     |  0 |    |                  {} |

%% daf rows=2; cols=6; keyfield='id'; name='t'
>>> sdaf.sql('SELECT id, typeof(n), typeof(x), typeof(ok), typeof(p), typeof(cv) FROM t')

| id | typeof(n) | typeof(x) | typeof(ok) | typeof(p) | typeof(cv) |
| -: | --------: | --------: | ---------: | --------: | ---------: |
| b1 |      text |      text |       text |      text |       text |
| b2 |      null |      null |       text |      null |       text |

%% daf rows=2; cols=6; keyfield=''; name=''
>>> _ = sdaf.apply_dtypes(dtypes={'n': int, 'x': float, 'ok': bool}, cols=['n', 'x', 'ok'])
>>> sdaf

| id | n  |  x  | ok | p  |         cv          |
| -: | -: | --: | -: | -: | ------------------: |
| b1 |  3 | 2.5 |  1 | P1 | {'c0': 1, 'c1': ''} |
| b2 |    |     |  0 |    |                  {} |

%% daf rows=2; cols=6; keyfield='id'; name='t'
>>> sdaf.sql('SELECT id, typeof(n), typeof(x), typeof(ok), typeof(p), typeof(cv) FROM t')

| id | typeof(n) | typeof(x) | typeof(ok) | typeof(p) | typeof(cv) |
| -: | --------: | --------: | ---------: | --------: | ---------: |
| b1 |   integer |      real |    integer |      text |       text |
| b2 |      null |      null |    integer |      null |       text |

%% daf rows=2; cols=6; keyfield=''; name=''
>>> sdaf.sql('SELECT SUM(n), COUNT(n), AVG(x) FROM t')

| SUM(n) | COUNT(n) | AVG(x) |
| -----: | -------: | -----: |
|      3 |        1 |    2.5 |

%% daf rows=1; cols=3; keyfield=''; name=''
>>> back_daf = sdaf.to_daf()
>>> back_daf.dtypes = dtypes
>>> back_daf.apply_dtypes(cols=['cv'])

| id | n  |  x  | ok | p  |         cv          |
| -: | -: | --: | -: | -: | ------------------: |
| b1 |  3 | 2.5 |  1 | P1 | {'c0': 1, 'c1': ''} |
| b2 |    |     |  0 |    |                  {} |

%% daf rows=2; cols=6; keyfield='id'; name='t'
>>> back_daf['b1', 'cv'].to_value(), type(back_daf['b1', 'cv'].to_value())
({'c0': 1, 'c1': ''}, <class 'dict'>)
>>> back_daf.lol == daf.lol
True
```

## What it shows

- The table prints the same at every step. SQL NULL prints as an empty cell, as NULLSTR does.
- Loaded from CSV, every value is text, and an empty field is already SQL NULL.
- `apply_dtypes()` of n, x and ok stores integers and reals. The empty cells stay SQL NULL. cv stays text.
- Then `SUM(n)` is 3, an int, `COUNT(n)` is 1, and `AVG(x)` is 2.5.
- `to_daf()`, then the Daf's `apply_dtypes()` of cv, gives back the original Daf, with cv a dict.

## Facts behind it, checked 2026-10-25

- SQLite 3.53.1 `json_valid(text, 2)`: daffodil PYON such as `{'x': 2, 'b': 1, 'n': ''}` is valid.
  `str()` of a cell holding True, False or None is not: "malformed JSON".
- Three cells 3, 4 and the text '': `SUM` is 7.0, a float. With SQL NULL in place of '': 7, an int.
  With the text '', `COUNT` counts it, `AVG` divides by it, and `MAX` returns it, since text sorts above numbers.
- SqlDaf's `apply_dtypes()` of an int or float column turns '' into SQL NULL. Of a bool column it gives 0,
  as a Daf does. A str column keeps '' as text.
- 1,000,000 rows, 2 int columns of text, converted by `apply_dtypes()`: 1.44 s.

## Open

- Q293, settled 2026-10-25: the direct write, with a test of each cell for dict or list. Ray first proposed
  writing a Daf to SQL by way of CSV. Measured on 1,000,000 rows, that stored the same, in 6.38 to 7.58 s
  against 2.48 s for the direct write. Ray: "keep the direct write, add the CSV round-trip test". The retry
  when SQLite refuses was not measured, and is not used.
- Whether writes should keep turning NULLSTR into SQL NULL cell by cell.
- Decided, not built: `apply_dtypes()` and `from_csv()` on a SqlDaf take dtype dict and list. They record the
  dtype and convert nothing, since the stored text is the value. Ray: it "should return str(dict)".
