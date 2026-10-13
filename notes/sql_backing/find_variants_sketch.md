# find_variants on SqlDaf: a sketch

Written on 2026-10-13 for Ray, as part of question 96. Signatures only. Nothing here is built.
It follows decisions 88 to 95 in design_draft.md. The source read is locate_variants() in
AuditEngine's utilities/sql_variants.py, lines 25 to 1440, at claude_dev 283d992f0.

## What locate_variants does now

For two inputs, audit votes and CVR votes, with an optional third, the modified CVR. Each step
changes tables in place.

| Step | Now                                                                        |
| ---: | :------------------------------------------------------------------------- |
|    2 | Create empty variants tables, with a unique index on ballot_id.            |
|    3 | residue[i] = rows of input[i] whose key is not in the other input. Then DELETE those rows from input[i]. Residue is exported to CSV. |
|    4 | INSERT OR IGNORE into variants[i] the rows of input[i] with a flag column = 1, or, for concise votes, a REGEXP match on concise_votes. |
|    5 | INSERT OR IGNORE into variants[i] the rows of input[i] whose key is in the other variants table. |
|    6 | With three inputs: INSERT OR IGNORE into variants[i] the rows whose key is in the modified CVR. |
|    7 | nonvariants[i] = rows of input[i] whose key is not in variants[i].         |
|    8 | disagreed[i] = nonvariants[0] JOIN nonvariants[1] on key, where a compared column differs, or concise_votes differ. |
|    9 | Concise only: keys whose normalized concise_votes agree go to a TEMP table, and are DELETEd from both disagreed tables. |
|   10 | INSERT OR IGNORE into variants[i] the rows of nonvariants[i] whose key is in disagreed[i]. |
|   11 | DELETE from nonvariants[i] the rows whose key is in variants[i].           |
|  12+ | Contest-level tables: json_each over a registered extract_contests() on concise_votes, then compares and joins. |

## The structure change

Views cannot be changed. So the result is described as sets of keys, and the tables come
from them. The steps above reduce to this:

    common keys     = keys in both inputs
    in[i]           = input[i] with a common key            (replaces the DELETE of step 3)
    residue[i]      = input[i] with a key not in the other  (step 3)
    flagged keys    = keys of in[0] flagged, plus keys of in[1] flagged   (steps 4 and 5)
    modified keys   = keys of the modified CVR                            (step 6)
    rest[i]         = in[i] without flagged or modified keys
    disagreed keys  = keys where rest[0] and rest[1] differ, less those that agree
                      after normalizing                                   (steps 8 and 9)
    variant keys    = flagged + modified + disagreed                      (steps 10 and 11)
    variants[i]     = in[i] with a variant key
    nonvariants[i]  = in[i] without a variant key

Step 5, harmonizing, disappears: a union of the two flagged sets already has every key from
either side. Steps 10 and 11 disappear the same way.

Things to check against the current code before trusting this:

- Row order. INSERT OR IGNORE builds variants[i] in the order of the steps: flagged rows
  first, then harmonized, then modified, then disagreed. A view keeps input order, by decision
  93. Does anything downstream depend on the old order?
- Duplicate keys. The unique index plus INSERT OR IGNORE keeps the first row of a duplicated
  ballot_id in variants and silently drops the rest. nonvariants keeps every copy. A view
  keeps every copy on both sides. Can an input have a duplicate ballot_id?
- Step 3 deletes rows from the input tables themselves, so later stages see the trimmed
  inputs. A view leaves the inputs whole. Does a later stage read the input tables?

## Signatures

### SqlStore

```python
class SqlStore:
    def __init__(
            self,
            path:       str = '',                       # database file; '' makes a temp file
            conn:       sqlite3.Connection | None = None,   # wrap an existing connection instead
            ) -> None: ...

    def table(self, name: str, keyfield: str = '') -> 'SqlDaf': ...          # an existing table
    def from_csv(self, path: str, name: str = '', keyfield: str = '',
                 dtypes: T_dtype_dict | None = None) -> 'SqlDaf': ...       # load without building in memory
    def from_daf(self, data_daf: Daf, name: str) -> 'SqlDaf': ...           # write an in-memory Daf
    def close(self) -> None: ...                       # TEMP objects and a temp file go
    def __enter__(self) -> 'SqlStore': ...
    def __exit__(self, *exc_info: Any) -> None: ...
```

### SqlDaf

Same names as Daf where the meaning is the same.

```python
class SqlDaf(Daf):
    name:       str                                 # the SQL name, read only (decision 92)
    keyfield:   str

    # reading
    def __len__(self) -> int: ...
    def columns(self) -> T_ls: ...                  # inherited, safe list
    def __getitem__(self, sel: int | slice | str) -> Daf: ...   # LIMIT/OFFSET, or by key
    def to_daf(self) -> Daf: ...                    # pull into memory, NULL restored
    def sql(self, stmt: str) -> Daf: ...            # general fallback, result in memory

    # selecting: each returns a TEMP VIEW
    def select_where(self,
            where:      str | T_da | Callable,      # SQL text, a dict, or a Python function
            cols:       T_ls | None = None,         # columns a Python function needs
            ) -> 'SqlDaf': ...
    def select_records_daf(self,
            keys_ls:    'T_ls | SqlDaf',            # a list, or a SqlDaf whose first column is keys
            inverse:    bool = False,               # True: the rows whose key is not there
            ) -> 'SqlDaf': ...                      # a semi-join, or with inverse an anti-join
    def select_kcols(self, kcols: T_ls) -> 'SqlDaf': ...        # some columns
    def concat(self, other_sdaf: 'SqlDaf') -> 'SqlDaf': ...     # UNION ALL
    def join(self,
            other_sdaf: 'SqlDaf',
            how:        str = 'inner',
            tag_other:  bool = False,               # as Daf.join: suffix the other's columns
            ) -> 'SqlDaf': ...

    # keeping
    def materialize(self) -> 'SqlDaf': ...          # TEMP TABLE, for a result read many times
    def save_as(self, name: str) -> 'SqlDaf': ...   # a real table with this name
    def to_csv(self, path: str) -> None: ...        # streamed, not built in memory
```

Two of these change what an existing name accepts, and need their own review later:
select_records_daf takes a SqlDaf as the keys, and select_where takes SQL text or a dict.

## The core of locate_variants, sketched

Two inputs, with an optional modified CVR, and compare_inputs true.

```python
def locate_variants_sdaf(
        store:              SqlStore,
        audit_sdaf:         SqlDaf,             # votes from the audit
        cvr_sdaf:           SqlDaf,             # CVR votes, original
        modi_sdaf:          SqlDaf | None,      # CVR votes, modified, if any
        flag_cols_lol:      T_lols,             # flag columns for audit and CVR
        compare_cols_ls:    T_ls,               # columns compared for agreement
        ) -> T_dosdaf:                          # the named result tables

    input_lsdaf: List[SqlDaf] = [audit_sdaf, cvr_sdaf]
    keys_lsdaf:  List[SqlDaf] = [sdaf.select_kcols(['ballot_id']) for sdaf in input_lsdaf]

    residue_lsdaf: List[SqlDaf] = [input_lsdaf[i].select_records_daf(keys_lsdaf[1 - i], inverse=True)
                                   for i in range(2)]
    in_lsdaf:      List[SqlDaf] = [input_lsdaf[i].select_records_daf(keys_lsdaf[1 - i])
                                   for i in range(2)]

    flagged_keys_sdaf: SqlDaf = (in_lsdaf[0].select_where(any_flag_sql(flag_cols_lol[0])).select_kcols(['ballot_id'])
                                 .concat(in_lsdaf[1].select_where(any_flag_sql(flag_cols_lol[1])).select_kcols(['ballot_id'])))
    if modi_sdaf is not None:
        flagged_keys_sdaf = flagged_keys_sdaf.concat(modi_sdaf.select_kcols(['ballot_id']))

    rest_lsdaf: List[SqlDaf] = [in_sdaf.select_records_daf(flagged_keys_sdaf, inverse=True)
                                for in_sdaf in in_lsdaf]

    pair_sdaf:           SqlDaf = rest_lsdaf[0].join(rest_lsdaf[1], tag_other=True)
    disagreed_keys_sdaf: SqlDaf = pair_sdaf.select_where(any_differ_sql(compare_cols_ls)).select_kcols(['ballot_id'])
    # concise votes: .select_where(lambda row: normalize(row['concise_votes'])
    #                                        != normalize(row['concise_votes_other']), cols=[...])

    variant_keys_sdaf: SqlDaf = flagged_keys_sdaf.concat(disagreed_keys_sdaf).materialize()

    result_dosdaf: T_dosdaf = {}
    for i, side in enumerate(['audit', 'cvr']):
        result_dosdaf[f'{side}_variants']    = in_lsdaf[i].select_records_daf(variant_keys_sdaf).save_as(f'{side}_variants')
        result_dosdaf[f'{side}_nonvariants'] = in_lsdaf[i].select_records_daf(variant_keys_sdaf, inverse=True).save_as(f'{side}_nonvariants')
        residue_lsdaf[i].to_csv(f'{side}_residue.csv')
    return result_dosdaf
```

any_flag_sql() and any_differ_sql() would be two small helpers in AuditEngine that build the
SQL text, such as `"flag_a" = 1 OR "flag_b" = 1`.

Only variant_keys_sdaf is materialized, since four views read it. Everything else runs once,
when save_as() writes the four tables. The intermediate tables of today, disagreed and
Temp_AgreedKeys, are never written.

## Steps 12 on: contest-level tables

These explode concise_votes into one row per contest, with json_each over a registered Python
function. That has no Daf method today. In the sketch it stays a `sql()` call, or becomes a
new method later. Steps 12 to 14 can be sketched next if this part looks right.
