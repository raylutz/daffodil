# find_variants on SqlDaf: a sketch

Written on 2026-10-13 for Ray, as part of questions 96 to 100. Signatures only. Nothing here is
built. It follows decisions 88 to 99 in design_draft.md. The source read is locate_variants()
in AuditEngine's utilities/sql_variants.py, lines 25 to 1440, at claude_dev 283d992f0.

Ray chose, on 2026-10-13, to follow today's steps one for one. An earlier version of this
sketch rebuilt the steps as sets of keys. Under decisions 97 and 99, that is not needed: a
selection is cheap, and records may be deleted, so each step maps directly.

Ray: row order is not important in find_variants.

## The steps, now and on SqlDaf

Two inputs, audit votes and CVR votes, and an optional third, the modified CVR. `i` is 0 or 1,
and `1 - i` is the other side.

| Step | Now                                                    | On SqlDaf                                    |
| ---: | :----------------------------------------------------- | :------------------------------------------- |
|    1 | Drop old variants tables.                              | `store.drop_table(name, missing_ok=True)`    |
|    2 | Create empty variants tables, unique index on ballot_id. | `input[i].clone_empty(name=...)`, a base table with a unique keyfield |
|    3 | residue[i] = input[i] rows whose key is not in the other input. DELETE them from input[i]. Export residue to CSV. | a selection with `inverse=True`, `save_as()`, `delete_records()`, `to_csv()` |
|    4 | INSERT OR IGNORE flagged rows into variants[i].        | `variants[i].concat(input[i].select_where(...), skip_existing_keys=True)` |
|    5 | INSERT OR IGNORE rows whose key is in the other variants table. | the same, with `select_records_daf(variants[1 - i])` |
|    6 | INSERT OR IGNORE rows whose key is in the modified CVR. | the same, with `select_records_daf(modi)`   |
|    7 | nonvariants[i] = input[i] rows whose key is not in variants[i]. | a selection with `inverse=True`, `save_as()` |
|    8 | disagreed = nonvariants[0] JOIN nonvariants[1] where a compared column differs. | `join(..., tag_other=True).select_where(...)`, a selection of pairs |
|    9 | Concise: normalize concise_votes, put agreed keys in a TEMP table, DELETE them from disagreed. | a second `select_where()` with a Python function, on the first selection. No table, no delete. |
|   10 | INSERT OR IGNORE nonvariant rows whose key is in disagreed. | `concat(..., skip_existing_keys=True)`       |
|   11 | DELETE from nonvariants[i] the keys in variants[i].     | `nonvariants[i].delete_records(variants[i])` |
|  12+ | Contest-level tables, with json_each over a registered extract_contests(). | `sql()` for now                         |

Row counts for the report come from `len()`, which is free on a selection that was read.

## Signatures

### SqlStore

```python
class SqlStore:
    def __init__(
            self,
            path:       str = '',                           # database file; '' makes a temp file
            conn:       sqlite3.Connection | None = None,   # wrap an existing connection instead
            ) -> None: ...

    def table(self, name: str, keyfield: str = '') -> 'SqlDaf': ...            # an existing table
    def from_csv(self, path: str, name: str = '', keyfield: str = '',
                 dtypes: T_dtype_dict | None = None) -> 'SqlDaf': ...         # load without building in memory
    def from_daf(self, data_daf: Daf, name: str) -> 'SqlDaf': ...             # write an in-memory Daf
    def drop_table(self, name: str, missing_ok: bool = False) -> None: ...
    def close(self) -> None: ...                         # TEMP objects and a temp file go
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
    def __getitem__(self, sel: int | slice | str) -> Daf: ...   # by position or by key
    def to_daf(self) -> Daf: ...                    # pull into memory, NULL restored
    def to_csv(self, path: str) -> None: ...        # streamed, not built in memory
    def sql(self, stmt: str) -> Daf: ...            # general fallback

    # selecting: each returns a selection of references (decision 97)
    def select_where(self,
            where:      str | T_da | Callable,      # SQL text, a dict, or a Python function
            cols:       T_ls | None = None,         # columns a Python function needs
            ) -> 'SqlDaf': ...
    def select_records_daf(self,
            keys_ls:    'T_ls | SqlDaf',            # a list, or a SqlDaf: its keyfield gives the keys
            inverse:    bool = False,               # True: the rows whose key is not there
            ) -> 'SqlDaf': ...
    def select_kcols(self, kcols: T_ls) -> 'SqlDaf': ...
    def join(self,
            other_sdaf: 'SqlDaf',
            how:        str = 'inner',
            tag_other:  bool = False,               # as Daf.join: suffix the other's columns
            ) -> 'SqlDaf': ...                      # a selection of pairs

    # making and changing tables
    def clone_empty(self, name: str) -> 'SqlDaf': ...           # empty base table, same columns
    def concat(self,
            other_sdaf:         'SqlDaf',
            skip_existing_keys: bool = False,       # True: INSERT OR IGNORE
            ) -> 'SqlDaf': ...
    def delete_records(self, keys_ls: 'T_ls | SqlDaf') -> int: ...  # number deleted
    def materialize(self) -> 'SqlDaf': ...          # TEMP TABLE of the rows
    def save_as(self, name: str) -> 'SqlDaf': ...   # a real table with this name
```

Three of these need their own review before they are built:

- `select_records_daf()` taking a SqlDaf as the keys, and `select_where()` taking SQL text or a
  dict. Both widen an existing Daf name.
- `concat(skip_existing_keys=True)`. Daf's `concat(respect_kd=True)` replaces a row with a
  duplicate key. find_variants needs the opposite.
- `delete_records()`. A Daf has no in-place delete, so this name is new.

## The core of locate_variants, sketched

```python
def locate_variants_sdaf(
        store:              SqlStore,
        input_lsdaf:        List[SqlDaf],       # audit, CVR, and optionally the modified CVR
        variants_names_ls:  T_ls,               # names of the two variants tables
        nonvariants_names_ls: T_ls,             # names of the two nonvariants tables
        flag_cols_lol:      T_lols,             # flag columns for audit and CVR
        compare_cols_ls:    T_ls,               # columns compared for agreement
        use_concise:        bool,
        ) -> T_lods:                            # report rows, as now

    side_lsdaf: List[SqlDaf] = input_lsdaf[:2]
    modi_sdaf:  SqlDaf | None = input_lsdaf[2] if len(input_lsdaf) == 3 else None

    # steps 1 and 2
    for name in variants_names_ls:
        store.drop_table(name, missing_ok=True)
    variants_lsdaf: List[SqlDaf] = [side_lsdaf[i].clone_empty(name=variants_names_ls[i]) for i in range(2)]

    # step 3
    for i in range(2):
        residue_sdaf: SqlDaf = side_lsdaf[i].select_records_daf(side_lsdaf[1 - i], inverse=True).save_as(f'residue_{i}')
        residue_sdaf.to_csv(f'residue_{i}.csv')
        side_lsdaf[i].delete_records(residue_sdaf)

    # step 4
    for i in range(2):
        if use_concise:
            flagged_sdaf: SqlDaf = side_lsdaf[i].select_where(
                    lambda row: has_variant_flag(row['concise_votes']), cols=['concise_votes'])
        else:
            flagged_sdaf = side_lsdaf[i].select_where(any_flag_sql(flag_cols_lol[i]))
        variants_lsdaf[i].concat(flagged_sdaf, skip_existing_keys=True)

    # steps 5 and 6
    for i in range(2):
        variants_lsdaf[i].concat(side_lsdaf[i].select_records_daf(variants_lsdaf[1 - i]), skip_existing_keys=True)
        if modi_sdaf is not None:
            variants_lsdaf[i].concat(side_lsdaf[i].select_records_daf(modi_sdaf), skip_existing_keys=True)

    # step 7
    nonvariants_lsdaf: List[SqlDaf] = [
            side_lsdaf[i].select_records_daf(variants_lsdaf[i], inverse=True).save_as(nonvariants_names_ls[i])
            for i in range(2)]

    # steps 8 and 9
    pair_sdaf: SqlDaf = nonvariants_lsdaf[0].join(nonvariants_lsdaf[1], tag_other=True)
    if use_concise:
        disagreed_sdaf: SqlDaf = (pair_sdaf
                .select_where('"concise_votes" != "concise_votes_other"')
                .select_where(lambda row: normalize(row['concise_votes']) != normalize(row['concise_votes_other']),
                              cols=['concise_votes', 'concise_votes_other']))
    else:
        disagreed_sdaf = pair_sdaf.select_where(any_differ_sql(compare_cols_ls))

    # steps 10 and 11
    for i in range(2):
        variants_lsdaf[i].concat(nonvariants_lsdaf[i].select_records_daf(disagreed_sdaf), skip_existing_keys=True)
        nonvariants_lsdaf[i].delete_records(variants_lsdaf[i])

    # steps 12 on: sql() for now
    ...
```

any_flag_sql(), any_differ_sql(), has_variant_flag() and normalize() would stay in AuditEngine.
The first two build SQL text, such as `"flag_a" = 1 OR "flag_b" = 1`. The last two are the
Python functions used today: the REGEXP search on concise_votes, and normalize_pyon.

The tag `_other` for the other side's columns is a placeholder. The real suffix is whatever
`Daf.join(tag_other=True)` uses.

## What changes against today

- Step 9 needs no TEMP table and no DELETE. The agreed pairs are left out by a second
  selection on the first.
- The disagreed tables of step 8 are never written. They are a selection of pairs, read once
  by step 10.
- Every SQL statement and every identifier escape moves into SqlDaf. locate_variants keeps
  only its own logic.

## Still open

- Can an input have a duplicate ballot_id? Today the unique index on the variants tables keeps
  the first copy and drops the rest, and nonvariants keeps every copy. clone_empty() with a
  unique keyfield keeps that, but select_records_daf() on the inputs would see every copy.
- Steps 12 to 14, the contest-level tables, are not sketched yet.
