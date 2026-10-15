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
        input_losdaf:        List[SqlDaf],       # audit, CVR, and optionally the modified CVR
        variants_names_ls:  T_ls,               # names of the two variants tables
        nonvariants_names_ls: T_ls,             # names of the two nonvariants tables
        flag_cols_lol:      T_lols,             # flag columns for audit and CVR
        compare_cols_ls:    T_ls,               # columns compared for agreement
        use_concise:        bool,
        ) -> T_lods:                            # report rows, as now

    side_losdaf: List[SqlDaf] = input_losdaf[:2]
    modi_sdaf:  SqlDaf | None = input_losdaf[2] if len(input_losdaf) == 3 else None

    # steps 1 and 2
    for name in variants_names_ls:
        store.drop_table(name, missing_ok=True)
    variants_losdaf: List[SqlDaf] = [side_losdaf[i].clone_empty(name=variants_names_ls[i]) for i in range(2)]

    # step 3
    for i in range(2):
        residue_sdaf: SqlDaf = side_losdaf[i].select_records_daf(side_losdaf[1 - i], inverse=True).save_as(f'residue_{i}')
        residue_sdaf.to_csv(f'residue_{i}.csv')
        side_losdaf[i].delete_records(residue_sdaf)

    # step 4
    for i in range(2):
        if use_concise:
            flagged_sdaf: SqlDaf = side_losdaf[i].select_where(
                    lambda row: has_variant_flag(row['concise_votes']), cols=['concise_votes'])
        else:
            flagged_sdaf = side_losdaf[i].select_where(any_flag_sql(flag_cols_lol[i]))
        variants_losdaf[i].concat(flagged_sdaf, skip_existing_keys=True)

    # steps 5 and 6
    for i in range(2):
        variants_losdaf[i].concat(side_losdaf[i].select_records_daf(variants_losdaf[1 - i]), skip_existing_keys=True)
        if modi_sdaf is not None:
            variants_losdaf[i].concat(side_losdaf[i].select_records_daf(modi_sdaf), skip_existing_keys=True)

    # step 7
    nonvariants_losdaf: List[SqlDaf] = [
            side_losdaf[i].select_records_daf(variants_losdaf[i], inverse=True).save_as(nonvariants_names_ls[i])
            for i in range(2)]

    # steps 8 and 9
    pair_sdaf: SqlDaf = nonvariants_losdaf[0].join(nonvariants_losdaf[1], tag_other=True)
    if use_concise:
        disagreed_sdaf: SqlDaf = (pair_sdaf
                .select_where('"concise_votes" != "concise_votes_other"')
                .select_where(lambda row: normalize(row['concise_votes']) != normalize(row['concise_votes_other']),
                              cols=['concise_votes', 'concise_votes_other']))
    else:
        disagreed_sdaf = pair_sdaf.select_where(any_differ_sql(compare_cols_ls))

    # steps 10 and 11
    for i in range(2):
        variants_losdaf[i].concat(nonvariants_losdaf[i].select_records_daf(disagreed_sdaf), skip_existing_keys=True)
        nonvariants_losdaf[i].delete_records(variants_losdaf[i])

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

- Duplicates, settled by decision 103: a duplicate ballot_id raises at the append, and is never
  skipped. So `concat(skip_existing_keys=True)` above becomes
  `concat(rows_sdaf.select_records_daf(target_sdaf, inverse=True))`. To mention to AuditEngine:
  today INSERT OR IGNORE in sql_variants.py drops a duplicate silently, if one reaches
  find_variants. Ray: duplicates must never be ignored.
- Steps 12 to 14, the contest-level tables, are not sketched yet.

## Trial, 2026-10-14

notes/sql_backing/find_variants_trial.py ports steps 1 to 11 one for one with SqlDaf, as above,
and runs on a copy of a cmpcvr.db that AuditEngine's find_variants has already written. It
rebuilds the inputs from the trimmed input tables and their residue tables, runs the port, and
compares each result table with AuditEngine's, as sets of rows. AuditEngine's own code and data
are not changed; its databases are copied first.

| Job                                    | Inputs | Audit ballots | Time   | Result |
| :------------------------------------- | -----: | ------------: | -----: | :----- |
| WI_Dane_20201103_functest              | 2      | 198           |        | all tables the same |
| GA_Bartow_20201103_functest            | 3      | 214           |        | all tables the same |
| US_MD_Somerset_20221108_ocrmapped      | 2      | 6,957         | 1.4 s  | all tables the same |
| US_CO_Broomfield_20231107_aimapper     | 2      | 26,286        |        | all tables the same |
| CA_SanFrancisco_20241105_pilot_d05     | 3      | 34,534        | 4.0 s  | all tables the same |
| US_MD_Rockvillecity_20231107_ocrmapped | 2      | 12,637        | 1.3 s  | one ballot differs, see below |

The times include copying the database and comparing. Peak memory was 105 MB at most.

The trial also builds the step report of sql_variants.py: the same columns and step
descriptions, with the rows and seconds of each step for each side. For San Francisco, every
row count of steps 0 to 11 is the same as in AuditEngine's saved reports/Find_Variants_Report.md,
and steps 1 to 11 took about 1.0 s in all, against about 0.9 s there.

The column num_contests is left out of the comparison. cmpcvr.py fills it into audit_nonvariants
with add_computed_column() after locate_variants() returns, so steps 1 to 11 never set it.

Rockville: the port finds ballot 2760 disagreed, and AuditEngine's run of 2026-09-10 did not.
Its concise votes differ in contest c1, 4 votes against 5, and AuditEngine's own
normalize_pyon() says so too. AuditEngine's own step 8 query, run read-only on its stored
nonvariant tables, finds 2760. But the disagreed tables in that file are empty. In
sql_variants.py, step 8 makes them with CREATE TABLE IF NOT EXISTS, and they are never dropped,
while cmpcvr.db is kept from run to run. So on a rerun, step 8 keeps the previous run's
disagreed tables, and a new disagreement is missed. This is a bug in AuditEngine, not in the
port. Not reported to AuditEngine yet; that is Ray's call.

The five large jobs were run on 2026-10-14, one at a time, Ray's question 120:

| Job                             | Audit ballots | Steps 1 to 11, port | Same, AuditEngine | Result |
| :------------------------------ | ------------: | ------------------: | ----------------: | :----- |
| GA_Paulding_20201103            | 85,589        | 3.0 s               | 3.7 s             | 209 ballots differ, see below |
| WI_Dane_20201103_aimapper       | 333,574       | 5.7 s               | 10.8 s            | all tables the same |
| FL_Collier_20201103_ocrmapped   | 417,726       | 6.7 s               | 12.7 s            | all tables the same |
| WI_Dane_20241105_aimapper       | 367,179       | 7.6 s               | 19.1 s            | all tables the same |
| GA_Dekalb_20220524_aimapper     | 142,810       | 10.7 s              | 103.1 s           | all tables the same |

The times are the sums of the step times in each report, for both sides: the port's, and
AuditEngine's saved reports/Find_Variants_Report.md from its own run, on whatever load the box had
then. Run alone on Collier, the port took 13.0 s of wall time for steps 1 to 11, counting the
row counts made for the report, and peak memory stayed at 40 MB, the memory of the import. The
whole trial, with the copy and the comparison, peaked at 0.4 to 0.8 GB, because the comparison
pulls every result row into Python sets.

Paulding: the same stale-table bug as Rockville, larger. AuditEngine's report agrees with the
port through step 7, row for row. Its step 8 shows 18 disagreed ballots in 0.00011 s, which is no
work: CREATE TABLE IF NOT EXISTS found the tables of an earlier run and kept them. AuditEngine's
own step 8 query, run read-only on its stored nonvariant tables, still finds 209 ballots whose
concise votes differ after normalize_pyon(). 18 + 209 = 227, the port's count. So that run left
209 disagreeing ballots among the nonvariants.
