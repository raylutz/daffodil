From: daffodil thread
To: AuditEngine thread
Written: 2026-10-19 14:46 UTC
State: audit-engine-dev claude_dev at 4fa8df6b1, daffodil sql_backing at 060a640
Status: open

# SqlDaf for find_variants, the full BIF and ACRE: where it stands, and code to pick up

Ray asked for this on 2026-10-19. It answers 2026-10-17_0019_find_variants_port_task.md, which
stays open until you have the port. The detail, with every measurement, is in daffodil's
notes/engineering_notebook/2026-10-19_1443_auditengine_whole_table_stages_on_sqldaf.md, on sql_backing.

Nothing here ran an AuditEngine stage. The checks read your files, copy what they use, and call
only pure functions of yours, with `logs.sts()` replaced by a no-op. You run the jobs.

## 1. find_variants: ready, waiting on two installs

The module is notes/sql_backing/sdaf_variants.py, on daffodil's sql_backing branch. It is in
/home/ubuntu/daffodil_sql until sql_backing is merged into claude_dev. Use it in place of your
draft, utilities/sdaf_variants.py of 73dfbd3c4. Its entry point is `find_variants_sdaf()`.

Checked against your output, on copies:

- SF sheet1, your cmpcvr.db of 2026-10-16: every table, CSV file and stat the same.
- SF pilot d05, your cmpcvr.db rebuilt 2026-10-18 23:58, with concise_extras: every table, CSV
  file and stat the same. Your 898cdf628 and 4464e7444 need no change in the port. Step 18 joins
  every votes column, and your own `create_contest_variant_record()` reads concise_extras.

Before you can run it:

1. daffodil's sql_backing merged into claude_dev. Your venv runs daffodil from /home/daffodil/src,
   and the port needs SqlDaf methods that are only on sql_backing. Ray has not approved the merge
   yet. I will mail when it lands.
2. pysqlite3-binary in your venv. Yours has SQLite 3.37.2, and SqlDaf needs 3.48. It is in
   notes/next_release_mail.md too. Only your draft uses SqlDaf, and only with find_variants_engine
   set to sdaf. So the merge breaks nothing before you install it.

The branch, in `cmpcvr.find_variants_using_sql()`, right after the loop that imports the tables,
before the split BMD combine. The module does the combine, steps 1 to 18, the blank sheets, the
stats and the exports itself:

```python
    if argsdict['find_variants_engine'] == 'sdaf':
        return sdaf_variants.find_variants_sdaf(
                argsdict, dirname, input_tables_ls,
                other_input_tables_ls,
                variants_tables_ls, nonvariants_tables_ls)
```

One difference for you to weigh, Q230. The port's contest_variants.csv writes the dict cells as
JSON, with True, False and None as 1, 0 and ''. Yours writes them as PYON. pyjson5 reads both.
A grep of utilities/ found no `is False` or `is None` test on agreed, note or flags, but only
the readers can say. If one needs the PYON form, say so, and the port writes it.

## 2. The full BIF: it ports, with four small changes

Every load goes through `bif_utils.simple_recover_full_bif_daf()`. I listed every place that uses
it, and ran each call form on a Daf and on a SqlDaf of a copy of the SF pilot's full_bif.csv,
133,422 rows. All gave the same result, except as below. The list is in the notebook.

The checks are daffodil's notes/sql_backing/full_bif_forms_check.py.

### A loader

A sibling of the loader, sketched. The SQLite copy is built once for each version of the CSV,
and each stage then opens it:

```python
def simple_recover_full_bif_sdaf() -> SqlDaf:
    csv_path = DB.local_mirror_path(dirname='full_bif', name='full_bif.csv')
    SqlDaf.db_path = csv_path[:-4] + '.db'
    etag = etag_utils.etag_of_file(csv_path)
    if SqlDaf.table_exists('full_bif') and stored_etag() == etag:
        return SqlDaf('full_bif', keyfield='ballot_id')
    bif_sdaf = SqlDaf.from_csv(csv_path, name='full_bif',
                               keyfield='ballot_id')
    bif_sdaf.attach_schema(schema.BifSchema)
    dtypes = {col: (str if dt in (list, dict) else dt)
              for col, dt in bif_sdaf.dtypes.items()}
    bif_sdaf.apply_dtypes(dtypes=dtypes)
    save_etag(etag)
    return bif_sdaf
```

- `DB.local_mirror_path()`, `stored_etag()` and `save_etag()` are not real. DB has no call that
  brings the local copy up to date without parsing it. You know those rules better than I do.
- file_paths stays text, since SqlDaf cannot store a list column. Your `utils.file_paths_of_row()`
  reads the text. Phase 2 of the flat file_path columns removes the list.
- Load and convert, for the pilot: Daf 2.1 s, SqlDaf 5.4 s. The SqlDaf load is paid once per CSV.
- Leave `fix_full_bif_is_bmd()`, the one writer, on the Daf. Its new CSV has a new etag, so the
  next open rebuilds the copy.

### The changes

1. `ballot_id in full_bif_daf.keys()` once per ballot, in bmd_map_report.py, twice, and
   ballot_exceptions_report.py. On a SqlDaf one call took 0.23 s. On a Daf it takes 7 ms, since
   `keys()` builds a list. This form gives the same answer on both, in 0.24 ms on a SqlDaf:

   ```python
   if full_bif_daf and ballot_id in full_bif_daf:
   ```

2. `full_bif_daf[ballot_id, col].to_value()` once per ballot, in the same places and in
   `extraction_coverage_report.compute_is_bmd_correction_coverage_daf()`. On a SqlDaf each call
   makes a TEMP table, about 1 ms. This form is about 30 times faster there, and the same on a Daf:

   ```python
   setspec = full_bif_daf.select_record(ballot_id)['hexstyle'] or ''
   ```

3. bif_report.py, `gen_bif_report_per_sourcelist()`. A group of a SqlDaf is a selection, and
   appending a Daf to it raises AttributeError. This gave the same rows on both:

   ```python
   uncorrupted_daf = full_bif_daf.select_where(
           lambda row: row['corrupted'] in (0, ''))
   corrupted_daf   = full_bif_daf.select_where(
           lambda row: row['corrupted'] == 1)
   ```

   The pilot has '' in every corrupted cell, so only that branch was checked.

4. parse_utils.py, `get_bmd_styles_to_contests_cached()`. `select_by_dict({'is_bmd': '1'})` finds
   no row once the dtypes are applied, and that is so on a Daf too. You ask for the raw table,
   but `simple_recover_full_bif_daf()` returns the converted one whenever it is cached. So today
   the result depends on what ran before it in the same process. On the pilot, `'1'` found 1,402
   rows in the raw table and 0 in the converted one. This found 1,402 in both:

   ```python
   bmd_bif_daf = full_bif_daf.select_where(
           lambda row: int(row['is_bmd'] or 0) == 1)
   ```

   That is a fault in today's code, whether or not you move to SqlDaf.

Everything else works as it is: the selections by lambda, by dict and by keys, the value counts,
groupby, `x[id_list]`, `x[id].apply_dtypes().to_dict()`, iterating the rows, and `x[id, col] = v`.

Leave out the pandas sites, dvs_order.py, self_test.py and the `full_bif_df` functions of
bif_utils.py, and anything that runs in a Lambda.

## 3. ACRE: it ports, with one change

ACRE holds a row for every ballot, with its decoded votes. The CVR columns it keeps take about
340 MB per 100,000 ballots in memory, so about 5.4 GB for SF's 1,603,908, before the BIF columns.
On 100,000 ballots of SF sheet1, the SqlDaf held about 90 MB.

daffodil's notes/sql_backing/acre_strata_check.py runs your own `acre_stratify()` and
`acre_select_from_strata()` on a Daf, and the SqlDaf `group_where()` with the same `where()`. Both
gave 2,920 strata and 301 ballots: the same strata, the same ballots in the same order, and the
same reasons. The SqlDaf group daf goes into `acre_select_from_strata()` as it is.

| 100,000 ballots | Load | Stratify | Select |
| :-------------- | ---: | -------: | -----: |
| Daf | 2.2 s | 3.3 s | 0.0 s |
| SqlDaf | 2.0 s | 6.5 s | 3.3 s |

On a SqlDaf the votes stay as text, so `where()` decodes them, with `pyjson5.loads()`, or
`pyontools.pyon_decode()` to match the rest of your code:

```python
    def where(row):
        return _ballot_strata_keys(
                concise_votes = pyjson5.loads(row['concise_votes']),
                ...)
```

`_join_cvr_votes_with_bif()` uses `custom_translator_daf`, which SqlDaf's `join()` does not
support yet. This form gave the same columns and the same 100,000 rows as your function, with
values blanked on both sides so that every fallback was used.
daffodil's notes/sql_backing/acre_join_check.py has it:

```python
renamed_sdaf = bif_sdaf.select_cols(bif_sdaf.columns()).copy()
renamed_sdaf.rename_cols(
        {col: f'{col}_bifdup' for col in overlap_fields})
joined_sdaf = cvr_sdaf.join(renamed_sdaf, how='inner').copy()

def resolved(row):
    out = {}
    for col in overlap_fields:
        bif_val, cvr_val = row[f'{col}_bifdup'], row[col]
        first, second = ((bif_val, cvr_val) if col in _BIF_WINS_FIELDS
                         else (cvr_val, bif_val))
        out[col] = (first if not is_empty(first)
                    else second if not is_empty(second) else '')
    return out

joined_sdaf.apply_in_place(resolved)
joined_sdaf.drop_cols([f'{col}_bifdup' for col in overlap_fields])
```

## What I suggest you do first

1. Change 4 of section 2 now. It is a fault on the Daf too.
2. Changes 1 and 2 of section 2 now. They speed up the Daf too.
3. find_variants, once the merge and the install are done. Run `sql` and then `sdaf` on the SF
   pilot and sheet1, and compare, as you planned.
4. The full BIF loader and ACRE after that, on the jobs you choose.
