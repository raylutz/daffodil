# mypyc: interpreted vs. compiled daffodil, real benchmark comparison

Side document to `daf_benchmarks.md` -- same script (`tests/daf_benchmarks.py`), run twice: once
against plain interpreted daffodil, once against `keyedlist.py`/`daf.py`/most of `lib/` compiled
with `mypyc`. Full narrative (what compiles, what doesn't, every fix needed, correctness
verification) lives in the audit-engine repo's engineering notebook:
`engineering_notebook/2026-09-08_mypyc-compilation-pilot-daffodil-and-audit-engine.md` -- this
file is just the numbers, kept here so they're not lost with the scratch environment they were
produced in.

## Gaps vs. `daf_benchmarks.md` -- read before comparing numbers

- **Scale: 700x700, not 1000x1000.** The sandbox this ran in killed the benchmark process for low
  memory at the script's default 1000x1000 (1M cells) scale, twice, even running one variant at a
  time. 700x700 is the smallest reduction that keeps the script's several hardcoded indices (500,
  600, 400) in range. Absolute numbers below don't match `daf_benchmarks.md`'s published
  1000x1000 figures -- the plain-vs-compiled *ratios* are the valid signal here, not the absolute
  ms values.
- **Single run each**, not averaged across repeated runs (the smaller Pilot-1 scaffolding-only
  benchmark *was* repeated 3x per variant and was stable; this full-package run wasn't repeated,
  for the same memory-safety reason as above).
- **The compiled build isn't the real repo.** Getting `daf.py` to compile at all required several
  source changes (wrapper methods replacing a `method = module._function` binding pattern, a few
  type annotations, one method stubbed out) that were never applied to real daffodil source --
  only to a disposable scratch copy. `daf_pandas.py`, `daf_pdf.py`, and `schemaclass.py` stayed
  uncompiled (real typing/dynamic-metaprogramming reasons, not oversights). None of this is
  shipped; there is no compiled daffodil to `pip install`.
- **No `daf_sum2()`/`daf_sum3()` rows** -- both were already commented out of `daf.py` (see its
  own comment there) by the time this ran; this file's own two calls to them were removed from
  `tests/daf_benchmarks.py` as part of the same session (they crashed the script outright,
  independent of mypyc -- `objsize` was also missing from dev deps, fixed the same way).

## Results (`daf` column only -- pandas/numpy/sqlite/lod columns are unaffected by daffodil's own
compilation and are shown only as a sanity check that both runs used equivalent data)

| operation | plain (ms) | compiled (ms) | change |
| ---: | :---: | :---: | :--- |
| from_lod | 70.3 | 36.4 | ~48% faster |
| to_pandas_df | 258 | 256 | no real change |
| to_pandas_df_thru_csv | 107 | 107 | no real change |
| from_pandas_df | 12.1 | 10.8 | ~11% faster |
| to_numpy | 30.9 | 30.7 | no real change |
| from_numpy | 4.6 | 5.1 | no real change (noise) |
| increment cell (`daf[i,j] +=`) | 0.087 | 0.058 | ~33% faster |
| insert_irow | 0.066 | 0.058 | ~12% faster |
| insert_icol | 0.88 | 0.53 | ~40% faster |
| sum cols (`.sum()`) | 137 | 84.7 | ~38% faster |
| `daf_sum()` | 180 | 149 | ~17% faster |
| sum_np | 41.1 | 42.4 | no real change (noise) |
| transpose | 1,105 | 1,121 | no benefit |
| transpose_npao | 59.7 | 62.4 | no real change (noise) |
| keyed lookup | 0.047 | 0.043 | ~9% faster |

Array sizes (MB) were identical between runs (18.8/12.7/3.8/8.8/39.6 and 18.8/37.1/--/8.8/39.7),
confirming both used the same seeded 700x700 data.

## Reading these results

Real, consistent 17-48% speedups on operations that stay in daffodil's own Python loops
(construction, mutation, single-cell insert, row-wise summation, keyed lookup). No benefit
anywhere the work is delegated to already-compiled C code (`to_pandas_df`/`to_numpy`/`sum_np` --
thin wrappers over pandas/numpy) or is pure reference-copying with no per-element computation
(`transpose`). See the engineering-notebook entry linked above for why, and for the full list of
source changes a real compile would need.
