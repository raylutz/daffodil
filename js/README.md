# daffodil-csv.js

A small, dependency-free (beyond the vendored JSON5 parser) JS reader for CSV files produced by
`Daf.to_csv_buff()` -- for browser-side static apps that need to consume daffodil-exported data
with no backend and no build step. Not a JS port of the `Daf` class itself: no tabular
transforms, no in-memory table operations, just "CSV file in, plain JS objects out," with the same
PYON-unflattening behavior `Daf.from_csv_buff()` applies on the Python side.

## Usage

```html
<script src="vendor/json5.js"></script>
<script src="daffodil-csv.js"></script>
<script>
  fetch('data.csv').then(r => r.text()).then(text => {
    var rows = Daffodil.csvToObjects(text);   // array of plain objects, one per CSV row
  });
</script>
```

Pass `{unflatten: false}` as a second argument to get raw string cells only, matching
`Daf.from_csv_buff(unflatten=False)`.

## Why this exists

`Daf.to_csv_buff()` flattens any `dict`/`list`/`tuple`-valued column using Python's own `__repr__`
("PYON" -- like JSON, but single-quoted strings, non-string dict keys, and `True`/`False`/`None`
instead of `true`/`false`/`null`), and `Daf.from_csv_buff()` reads it back the same shape-driven
way (`daf_utils.unflatten_val()`): if a cell looks like `{...}`, `[...]`, or `(...)`, try to parse
it as a Python literal, no explicit per-column schema required.

JSON5 already accepts PYON's single-quoted-string/dict/list syntax directly -- the only real gap
is the capitalized Python keywords, which `pyonKeywordsToJson5()` rewrites before handing the text
to JSON5, tracking quote state so a value that legitimately contains the word "None" or "True"
inside a string is left alone. Python tuples (`(...)`) are treated as JS arrays, since JSON has no
tuple type either.

`vendor/json5.js` is the unmodified upstream `json5` npm package (MIT licensed, see
`vendor/LICENSE-json5.md`), vendored rather than loaded from a CDN so pages using this stay fully
self-contained and offline-capable.

## What this deliberately does NOT do

- No bool-column unflattening (`True`/`False` as bare CSV cells, not inside a dict/list) -- this
  matches `Daf.from_csv_buff()`'s own default behavior without an explicit `dtypes` declaration
  (bool conversion is dtype-driven in `apply_dtypes()`, not shape-driven like dict/list/tuple).
  Columns that need a real boolean should use plain `0`/`1` instead, which round-trips as a normal
  int with no ambiguity either side.
- No writer (`Daf.to_csv_buff()` has no JS-side counterpart here) -- this is a reader only.
