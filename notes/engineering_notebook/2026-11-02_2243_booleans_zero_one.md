# Booleans are 0 and 1: the reasons and the decisions

Started 2026-11-02, from the detail that CLAUDE.md held. CLAUDE.md keeps the rules.
The public statement is the Guide page docsite/guide/booleans.md.

## Decisions

- Q299, Ray, 2026-10-26: a boolean in any daffodil table is the int 0 or 1, never True or False.
  Converted to str it is '0' or '1'. Released in 0.9.0 as a breaking change.
- Q300, Ray, 2026-10-26: values are put into daffodil's form once, when the row is built.
  Bools become 0 and 1, and None becomes NULL. Daffodil does not re-normalize on a write or in
  `apply_dtypes()`, since that would cost a pass over every dict and list cell each time.
- Q334, Ray, 2026-11-02: a cell in this form is PYON and JSON5 at once. Earlier text said JSON.

## Why 0 and 1

- The words differ from one system to the next. JSON writes true and false. Python writes True
  and False. SQLite, Excel and others use TRUE and FALSE. Any of them makes the data hard to work
  with, and asks for conversions back and forth.
- 0 and 1 mean the same in all of them.
- A table written out and read back then compares equal to its source as text, with no special
  case for bools.
- A dict or list cell, written as PYON with 0 and 1, is also JSON5. SQLite's JSON functions read
  JSON5 since SQLite 3.42, so SQL can read inside the cell. SqlDaf needs SQLite 3.48.
  Checked 2026-11-02: Python's own sqlite3 on the EC2 box is 3.37.2, and its json_extract()
  refuses `{'news': 1}` as malformed. Daffodil's SQLite 3.53.1 reads it.
  Python's json module refuses it too, since JSON needs double quotes.

## How the code follows it

- Converting to bool gives 0 or 1. `apply_dtypes()` with dtype bool reads the words, in any case,
  and gives the int.
- Converting to str gives '0' or '1'. A Python True or False is made an int first, then text.
- `normalize_val()` in daf_utils.py turns bools into 0 and 1, and None into NULL, at any depth in a
  dict, list or tuple.

## Accepted risk

A type checker sees an int, which may hold a value other than 0 or 1. The Booleans Guide page
states this.
