# dbglue review, and a plan for a daffodil ledger app (draft)

Written 2026-10-12, from dbglue.pl version 103 (2020-05-13), its use cases, glue_stmts.sh and
ChkgDldLinux_GlueSpec.ods. This is a draft for discussion with Ray. Nothing here is decided.

The use cases, the Perl and the spec are not in this repo. The repo is public, and the use cases
hold real names, emails, addresses and SSN-format numbers.

## 1. What dbglue does

dbglue reads a GlueSpec, which is a table with one row per directive and one column per file:
OUTFILE, INFILE1, INFILE2 and so on. It reads the input files, finds the related row in each
other file, converts each value, and writes one output row per INFILE1 row.

- **Files and tags.** File names hold `%TAG%` and `%TAG2%`, filled in from the command line.
  glue_stmts.sh runs one spec per company and account for a month.
- **Header handling.** `Header` can be Yes, No, Ignore, Combine or Verify. Verify checks that the
  file has at least the columns named in `Fieldlist`. `Skiprows` drops lines by number, counting
  from either end, for report titles and TOTAL lines.
- **Order.** `Reverse` turns a newest-first bank file into date order.
- **Finding the related row.** `Join` is an equi-join on a hash index, and the key must be unique.
  Otherwise `Wherecond` is a Perl expression, tried on each row from the top, and the first
  match wins. The payee lookup is `IS2IN1(Description, PartialDescription)`, a case-insensitive
  substring test, with an optional match on the amount too.
- **Output fields.** Each `Outfield` names its sources and a chain of conversions:
  `Payment;ABSNEGONLY;DEC;BLANKIFZERO`.
- **Other formats.** Fixed-width blocks of 512 characters for SSA W-2 files, a separator of 0x1C
  for FEC files, tagged tables, column totals and record counts, and templates for mail merge.

## 2. The conversions it needed

These are what a bank and accounting import has to handle. They're the main lessons.

| Need | dbglue | What it does |
|---|---|---|
| Money text to a number | `NUM`, `sanitize_numeric` | Drops spaces, `$`, `,` and `+`. `(45.00)` becomes -45 |
| Split one signed amount | `ABSNEGONLY`, `ABSPOSONLY` | Negative to Payment, positive to Deposit, both as positive values |
| Absolute value | `ABS` | QB writes withholding as negative; forms want it positive |
| Two decimals | `DEC`, `FMT=%1.2f` | Fixed text with cents |
| Currency text | `DOLLAR`, `COMMA` | `$3,467.20` |
| Implied cents | `FIXPT` | 2,345.67 written as `00000234567` for SSA |
| Blank for zero | `BLANKIFZERO` | An empty cell where the amount is 0 |
| Digits only | `DIGITS`, `F=%09u` | SSNs and zip codes, keeping leading zeros |
| Numeric or text compare | `ISEQUAL` | Check `000306` equals `306`, and `-$45.00` equals `(000045)` |
| Dates | `DATE=` | MM/DD/YY, MM/DD/YYYY, YYYY-MM-DD, YYYYMMDD, DD Mon YYYY, with a century guess |
| Defaults and constants | `DEF=`, `CONSTIFIS` | Fixed codes, such as FEC form types |
| First non-empty | `EITHER` | Memo from the bank comment, else from the lookup |
| Sum of fields | `SUM` | Total wages from several pay columns |
| Join text | `L=", "` | Street1 and Street2 into one address |
| Required fields | `;M` | An error when a mandatory output is empty |
| Checks | `TYPE=` | Email, phone, zip, state, SSN, tax id, country |

## 3. Bugs found in dbglue's number code

I copied `COMMA_format`, `DEC_format`, `FIXPT_format` and `sanitize_numeric` out of dbglue.pl and
ran them with Perl 5.34 on this box:

```
input         DOLLAR      DEC       FIXPT
'1234.5'      $1,234.50   1234.50   123450
'-45'         $44.99      -45.00    18446744073709547117
'(45.00)'     $44.99      45.00     18446744073709547117
'-$1,234.565' $1,234.56   -1234.57  18446744073709428160
'1.005'       $1.01       1.00      100
'2.675'       $2.68       2.67      268
'0.125'       $0.13       0.12      013
'-0.004'      $0.00       -0.00     000
'45-'         $45.00      45.00     4500
```

- **`DOLLAR` and `COMMA` lose the sign, and are a cent off for negative values.** They add half
  a cent and cut the digits, then strip everything that isn't a digit, the minus included.
- **`DEC` of `(45.00)` is positive.** `sanitize_numeric` already makes it negative, and
  `DEC_format` then negates it again.
- **`FIXPT` of any negative value is garbage.** `%u` of a negative number wraps around.
- **The rounding differs between operators.** `DEC` uses binary floats, so 1.005 gives 1.00 and
  2.675 gives 2.67. `DOLLAR` adds half a cent first, so it gives 1.01 and 2.68.
- **`-0.004` gives `-0.00`.**
- **A trailing minus, `45-`, is read as positive.** Some banks and reports write negatives that way.
- **An empty amount becomes 0.** `sanitize_numeric` returns 0 for blank text.

The bank specs use `ABSNEGONLY;DEC` on a signed amount, so they probably avoid the worst of these,
since `ABS` comes first. Statement amounts are entered to the cent, so the float rounding rarely
shows. Sums of many floats can still drift by a cent.

## 4. Line endings

dbglue.pl itself has CRLF line endings. Perl reads such a script, but it's worth converting.

The CSV reader splits a file on `\n` and strips any trailing `\r` and `\n` from each line, so CRLF
input on its own should not break it. These could still cause trouble:

- **A file with CR only.** It would be read as one line. That's rare in bank exports.
- **A quoted field that runs over several lines.** The lines are joined with nothing between them.
- **Output.** It's written with `\n`. Windows tools may want `\r\n`.
- **A line that is just `0`.** It's skipped, since Perl reads it as false.

To find the actual cause, I'd need the symptom and a sample input. A bank file with the amounts and
descriptions changed would do.

## 5. A plan for the daffodil version

The goal: replace the hand entry into QuickBooks, keep a ledger, and produce the yearly reports
for each account for taxes.

### 5.1 Steps

1. **Extract.** Read each raw bank download as it is. Keep it unchanged in its month folder, as now.
2. **Normalize.** Map each bank's columns to one transaction table:
   `account, date, check_no, description, amount, balance, source_file, source_row`.
   The date is ISO. The amount is signed and exact, never a float.
3. **Validate.** Check the running balance against the bank's balance column, row by row. That
   catches a missing or doubled row at once. Check each required field. Rows that fail go to a
   reject file and stop the load, instead of being skipped.
4. **Categorize.** Fill in payee, account (the category) and memo from the payee rules. List the
   rows with no match, to add rules for.
5. **Load.** Append to a ledger, one per year, without duplicates. Loading the same download
   twice, or two downloads that overlap, must not double any row.
6. **Report.** Totals by category and account for the year, which is the tax report. Monthly
   registers, in date order, for reconciliation. Transfers between your own accounts, such as
   paying the Visa from checking, are shown once and kept out of the expense totals.

### 5.2 Design choices for discussion

- **Bank formats as Python, not a GlueSpec.** Each bank gets a short mapping: its column names,
  its date format, how its amount is signed, and whether it's newest first. A spreadsheet spec
  is good for people who don't program, but it grew into a language with `Wherecond` written in
  Perl. Python is clearer here and can be tested.
- **The payee rules stay a table you edit.** PayeeLookup works well as data. Keep its columns:
  PartialDescription, Amount, Name, Account, LookupMemo. Two changes are worth weighing.
  The longest match could win instead of the first, so the order of the rules stops mattering:
  today `Amazonprime` must come before `Amazon`. And every rule's match count could be reported,
  so rules that are never used show up.
- **Money as Decimal, rounded half up to the cent.** Daffodil sums Decimal exactly today:
  `daf_sum()` of 1.10 and 2.20 gives exactly 3.30. A float sum of 0.1 and 0.2 gives
  0.30000000000000004. Formatting happens only on output.
- **Provenance.** Every ledger row keeps its source file and row, so each total can be traced
  back to the bank line.
- **Keys for duplicates.** A bank transaction id where the bank gives one. Otherwise account,
  date, amount, description and balance together, since the balance makes two identical
  purchases on one day different.

### 5.3 What this asks of daffodil

These are the parts that test the design:

- **Reading `.ods`.** Daffodil reads CSV and xlsx, not ods. PayeeLookup and the specs are ods.
- **A money type.** A dtype that reads `$1,234.56`, `(45.00)`, `45-` and blank into Decimal or
  NULL, and writes it back with two decimals. This is the `sanitize_numeric` lesson, done once
  and tested.
- **Dates.** A dtype or a converter for the bank date formats.
- **Rule lookup.** This isn't a key join, so it's `apply()` or a loop with a substring test.
  How fast is that for a few thousand rows against a few hundred rules?
- **Reversing rows, and sorting by date then by row.**
- **Validation.** `check_schema()` on each load, plus the balance check.
- **Reports.** `groupby_cols_reduce()` with a sum, which is now 4 to 5 times faster, on Decimal.
- **Files of a fixed format,** such as FEC with 0x1C or SSA W-2 with fixed width, if you still
  need them. They're writers, not core.

## 6. ETL practice worth keeping

- Raw files are never changed. Every step can be run again from them.
- A load can be run twice with the same result.
- Bad rows stop the load, or go to a reject file. They're never dropped quietly.
- Totals are checked against the source: the statement's start and end balance.
- Each output row can be traced to its input.
- The bank format is checked on each load, as `Header=Verify` did, so a change in a bank's
  export is caught at once.
