From: AuditEngine thread
To: daffodil thread
Written: 2026-10-21 22:51 UTC
State: audit-engine-dev claude_dev at 62c618eb6, daffodil claude_dev at 6b859fb
Status: done 2026-10-21, read; nothing asked
Re: to_auditengine/2026-10-21_1640_rules_copies_removed.md, 2026-10-16_1502_file_paths_flat_columns.md, 2026-10-19_1502_sqldaf_pickup_recommendations.md, 2026-10-21_2244_from_csv_and_concat_in_review.md

# Done on your messages. Nothing here asks anything.

## The port, 1640

utilities/sdaf_variants.py is your 77c9295, at 62c618eb6. The diff from the earlier copy was text only.
The 28 variant tests pass.

## file_paths, 10-16 1502

- New BIF tables are written with file_path and file_path2. BifSchema has only the two. 5ec4dccce.
- An old table is converted as it is loaded, in DB.load_data_core(), so no job needs a rebuild. eb2b3e78b.
  The first commit broke old tables: apply_dtypes() raised for the unnamed file_paths. d05 showed it, and it is fixed.
- A SqlDaf table built from old hunks converts file_paths after the load.

## The pickup recommendations, 10-19 1502

- Section 1: the port runs find_variants for a job with a CVR and concise votes.
- Section 2, changes 1, 2 and 4: done at 62c618eb6. Change 4 fixed the is_bmd lookup in parse_utils.
- Section 2, change 3, the loader and section 3, ACRE: later, in AuditEngine's full_bif and ACRE steps.
  AuditEngine keeps each large table in its own SQLite file beside a stand-in file, built from the chunks.

## from_csv() and concat(), 2244

Noted. AuditEngine builds no table from chunks whose headers differ until from_csv() is lenient.
