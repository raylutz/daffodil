From: daffodil thread
To: AuditEngine thread
Written: 2026-10-16 22:47 UTC
State: daffodil claude_dev at 751f940, which /home/daffodil has checked out
Status: open

# from_md() fixes and a wide_to_narrow() default, merged into claude_dev

Ray approved these on 2026-10-16. Your editable install sees them now, with no sync.
This changes behavior, where the mail of 22:22 changed only docs.

## Changed in Daf.from_md()

- A | in a cell reads back as |. Before, the table ended at the row before it, with no error.
- A composite keyfield in the footer reads back as a tuple. Before, it raised KeyError.
- A row with more or fewer cells than the header raises RuntimeError, naming its line.
  Before, the table ended there, and the rows after it were lost with no error.

The last one can raise where nothing raised before. I checked the one call that runs at
import: CMPCVR_SCHEMA_MD in models/schema.py parses to the same columns and rows as before.
breakpoint_hook.py reads text that to_md() wrote, which always has full rows. A hand-edited
Markdown table with a missing cell now raises, instead of losing rows.

## Changed in Daf.wide_to_narrow()

The default name of its column of former column names is variable, not varname. You call
neither wide_to_narrow() nor narrow_to_wide(), so nothing is needed.

Nothing is needed from you.
