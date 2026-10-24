From: AuditEngine thread
To: daffodil thread
Written: 2026-10-24 23:48 UTC
State: audit-engine-dev claude_dev at cca5e3432, daffodil claude_dev at 4002b87
Status: open, in review with Ray as Q293, 2026-10-25

# A dict or list cell written to a SqlDaf

Ray asks for this. A Daf takes a dict cell, and a SqlDaf raises. Please make a SqlDaf store it as text.

## What happens at 4002b87

    daf = Daf(cols=['id', 'cv'], lol=[['a1', {'x': 1}]], keyfield='id')
    daf.update_record_krow('a1', {'cv': {'x': 2}})      # works
    daf.append({'id': 'b2', 'cv': {'y': 1}})             # works

    SqlDaf.from_daf(daf, name='t1')                       # ProgrammingError: type 'dict' is not supported
    sdaf.update_record_krow('a1', {'cv': {'x': 2}})      # the same
    sdaf.append({'id': 'b2', 'cv': {'y': 1}})             # the same

A text value, such as "{'x': 3}", updates fine.
A table built by from_csv() holds a dict column as its CSV text, and reads it back as a dict when its dtype is dict.

## The ask

When a SqlDaf writes a cell that is a dict or a list, it stores the text Daf.to_csv_file() writes for that cell.
This covers from_daf(), append(), extend(), update_record_krow() and assignment with brackets.
A column with dtype dict or list then reads the value back as a dict or list, as a table from from_csv() does.

## Why AuditEngine needs it

AuditEngine's corrections log updates rows of a run table in a working copy, on either class.
The votes table has dict columns. On a SqlDaf copy it keeps those cells as text, and on a Daf it decodes them.
So the call site chooses the dtypes by class. With the change, one way works on both, and that choice goes away.
