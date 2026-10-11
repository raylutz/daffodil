#!/bin/bash
# Run one of these scripts with marshmallow, pydantic and this repo's daffodil.
# Example: notes/classes_vs_tables/run.sh faults.py
cd "$(dirname "$0")"
exec uv run --no-project --python 3.13 --with marshmallow --with pydantic --with pyjson5 --with-editable ../.. python "$@"
