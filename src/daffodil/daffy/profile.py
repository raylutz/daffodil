# profile.py -- optional sidecar metadata for a CSV file: display widths, keyfield, dtypes.
#
# Naming convention (Ray, 2026-09-27): "foo.csv,profile.json" -- the sidecar name is the CSV's
# own full filename (extension included) with ",profile.json" appended, same directory. This
# keeps the CSV's real name fully visible in a directory listing and avoids ever colliding with
# a real data file (no CSV file is named "*.csv,profile.json").
#
# dtypes is a plain {colname: typename} map, not a daffodil @schemaclass -- Ray, 2026-09-27:
# "Basic schema is only to convert str to python types when needed... dtypes_dict is sufficient
# for that as it is the result of the schemaclass anyway." Daf.set_dtypes()/apply_dtypes()
# consume exactly this shape already; no schema-discovery machinery needed for Daffy's own use.
# Deliberately minimal -- more fields get added if real usage shows they're needed, not upfront.

import json
from pathlib import Path
from typing import Any, Dict, Optional

PROFILE_FIELDS = ('keyfield', 'widths', 'dtypes')


def profile_path_for(csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.name + ',profile.json')


def load_profile(csv_path: str | Path, explicit_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """ Load the sidecar profile for csv_path, or {} if none exists.
        explicit_path overrides discovery (--profile PATH on the CLI).
    """
    path = Path(explicit_path) if explicit_path else profile_path_for(csv_path)
    if not path.exists():
        return {}
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Profile file {path} must contain a JSON object, got {type(data).__name__}")
    return data


def save_profile(csv_path: str | Path, profile: Dict[str, Any], explicit_path: Optional[str | Path] = None) -> Path:
    """ Write the sidecar profile, preserving any fields Daffy doesn't itself recognize
        (a hand-added field, or one from a future version, survives an update unchanged).
    """
    path = Path(explicit_path) if explicit_path else profile_path_for(csv_path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, sort_keys=True)
        f.write('\n')
    return path
