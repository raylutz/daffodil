"""pydantic: one model class does the record, the checks, and load and dump."""
from pydantic import BaseModel, TypeAdapter

class Batch(BaseModel):
    batch_id:     str
    precinct:     str
    ballot_type:  str
    num_ballots:  int
    is_duplicate: bool
    flagged:      bool = False

adapter = TypeAdapter(list[Batch])

def load(text: str) -> list[Batch]:
    return adapter.validate_json(text)

def totals(batches: list[Batch]) -> dict:
    out: dict = {}
    for b in batches:
        if b.is_duplicate:
            continue
        out[b.precinct] = out.get(b.precinct, 0) + b.num_ballots
    return out

def flag(batches: list[Batch]) -> None:
    for b in batches:
        b.flagged = b.num_ballots > 500

def dump(batches: list[Batch]) -> str:
    return adapter.dump_json(batches).decode()
