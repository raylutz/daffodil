"""marshmallow: a dataclass for the record, a schema to load and dump it."""
import json
from dataclasses import dataclass
from marshmallow import Schema, fields, post_load

@dataclass
class Batch:
    batch_id:     str
    precinct:     str
    ballot_type:  str
    num_ballots:  int
    is_duplicate: bool
    flagged:      bool = False

class BatchSchema(Schema):
    batch_id     = fields.Str(required=True)
    precinct     = fields.Str(required=True)
    ballot_type  = fields.Str(required=True)
    num_ballots  = fields.Int(required=True)
    is_duplicate = fields.Bool(required=True)
    flagged      = fields.Bool(load_default=False)

    @post_load
    def make(self, data, **kwargs):
        return Batch(**data)

schema = BatchSchema(many=True)

def load(text: str) -> list[Batch]:
    return schema.loads(text)

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
    return schema.dumps(batches)
