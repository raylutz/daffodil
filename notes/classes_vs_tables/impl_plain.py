"""Plain classes: everything a class needs is written by hand, with no library to generate it."""
import json

class Batch:
    def __init__(self, batch_id: str, precinct: str, ballot_type: str, num_ballots: int,
                 is_duplicate: bool, flagged: bool = False) -> None:
        self.batch_id     = batch_id
        self.precinct     = precinct
        self.ballot_type  = ballot_type
        self.num_ballots  = num_ballots
        self.is_duplicate = is_duplicate
        self.flagged      = flagged

    def __repr__(self) -> str:
        return (f'Batch(batch_id={self.batch_id!r}, precinct={self.precinct!r}, ballot_type={self.ballot_type!r}, '
                f'num_ballots={self.num_ballots!r}, is_duplicate={self.is_duplicate!r}, flagged={self.flagged!r})')

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Batch):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    @classmethod
    def from_dict(cls, d: dict) -> 'Batch':
        for name in ('batch_id', 'precinct', 'ballot_type', 'num_ballots', 'is_duplicate'):
            if name not in d:
                raise ValueError(f'{name} is missing')
        extra = set(d) - {'batch_id', 'precinct', 'ballot_type', 'num_ballots', 'is_duplicate', 'flagged'}
        if extra:
            raise ValueError(f'unknown fields: {sorted(extra)}')
        if not isinstance(d['num_ballots'], int) or isinstance(d['num_ballots'], bool):
            raise ValueError(f"num_ballots is not an int: {d['num_ballots']!r}")
        if not isinstance(d['is_duplicate'], bool):
            raise ValueError(f"is_duplicate is not a bool: {d['is_duplicate']!r}")
        return cls(d['batch_id'], d['precinct'], d['ballot_type'], d['num_ballots'],
                   d['is_duplicate'], d.get('flagged', False))

    def to_dict(self) -> dict:
        return {'batch_id': self.batch_id, 'precinct': self.precinct, 'ballot_type': self.ballot_type,
                'num_ballots': self.num_ballots, 'is_duplicate': self.is_duplicate, 'flagged': self.flagged}

def load(text: str) -> list[Batch]:
    return [Batch.from_dict(d) for d in json.loads(text)]

def dump(batches: list[Batch]) -> str:
    return json.dumps([b.to_dict() for b in batches])

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
