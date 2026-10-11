"""Make a JSON list of ballot-batch records, the same each time."""
import json, random

def make_records(n: int) -> list[dict]:
    rng = random.Random(1)
    return [{'batch_id':     f'b{i:06d}',
             'precinct':     f'P{rng.randrange(40):02d}',
             'ballot_type':  rng.choice(['mail', 'in_person', 'provisional']),
             'num_ballots':  rng.randrange(1, 600),
             'is_duplicate': rng.random() < 0.02,
            } for i in range(n)]

if __name__ == '__main__':
    print(json.dumps(make_records(3), indent=1))
