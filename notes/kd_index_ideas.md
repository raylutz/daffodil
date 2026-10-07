# Ideas for a smaller key index (kd)

Parked by Ray on 2026-10-07. Most Daf tables in AuditEngine are small, so these gain little.
Revisit only if a survey of table sizes in AuditEngine shows many large keyed tables.

## Measurements

Python 3.10, 1,000,000 string keys, on the EC2 box.

| Index | Memory | Lookup |
|---|--:|--:|
| `kd` as today, `dict(zip(keys, range(n)))` | 70 MB | `lol[kd[k]]` 288 ns |
| of which, the position ints | 28 MB | |
| key to row list, `dict(zip(keys, lol))` | 42 MB | 257 ns |

- Python shares the ints from -5 to 256. A table with up to 257 rows or columns pays nothing
  for the positions in `kd` or `hd`. Above that, each position is its own 28-byte int, and
  each table has its own copies.
- `dict(zip(keys, range(len(keys))))` in `_build_hd()` is the fastest pure-Python build:
  351 ms for a million keys, against 388 ms for the enumerate comprehension, and 1.4 times
  faster at 1,000 keys. It runs with no Python loop per item.
- Most of a lookup is the dict lookup itself, which is limited by memory speed. The step
  `lol[...]` is 39 ns of the 288.

## Options

1. Shared pool of position ints. One module-level list of ints, grown as needed, used by
   `_build_hd()` and the line in `KeyedIndex.__init__`. About an hour of work. Rebuilds are
   about 15% faster, and large tables share one set of ints. The pool only grows, so give it
   a cap, such as 10 million positions.
2. Key to row list. 40% smaller and 11% faster for lookups, but a key no longer gives its
   row number. Rejected: too many methods need positions.
3. Indexed set in C. A hash table that returns the entry offset, with no values stored.
   About 30 MB per million keys, positions included. Needs a compiled extension.
