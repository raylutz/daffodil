# Add-ons

An add-on is a package that adds its own methods to Daf or to SqlDaf. It needs no change to
daffodil. Its methods sit under one name, as in `daf.stats.share('qty')`. pandas, xarray and
polars use the same pattern for their add-ons.

## Writing an add-on

An add-on is a class. Its constructor takes the table as its only argument. The decorator
`register_daf_namespace()` adds it to every Daf, under the name given.

```pycon
>>> from daffodil.daf import Daf, register_daf_namespace
>>> @register_daf_namespace('stats')
... class Stats:
...     def __init__(self, daf):
...         self._daf = daf
...
...     def share(self, col):
...         """ Each row's part of the column total, keyed by the keyfield. """
...         total = sum(self._daf.col(col))
...         return {key: round(val / total, 2)
...                 for key, val in zip(self._daf.keys(), self._daf.col(col))}
```

Then every Daf has `stats`:

```pycon
>>> daf = Daf(cols=['id', 'g', 'qty'], keyfield='id',
...           lol=[['a1', 'x', 3], ['b2', 'y', 10], ['c3', 'x', 7]])
>>> daf.stats.share('qty')
{'a1': 0.15, 'b2': 0.5, 'c3': 0.35}
```

## Using an add-on

An add-on is active once its package is imported. Nothing loads add-ons by itself, so importing
daffodil never runs code from another package.

Reading `daf.stats` makes a new Stats object for that Daf, each time. It is not kept on the Daf.
So a selection, a copy or any other Daf a method returns has the add-on too, bound to its own rows:

```pycon
>>> daf.select_by_dict({'g': 'x'}).stats.share('qty')
{'a1': 0.3, 'c3': 0.7}
```

## Type checkers

mypy and editors do not see a name added while the program runs. mypy reports
`"Daf" has no attribute "stats"`. For code that is type checked, use the add-on's class directly.
It gives the same result, and mypy checks every call:

```pycon
>>> Stats(daf).share('qty')
{'a1': 0.15, 'b2': 0.5, 'c3': 0.35}
```

## Daf and SqlDaf

A SqlDaf keeps its rows in SQLite, not in memory. So an add-on written for a Daf does not work on
a SqlDaf. Reading it there raises NotImplementedError:

```pycon
>>> import tempfile
>>> from daffodil.sqldaf import SqlDaf, register_sqldaf_namespace
>>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
>>> sdaf = SqlDaf.from_daf(daf, name='items')
>>> sdaf.stats
Traceback (most recent call last):
...
NotImplementedError: The add-on 'stats' is registered for Daf, not for SqlDaf. Its package can register a version for SqlDaf too.
```

An add-on can register a SqlDaf version with `register_sqldaf_namespace()`. It may use the same
name. This version sums in SQLite:

```pycon
>>> @register_sqldaf_namespace('stats')
... class SqlStats:
...     def __init__(self, sdaf):
...         self._sdaf = sdaf
...
...     def share(self, col):
...         total = self._sdaf.sum([col])[col]
...         return {key: round(val / total, 2)
...                 for key, val in zip(self._sdaf.keys(), self._sdaf.col(col))}
```

Then a Daf gets the Daf version, and a SqlDaf and its selections get the SqlDaf version:

```pycon
>>> sdaf.stats.share('qty')
{'a1': 0.15, 'b2': 0.5, 'c3': 0.35}
>>> type(daf.stats).__name__, type(sdaf.stats).__name__
('Stats', 'SqlStats')
```

An add-on registered only for SqlDaf is not seen on a Daf.

## Names

An add-on never replaces a method, so a name already in use raises ValueError. In use means a
method or an attribute of the class or of its tables, or another add-on under that name:

```pycon
>>> register_daf_namespace('select_where')(Stats)
Traceback (most recent call last):
...
ValueError: Daf.select_where is already a method or an attribute.
>>> class OtherStats:
...     def __init__(self, daf):
...         self._daf = daf
>>> register_daf_namespace('stats')(OtherStats)
Traceback (most recent call last):
...
ValueError: Daf.stats is already registered, by __main__.Stats.
```

The message names the module and the class that holds the name. Here the module is `__main__`.

The name must be a public identifier, so it cannot start with an underscore. Registering the
same class again under the same name is allowed, so a module can be reloaded.

To remove an add-on, delete its name from the class:

```pycon
>>> del Daf.stats, SqlDaf.stats
>>> hasattr(daf, 'stats'), hasattr(sdaf, 'stats')
(False, False)
>>> SqlDaf.close()
```

## Reference

- [register_daf_namespace()][daffodil.daf.register_daf_namespace]
- [register_sqldaf_namespace()][daffodil.sqldaf.register_sqldaf_namespace]
