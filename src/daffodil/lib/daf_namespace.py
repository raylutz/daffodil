# daf_namespace.py
# copyright (c) 2026 Ray Lutz
"""
Add-on namespaces for Daf and SqlDaf.

A package outside daffodil adds its own methods under one name, as `daf.hist.update_with_history()`,
without changing daffodil. It registers a class with `register_daf_namespace()` for a Daf, and with
`register_sqldaf_namespace()` for a SqlDaf. pandas, xarray and polars use the same pattern.

The functions here are shared by the two. Each class calls them with its own names.
"""

import inspect
from typing import Any, Callable, Iterable


class Namespace:
    """
    The class attribute that gives an add-on's object for a table. Internal use.

    Reading it on a table makes a new object of the add-on's class, with the table as its only
    argument. It is not cached on the table. A Daf copies its attributes to a selection, so a
    cached object would follow the selection, still bound to its parent.

    An add-on for tables held in memory, as a Daf, is refused on a SqlDaf, whose rows are in SQLite.
    """

    def __init__(self, name: str, namespace_cls: type, owner_cls: type) -> None:
        self.name:          str     = name
        self.namespace_cls: type    = namespace_cls
        self.owner_cls:     type    = owner_cls

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self.namespace_cls
        if getattr(self.owner_cls, '_rows_in_memory', True) and not getattr(type(obj), '_rows_in_memory', True):
            raise NotImplementedError(
                f"The add-on '{self.name}' is registered for {self.owner_cls.__name__}, not for {type(obj).__name__}. "
                f"Its package can register a version for {type(obj).__name__} too.")
        return self.namespace_cls(obj)


def register_namespace(
        name:           str,                    # the attribute name of the add-on.
        owner_cls:      type,                   # the class it is added to.
        instance_attrs: Iterable[str],          # the names each table of the class sets on itself.
        ) -> Callable[[type], type]:
    """
    A decorator that adds a class of methods to `owner_cls` under `name`. Internal use.

    A name already in use raises ValueError, so an add-on never replaces a method or another
    add-on. In use means a method, an attribute of the class or of its tables, or an add-on already
    registered under that name for this class. An add-on of a parent class may be given its own
    version under the same name. Registering the same class again, by module and name, is allowed,
    so a module can be reloaded.
    """
    if not name.isidentifier() or name.startswith('_'):
        raise ValueError(f"An add-on name must be a public identifier, not {name!r}.")
    if name in set(instance_attrs):
        raise ValueError(f"{owner_cls.__name__}.{name} is an attribute of every {owner_cls.__name__}.")

    def decorate(namespace_cls: type) -> type:
        existing: Any = inspect.getattr_static(owner_cls, name, None)
        if isinstance(existing, Namespace):
            same_cls: bool = (existing.namespace_cls.__module__, existing.namespace_cls.__qualname__) == \
                             (namespace_cls.__module__, namespace_cls.__qualname__)
            if existing.owner_cls is owner_cls and not same_cls:
                raise ValueError(f"{owner_cls.__name__}.{name} is already registered, by "
                                 f"{existing.namespace_cls.__module__}.{existing.namespace_cls.__qualname__}.")
        elif existing is not None:
            raise ValueError(f"{owner_cls.__name__}.{name} is already a method or an attribute.")
        setattr(owner_cls, name, Namespace(name, namespace_cls, owner_cls))
        return namespace_cls

    return decorate
