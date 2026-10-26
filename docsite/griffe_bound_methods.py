"""
A griffe extension for the docs build. It is not part of the daffodil package.

Some Daf methods are written as functions in a helper module and assigned in the class body.
Such a function takes the table as its first parameter, named self, or cls for a classmethod.
Python binds it when the method is called, so a caller never passes it.
This extension leaves that parameter out of the signature in the docs, as for any other method.
It changes only a module-level function whose name starts with an underscore,
and whose first parameter is named self or cls. Each such method is named that way.
"""

from typing import Any

import griffe


class BoundMethodSignatures(griffe.Extension):
    """ Leave out the self or cls of a module-level function in the docs signature. """

    def on_function_instance(self, *, func: griffe.Function, **kwargs: Any) -> None:
        if not func.name.startswith('_') or not func.parent or not func.parent.is_module or not len(func.parameters):
            return
        first: griffe.Parameter = next(iter(func.parameters))
        if first.name in ('self', 'cls'):
            del func.parameters[first.name]
