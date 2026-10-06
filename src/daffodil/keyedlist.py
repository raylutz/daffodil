# keyedlist.py

from typing import List, Dict, Any, \
                    Callable, KeysView, Tuple

from collections.abc import Hashable, Iterable, Iterator                    

TKey = Hashable
T_la = List[Any]

import json


class KeyedList:
    """
    A row that reads like a dict but points at a list instead of copying its values.

    A KeyedList pairs a list of values with an index of the keys. You read it by key, as in
    `row['qty']`, the same way you read a dict. But the values are not copied into it. It
    adopts the list you give it. It can also share one index of keys with many other
    KeyedLists, which is what a table does with the column names of its rows.

    Daffodil gives you its rows as KeyedLists when `itermode` is 'keyedlist', and from
    `iter_klist()` and `iloc()`. Each one points at a row of the table.

    How it differs from a dict:

    - Writing to it writes to the list it adopted. If that list is a row of a Daf, the Daf
      changes. A dict row never changes the table.
    - `values()` returns the adopted list itself, not a copy. Use `to_dict()` for an
      independent copy.
    - Creating one is cheap if you give it an existing index. A dict copies every value in.
    - Keys must be unique, and keys and values must be the same length.

    Ways to create one:

    - `KeyedList(keys, values)`: adopts the values list. No copy.
    - `KeyedList(index, values)`: the same, but reuses a `KeyedIndex` that you already built.
      This is the fastest, and it is what Daf uses for its rows.
    - `KeyedList(a_dict)`: copies the values out of the dict. Slower than the forms above.
    - `KeyedList(keys, default=0)`: every key gets the same default value.
    - `KeyedList(other_klist)`: shares the values list and copies the index.

    Reading and writing:

    - Read with `row['a']`, `row.get('a', default)`, `'a' in row` or `row.items()`. A list of
      keys, as in `row[['a', 'b']]`, returns a list of values and skips keys that are missing.
    - Assigning to an existing key replaces the value in the list.
    - Assigning to a new key adds the key, and adds the value to the end of the list. If the
      list is a row of a Daf, that row is now longer than the others.
    - Deleting a key removes its value from the list.
    - A KeyedList that shares an index copies it before it adds a key. So adding a key to
      one row does not give that key to the other rows.

    Speed:

    Looping over a table as KeyedLists can be faster than looping over dicts, even if you
    never assign to a row. All the rows share one index, so no row builds its own. The gain
    grows with the number of columns. In a test with 50,000 rows of 400 columns, reading
    one field in a loop took 0.047 s with KeyedLists and 0.583 s with dicts. With 5 columns
    the two were about even.

    Because a KeyedList points into the table, assigning to it changes the table. If you
    need a copy, use `to_dict()`, or loop with `iter_dict()` on the Daf.

    Examples:
        >>> values = [1, 2, 3]
        >>> klist = KeyedList(['a', 'b', 'c'], values)
        >>> klist['b']
        2
        >>> klist['b'] = 20
        >>> values
        [1, 20, 3]
        >>> klist.to_dict()
        {'a': 1, 'b': 20, 'c': 3}
        >>> klist.get('z', 0)
        0

        A row of a Daf is a KeyedList that points at the row in the table:

        >>> from daffodil.daf import Daf
        >>> daf = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
        >>> for row in daf.iter_klist():
        ...     row['y'] = 0
        >>> daf
        <BLANKLINE>
        | x  | y  |
        | -: | -: |
        |  1 |  0 |
        |  3 |  0 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """
    """
    KeyedList is a custom data structure in Python that combines the functionality of a dictionary and a list,
    optimized for efficient indexing and manipulation of data. 
    
    It maintains an index for fast key-based access to values stored in a list. 
    This is similar to a conventional dictionary, but the list items are
    not distributed to each item in the dict, but can be an existing list, used without copying.
    
    hd (header dict) -- Implemented as KeyedIndex(keys)
    
    # The keys are implemented as a dictionary of indexes, i.e. {'key0': 0, 'key1', 1, ... 'keyn': n} where the keys
    # are just examples here. For convenience, we call this structure a "header dict" or 'hd'. 
    # To create this the following can be used:
    
    #     hd = KeyedIndex(keys)   # used to be dict(zip(keys, range(len(keys))))
        
    # Which is more performant and equivalent to:
    
    #     hd = {col: idx for idx, col in enumerate(keys)}
        
    # But this likely can be improved if a concise standard library function is created, since the current nature
    # of a dict is that it is ordered, and thus has an implied index.

    Keys must be unique. Duplicate keys will raise an error during construction or append.
    
    Values
    
    For values that are already stored as a list, the list can be adopted as values without copying. An 
    important attribute of this approach is that the parent list is modified if values in the KeyedList 
    are modified, and vice versa. The code should make a copy if the values in the source list need to 
    remain unaltered, or convert to a conventional dict which will inherently make a copy, such as by using .to_dict()
    
    Similarly, if the hd portion already exists, it can be reused on many instances of keyed list. Further, the hd can 
    be used with a list-of-list structure as the column indexes of all rows. If related to such an array, then the
    keys are frequently called 'cols'. See the daffodil package for a full implementation of such a dataframe array.

    Usage:
        - KeyedList can be initialized from either a list of keys and values, a dictionary, or an existing hd and
          list of values, or from another KeyedList.
          
        - It supports standard dictionary operations such as __getitem__, __setitem__, __delitem__, __len__,
          __iter__, keys, values, items, get, update, and conversion to a conventional dictionary using .to_dict().
          
        - KeyedList instances provide fast key-based access to values similar to dictionaries, while also allowing
          list-like operations for efficient value manipulation.
          
        - Most importantly, creation of a KeyedList instance is much faster than creating a conventional dict, because
          the list can be adopted as a reference without copying.

    Initialization:
        - KeyedList(keys_iter, values_list) -> KeyedList: Initialize KeyedList from a list of keys and values_list. 
            Like dict(zip(keys, values_list)) but the values_list is adopted without copying.
            
        - KeyedList(full_dict) -> KeyedList: Initialize KeyedList from a full_dict, a conventional dictionary.
            This method of initialization will copy the values and is relatively expensive.
            
        - KeyedList(keys_iter, default=None) -> KeyedList: Initialize KeyedList from a list of keys and constant 'default' 
            Like dict.fromkeys(keys, default). If unspecified, default is None.
            
        - KeyedList(hd, values_list) -> KeyedList: Initialize KeyedList from an existing hd and values_list.
            This is very fast because no copying occurs. 
    
    Operation
        - A KeyedList instance acts just like a conventional dictionary, but it can be much less expensive to use,
            because the hd portion can be reused, and the values can be adopted without copying. However, there is 
            a slight penalty in access because of the additional indirection.
            - values are stored in a list
            - keys are managed by a KeyedIndex
            - structural mutations rebuild the index
            
        - If a KeyedList is created from an associated 'record' in a list-of-list (lol) with an associated hd, then
            the KeyedList can be created by reference to the hd and a list in the lol array. Changes to the 
            KeyedList instance will update the lol array, because the list item is actually the same list a the one in the array.
            This behavior is not possible with dictionaries.
            
        - As a result, a KeyedList instance is more 'dangerous' for beginner Python programmers. Dicts are constructed
            always by copying in the keys and values and do not maintain the connection to the prior source of the list
            portion. Instead, a KeyedList instance may may have only references to existing hd and values.
            
        - similar to a dictionary, a KeyedList can provide the keys and values as iterators or lists. The difference is that
            a KeyedList allows assignment to the hd and the list.
            
            klist = KeyedList({'a': 5, 'b' 8})
            print(klist)            # output: {'a': 5, 'b': 8}
            print(klist.keys())     # output: dict_keys(['a', 'b'])
            print(klist.values())   # output: [5, 8]
            alist = klist.values    # grab a reference to the values (no copying)
            alist[1] = 10           # assign a value to the list.
            print(alist)            # output: [5, 10]
            print(klist)            # output: {'a': 5, 'b': 10}  <-- note that klist changes too!
            
            klist._values = [2,3]   # assign new values to the list portion. Assignment like this is NOT supported for dicts.
            print(klist)            # output: {'a': 2, 'b': 3}
            print(alist)            # output: [2, 3]    # note that the list that is a reference to the list portion changes.


        Example2:
            keys = ['a', 'b', 'c']
            values = [1, 2, 3]
            klist = KeyedList(keys, values)     # initialize like you would a dict using dict(zip(keys, values))
            print(klist)                        # Output: {'a': 1, 'b': 2, 'c': 3}
            print(klist['a'])                   # Output: 1
            klist['b'] = 5                      # overwrite a value
            print(klist)                        # Output: {'a': 1, 'b': 5, 'c': 3}
            print(klist.values())               # Output: [1, 2, 3]
            print(klist.values)                 # Output: [1, 2, 3]
           
        See also:
            https://peps.python.org/pep-0412/#alternative-implementation
            
    """

    # True while hd is a KeyedIndex adopted from elsewhere, and may be shared with other KeyedLists.
    # The index is copied before the first new key is added, so no other KeyedList sees that key.
    _hd_shared: bool = False

    def __init__(self, 
            arg1: 'Dict[Any, Any] | List[Any] | KeyedList | KeyedIndex | None' = None, 
            arg2: List[Any] | None = None,
            default: int | str | float | None = None,
            ):
            
        if isinstance(arg1, KeyedIndex) and isinstance(arg2, list):
            # Case: hd + row (critical for reference semantics). This is the case that Daf iteration uses
            # for every row, so it is tested first.
            if len(arg1) != len(arg2):
                raise ValueError("hd and values must have the same length")
            self.hd = arg1              # reuse, DO NOT rebuild
            self._values = arg2         # direct reference
            self._hd_shared = True      # copied before a key is added
            return

        if isinstance(arg1, dict):
            if arg2 is None:
                # Case 1: from_dict
                # self.hd = type(self)._build_hd(arg1.keys())
                self.hd = KeyedIndex(arg1)
                self._values = list(arg1.values())
                return
                
            if isinstance(arg2, list):
                # Case 2: from hd plus values
                if len(arg1) != len(arg2):
                    raise ValueError("keys and values must have the same length")
                self.hd = KeyedIndex(arg1)
                self._values = arg2
                return
            
        elif isinstance(arg1, list) and isinstance(arg2, list):
            # Case 3: from list of keys and values
            if len(arg1) != len(arg2):
                raise ValueError("keys and values must have the same length")
            # self.hd = type(self)._build_hd(arg1)
            self.hd = KeyedIndex(arg1)
            self._values = arg2
            return
            
        elif isinstance(arg1, list) and arg2 is None:
            # Case 4: from list of keys and default
            self.hd = KeyedIndex(arg1)
            self._values = [default] * len(arg1)
            return
            
        elif isinstance(arg1, type(self)) and arg2 is None:
            # Case 5: from keyedlist type
            self.hd = KeyedIndex(arg1)
            self._values = arg1._values
            return
            
        elif arg1 is None and arg2 is None:
            # Case 6, Empty - return a functional empty keyedlist, like {}
            self.hd = KeyedIndex()
            self._values = []
            return
        
        raise ValueError("Must provide either a dict, keys and values, hd and list, or KeyedList")
    
    def __getitem__(self, key: TKey | List[TKey]) -> Any:
        """
        Get the value for a key, or a list of values for a list of keys.

        Args:
            key: A key, or a list of keys.

        Returns:
            The value. For a list of keys, a list of values in that order. A key of the list that is
            missing is skipped, so the list can be shorter.

        Raises:
            KeyError: A single key is not found.
            ValueError: The key is not a list and cannot be hashed.

        Examples:
            >>> klist = KeyedList(['a', 'b', 'c'], [1, 2, 3])
            >>> klist['b']
            2
            >>> klist[['c', 'a']]
            [3, 1]
            >>> klist[['a', 'zz']]
            [1]
        """
        
        try:
            return self._values[self.hd[key]]       # type: ignore[index]  # a list key raises TypeError, handled below.
        except TypeError:
            pass                                    # a list of keys, or a key that cannot be hashed.

        if isinstance(key, list):
            return [self._values[self.hd[onekey]] for onekey in key if onekey in self.hd]
            
        elif isinstance(key, Hashable):
            return self._values[self.hd[key]]

        else:
            raise ValueError
    
    def __setitem__(self, key: TKey, value: Any) -> None:
        """
        Set the value for a key, as `row[key] = value`.

        A key that exists gets the new value in the list that this KeyedList points at. If
        that list is a row of a Daf, the Daf changes. A key that is new is added at the end of
        the keys, and its value at the end of the list, so a row of a Daf is then longer than
        the others. The index is copied before it gets a new key, so other KeyedLists that
        shared it do not get the key.

        Args:
            key: The key.
            value: The value.

        Examples:
            >>> row = [1, 2]
            >>> klist = KeyedList(['a', 'b'], row)
            >>> klist['b'] = 20
            >>> klist['c'] = 30
            >>> row
            [1, 20, 30]
            >>> list(klist)
            ['a', 'b', 'c']
        """
        if key not in self.hd:
            if self._hd_shared:
                self.hd = KeyedIndex(list(self.hd))     # a copy, so other KeyedLists do not get the key
                self._hd_shared = False
            # extend hd
            self.hd.append(key)
            # self.hd[key] = len(self.hd)
            
            self._values.append(value)
        else:    
            self._values[self.hd[key]] = value

    
    def __delitem__(self, key: TKey) -> None:
        """
        Delete a key and its value, as `del row[key]`.

        The value is removed from the list that this KeyedList points at, and the later
        values move down. If that list is a row of a Daf, the row is then shorter than the
        others. The index is rebuilt, and this KeyedList no longer shares it.

        Args:
            key: The key to delete.

        Raises:
            KeyError: The key is not found.

        Examples:
            >>> row = [1, 2, 3]
            >>> klist = KeyedList(['a', 'b', 'c'], row)
            >>> del klist['b']
            >>> row, list(klist)
            ([1, 3], ['a', 'c'])
        """
        # index = self.hd.pop(key)
        # del self._values[index]
                
        # # it is necessary to rebuild hd whenever it is changed.
        # self.hd = type(self)._build_hd(self.hd.keys())        
        index = self.hd[key]
        del self._values[index]

        new_keys = list(self.hd)
        del new_keys[index]

        self.hd = KeyedIndex(new_keys)
        self._hd_shared = False


    def __len__(self) -> int:
        """
        Get the number of values.

        Returns:
            The number of values in the list that this KeyedList points at.

        Examples:
            >>> len(KeyedList(['a', 'b'], [1, 2]))
            2
        """
        return len(self._values)
    
    def __iter__(self) -> Iterator[TKey]:
        """
        Loop over the keys, as a dict does.

        Returns:
            An iterator of the keys, in the order of their values.

        Examples:
            >>> list(KeyedList(['a', 'b'], [1, 2]))
            ['a', 'b']
        """
        return iter(self.hd)

    def __contains__(self, key: object) -> bool:
        """
        Test whether a key is one of the keys, as `key in row`.

        Args:
            key: The key to look for.

        Returns:
            True if the key is found.

        Examples:
            >>> 'a' in KeyedList(['a', 'b'], [1, 2])
            True
            >>> 'z' in KeyedList(['a', 'b'], [1, 2])
            False
        """
        return key in self.hd

    def keys(self) -> KeysView[TKey]:
        """
        Get the keys, in the order of their values.

        This is a keys view of the index itself. When the index is shared with other
        KeyedLists, as it is for the rows of a Daf, it is not a copy. For a list of the
        keys, use `list(row.keys())`.

        Returns:
            The keys.

        Examples:
            >>> KeyedList(['a', 'b'], [1, 2]).keys()
            dict_keys(['a', 'b'])
        """
        return self.hd.keys()

    def set_values(self, new_values: List[Any]) -> None:
        """
        Point this KeyedList at a different list of values.

        This replaces the list that the KeyedList adopted. It does not write into the old
        list. If the old list was a row of a Daf, the row is not changed, and this KeyedList
        stops being a view of it. The new list is adopted, not copied. To change the values
        in the old list, assign to the keys.

        Args:
            new_values: The new list of values, one for each key.

        Raises:
            TypeError: `new_values` is not a list. A tuple is not accepted.
            ValueError: The list does not have one value for each key.

        Examples:
            >>> row = [1, 2]
            >>> klist = KeyedList(['a', 'b'], row)
            >>> klist.set_values([10, 20])
            >>> klist.values(), row
            ([10, 20], [1, 2])
            >>> klist.set_values([1])
            Traceback (most recent call last):
                ...
            ValueError: values length must match keys
        """
        if not isinstance(new_values, list):
            raise TypeError

        if len(new_values) != len(self.hd):
            raise ValueError("values length must match keys")

        self._values = new_values


    def values(self, astype: Callable | str | type | None = None) -> List[Any]:
        """
        Get the values as a list.

        With no `astype`, or with `list`, this is the list that the KeyedList adopted, not
        a copy. If it is a row of a Daf, changing it changes the Daf. With any other
        `astype` it is a new list, and the original is not changed. An empty cell, which is
        NULL in daffodil, is not converted. It stays as the empty string.

        Args:
            astype: How to convert each value. A type such as `int`, a function, or one of
                the names `'int'`, `'str'`, `'float'` and `'bool'`. None does not convert.

        Returns:
            The values.

        Raises:
            ValueError: `astype` is a name that is not supported, or a value cannot be
                converted by `int` or `float`. Another converter raises its own error.

        Examples:
            >>> klist = KeyedList(['a', 'b'], ['1', '2'])
            >>> klist.values()
            ['1', '2']
            >>> klist.values(int)
            [1, 2]
            >>> klist.values('float')
            [1.0, 2.0]
            >>> KeyedList(['a', 'b'], ['1', '']).values(int)
            [1, '']
            >>> klist.values() is klist.values()
            True
            >>> klist.values(int) is klist.values(int)
            False
            >>> klist.values('date')
            Traceback (most recent call last):
                ...
            ValueError: astype not supported: date
        """
        # fast path: return underlying list
        if astype is None or astype is list:
            return self._values

        return _astype_la(self._values, astype)


    def items(self) -> Iterator[Tuple[TKey, Any]]:
        """
        Loop over the (key, value) pairs, as `dict.items()` does.

        This is an iterator, not a view. It can be used once. To keep the pairs, use
        `list(row.items())`.

        Returns:
            An iterator of (key, value) tuples.

        Examples:
            >>> list(KeyedList(['a', 'b'], [1, 2]).items())
            [('a', 1), ('b', 2)]
            >>> pairs = KeyedList(['a'], [1]).items()
            >>> list(pairs), list(pairs)
            ([('a', 1)], [])
        """
        return zip(self.hd, self._values)


    def get(self, key: TKey, default: Any = None) -> Any:
        """
        Get the value for a key, or a default if the key is not found.

        Args:
            key: The key.
            default: What to return if the key is not found.

        Returns:
            The value, or `default`.

        Raises:
            TypeError: The key cannot be hashed, such as a list. Unlike `row[...]`, a list
                of keys is not accepted here.

        Examples:
            >>> klist = KeyedList(['a'], [1])
            >>> klist.get('a'), klist.get('z'), klist.get('z', 0)
            (1, None, 0)
        """
        try:
            return self._values[self.hd[key]]
        except KeyError:
            return default


    def update(self, other: 'KeyedList | Dict[Any, Any]') -> None:
        """
        Set many keys from a dict or from another KeyedList.

        Each key of `other` is assigned as in `row[key] = value`. A key that exists gets
        the new value in the list. A key that is new is added at the end of the keys and of
        the list. If the list is a row of a Daf, that row is then longer than the others,
        so use `update()` only with keys that exist. `other` is not changed.

        Args:
            other: A dict or a KeyedList.

        Examples:
            >>> klist = KeyedList(['a', 'b'], [1, 2])
            >>> klist.update({'b': 20, 'c': 30})
            >>> klist.to_dict()
            {'a': 1, 'b': 20, 'c': 30}
        """
        # this could allow direct updating.
        for key, value in other.items():
            self[key] = value
    
    def to_dict(self) -> Dict[TKey, Any]:
        """
        Make a dict of the keys and values.

        The dict is independent of this KeyedList. Later writes to the row do not change
        it, and changing it does not change the row. The values are not copied deeply, so a
        list or dict held in a cell is the same object in both.

        Returns:
            A new dict.

        Examples:
            >>> klist = KeyedList(['a', 'b'], [1, 2])
            >>> as_dict = klist.to_dict()
            >>> klist['a'] = 99
            >>> as_dict
            {'a': 1, 'b': 2}
        """
        return dict(self.items())
    
    def __repr__(self) -> str:
        """
        Show the keys and values as a dict.

        Returns:
            The text of the dict.

        Examples:
            >>> KeyedList(['a', 'b'], [1, 2])
            {'a': 1, 'b': 2}
        """
        return repr(dict(self.items()))
        
    def __bool__(self) -> bool:
        """ return true if there is something in the values list. """
        return bool(self._values)
        
   
    # @staticmethod
    # def _build_hd(keys: Iterator):
    #     # it is necessary to rebuild hd whenever it is changed.
        
    #     # this is equivalent to:
        
    #     #   {col: idx for idx, col in enumerate(keys)}
        
    #     # but this is substantially faster

    #     return dict(zip(keys, range(len(keys))))
        

    def to_json(self) -> str:
        """
        Make a JSON string that holds the keys and the values.

        The text has the marker `__KeyedList__`, the keys with their positions under `hd`,
        and the `values`. Every value must be one that `json` can write. JSON keys are text,
        so a key that is not a string, such as an int, comes back from `from_json()` as a
        string.

        Returns:
            The JSON text.

        Raises:
            TypeError: A value cannot be written as JSON.

        Examples:
            >>> KeyedList(['a', 'b'], [1, 'x']).to_json()
            '{"__KeyedList__": true, "hd": {"a": 0, "b": 1}, "values": [1, "x"]}'
        """
        # Serialize KeyedList object to a JSON-compatible dictionary
        # NOTE: to_json/from_json appear unused elsewhere in daffodil (Daf.to_json/from_json
        # serialize lol/hd directly and do not call these). Fixed anyway since the risk is low.
        return json.dumps({"__KeyedList__": True, "hd": self.hd.to_dict(), "values": self._values})

    @classmethod
    def from_json(cls, json_str: str) -> 'KeyedList':
        """
        Make a KeyedList from the text that `to_json()` writes.

        Args:
            json_str: The JSON text.

        Returns:
            The new KeyedList. It has its own list of values.

        Raises:
            ValueError: The text is not JSON, or the JSON does not have the `__KeyedList__`
                marker.

        Examples:
            >>> klist = KeyedList(['a', 'b'], [1, 'x'])
            >>> KeyedList.from_json(klist.to_json()).to_dict()
            {'a': 1, 'b': 'x'}
            >>> KeyedList.from_json('{"a": 1}')
            Traceback (most recent call last):
                ...
            ValueError: Invalid JSON string for KeyedList
        """
        # Deserialize JSON string into a KeyedList object
        obj_dict = json.loads(json_str)
        if "__KeyedList__" in obj_dict and obj_dict["__KeyedList__"]:
            return cls(obj_dict.get("hd", {}), obj_dict.get("values", []))
        else:
            raise ValueError("Invalid JSON string for KeyedList")

NULL = ''       # a missing value is the empty string, as in daf.py. Test it with `val is NULL`.

_ASTYPE_BY_NAME: Dict[str, Callable] = {'int': int, 'str': str, 'float': float, 'bool': bool}


def _astype_la(la: T_la, astype: Callable | str | type | None = None) -> T_la:
    """
    Convert each value of a list, and keep a missing value as it is.

    An empty cell, which is NULL in daffodil, is not converted. So `int` of `'1'` and `''` gives
    `1` and `''`, and does not fail. This is the same rule as `daf_utils.astype_la()`, which this
    module cannot import without a circular import. Internal use, by `KeyedList.values()`.

    Args:
        la: The values.
        astype: A type such as `int`, a function, or one of the names `'int'`, `'str'`, `'float'` and
            `'bool'`. None returns the list as it is, not a copy.

    Returns:
        A new list, or `la` itself if `astype` is None.

    Raises:
        ValueError: `astype` is a name that is not supported, or is not callable.

    Examples:
        >>> _astype_la(['1', '', '3'], int)
        [1, '', 3]
        >>> _astype_la(['1.5'], 'float')
        [1.5]
    """
    if astype is None:
        return la

    if isinstance(astype, str):
        convert = _ASTYPE_BY_NAME.get(astype)
        if convert is None:
            raise ValueError(f"astype not supported: {astype}")
    elif callable(astype):
        convert = astype
    else:
        raise ValueError(f"astype not supported: {astype}")

    return [val if val is NULL else convert(val) for val in la]


class KeyedIndex:
    """
    The index of keys that KeyedLists share. Each key maps to the position of its value.

    A KeyedIndex turns a key into the position of its value in a list. A table uses one for
    its column names. Every [KeyedList][daffodil.keyedlist.KeyedList] row of that table can
    point at the same KeyedIndex, so no row has to build its own.

    Keys must be unique. They can be of mixed types, if they are hashable. You can add a key
    at the end with `append()`. You can't delete a key or insert one in the middle. To
    change anything else, build a new KeyedIndex.

    You can create one from:

    - a list or tuple of keys.
    - a dict. Its keys are used, and its values are ignored.
    - a keys view.
    - a KeyedList, which gives a copy of its index.
    - another KeyedIndex. This shares the index with no copy. Appending to one changes both.

    Anything else, such as a generator, is rejected.

    Examples:
        >>> kidx = KeyedIndex(["a", "b", "c"])
        >>> kidx["b"]
        1
        >>> "c" in kidx
        True
        >>> len(kidx)
        3
        >>> kidx.append("d")
        >>> kidx["d"]
        3
        >>> kidx.get("x") is None
        True
        >>> kidx.get("x", -1)
        -1
        >>> kidx.to_dict()
        {'a': 0, 'b': 1, 'c': 2, 'd': 3}

        Keys can be of mixed types, but they must be unique:

        >>> KeyedIndex(["a", 1, (2, 3)])[(2, 3)]
        2
        >>> KeyedIndex(["a", "b", "a"])
        Traceback (most recent call last):
            ...
        ValueError: Duplicate keys not allowed in KeyedIndex
        >>> kidx.append("b")
        Traceback (most recent call last):
            ...
        ValueError: Duplicate key: b
        >>> KeyedIndex(k for k in "ab")
        Traceback (most recent call last):
            ...
        TypeError: Unsupported type for KeyedIndex: generator. Expected list, tuple, dict, or dict_keys.
    """
    """
    KeyedIndex: compiled index over a sequence of UNIQUE keys.

    Semantics:
        - key → integer index (position)
        - keys must be unique (enforced at construction and append)
        - append-only mutation supported
        - no delete / insert-in-middle support

    Supported input types:
        - list
        - tuple
        - dict        (uses dict.keys())
        - dict_keys   (keys view)
        - KeyedList   (uses its own .hd)
        - KeyedIndex  (shares the other index, with no copy. Appending to one changes both.)

    Unsupported:
        - arbitrary iterables (explicit rejection to avoid ambiguity)


    Examples
    --------

    Basic usage
    ~~~~~~~~~~~

    >>> kidx = KeyedIndex(["a", "b", "c"])
    >>> kidx["b"]
    1
    >>> "c" in kidx
    True
    >>> len(kidx)
    3

    Empty initialization
    ~~~~~~~~~~~~~~~~~~~~

    >>> kidx = KeyedIndex()
    >>> bool(kidx)
    False
    >>> list(kidx.keys())
    []

    Append keys
    ~~~~~~~~~~~

    >>> kidx = KeyedIndex(["a", "b"])
    >>> kidx.append("c")
    >>> kidx["c"]
    2

    Duplicate keys (construction)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    >>> KeyedIndex(["a", "b", "a"])
    Traceback (most recent call last):
        ...
    ValueError: Duplicate keys not allowed in KeyedIndex

    Duplicate keys (append)
    ~~~~~~~~~~~~~~~~~~~~~~~

    >>> kidx = KeyedIndex(["a", "b"])
    >>> kidx.append("b")
    Traceback (most recent call last):
        ...
    ValueError: Duplicate key: b

    Mixed key types
    ~~~~~~~~~~~~~~~

    >>> kidx = KeyedIndex(["a", 1, (2, 3)])
    >>> kidx["a"]
    0
    >>> kidx[1]
    1
    >>> kidx[(2, 3)]
    2

    Using dict input
    ~~~~~~~~~~~~~~~~

    >>> kidx = KeyedIndex({"a": 10, "b": 20})
    >>> sorted(kidx.keys())
    ['a', 'b']
    >>> kidx["b"]
    1

    Iteration
    ~~~~~~~~~

    >>> kidx = KeyedIndex(["x", "y", "z"])
    >>> [k for k in kidx]
    ['x', 'y', 'z']

    to_dict and repr
    ~~~~~~~~~~~~~~~~

    >>> kidx = KeyedIndex(["a", "b"])
    >>> kidx.to_dict()
    {'a': 0, 'b': 1}
    >>> kidx
    {'a': 0, 'b': 1}

    get with default
    ~~~~~~~~~~~~~~~~

    >>> kidx = KeyedIndex(["a", "b"])
    >>> kidx.get("b")
    1
    >>> kidx.get("x") is None
    True
    >>> kidx.get("x", -1)
    -1

    Notes
    -----

    - Keys must be hashable and unique.
    - Keys may be of mixed types (e.g., str, int, tuple).
    - The index reflects the position at insertion time.
    - Structural mutations outside of append (e.g., reordering source data)
      require rebuilding the KeyedIndex.

    """

    __slots__ = ("_index",)

    _index: Dict[TKey, int]

    def __init__(
        self,
        keys: 'List[TKey] | Tuple[TKey, ...] | Dict[TKey, Any] | KeysView[TKey] | KeyedList | KeyedIndex | None' = None,
    ) -> None:
        # normalize input → list
        if keys is None:
            keys_list: List[TKey] = []

        elif isinstance(keys, KeyedIndex):
            self._index = keys._index
            return

        elif isinstance(keys, KeyedList):
            self._index = dict(keys.hd)
            return

        elif isinstance(keys, list):
            keys_list = keys

        elif isinstance(keys, tuple):
            keys_list = list(keys)

        elif isinstance(keys, dict):
            keys_list = list(keys.keys())

        elif isinstance(keys, KeysView):
            keys_list = list(keys)

        else:
            raise TypeError(
                f"Unsupported type for KeyedIndex: {type(keys).__name__}. "
                "Expected list, tuple, dict, or dict_keys."
            )

        # build index (single pass, preserves order)
        index: Dict[TKey, int] = dict(zip(keys_list, range(len(keys_list))))

        # enforce uniqueness
        if len(index) != len(keys_list):
            raise ValueError("Duplicate keys not allowed in KeyedIndex")

        self._index = index

    # --- core lookup ---

    def __getitem__(self, key: TKey) -> int:
        """
        Get the position of a key, as `kidx[key]`.

        Args:
            key: The key.

        Returns:
            The position of its value in the list.

        Raises:
            KeyError: The key is not found.
        """
        return self._index[key]

    def __contains__(self, key: object) -> bool:
        """
        Test whether a key is in the index, as `key in kidx`.

        Args:
            key: The key to look for.

        Returns:
            True if the key is found.
        """
        return key in self._index

    def get(self, key: TKey, default: int | None = None) -> int | None:
        """
        Get the position of a key, or a default if the key is not found.

        Args:
            key: The key.
            default: What to return if the key is not found.

        Returns:
            The position, or `default`.

        Examples:
            >>> kidx = KeyedIndex(['a', 'b'])
            >>> kidx.get('b'), kidx.get('z'), kidx.get('z', -1)
            (1, None, -1)
        """
        return self._index.get(key, default)

    def index(self, key: TKey) -> int:
        """
        Get the position of a key. This is the same as `kidx[key]`.

        Args:
            key: The key.

        Returns:
            The position of its value in the list.

        Raises:
            KeyError: The key is not found.

        Examples:
            >>> KeyedIndex(['a', 'b']).index('b')
            1
        """
        return self._index[key]

    # --- size / truth ---

    def __len__(self) -> int:
        """
        Get the number of keys.

        Returns:
            The number of keys.
        """
        return len(self._index)

    def __bool__(self) -> bool:
        """
        Test whether the index has any keys.

        Returns:
            False for an empty index.
        """
        return bool(self._index)

    def __eq__(self, other: object) -> bool:
        """
        Compare with another KeyedIndex. They are equal if they have the same keys at the same positions.

        Args:
            other: The object to compare with.

        Returns:
            True or False for a KeyedIndex. For any other type, `NotImplemented`, so Python tries the other side.
        """
        if isinstance(other, KeyedIndex):
            return self._index == other._index
            
        return NotImplemented
    
    # --- key access ---

    def keys(self) -> KeysView[TKey]:
        """
        Get the keys, in the order of their positions.

        Returns:
            A keys view of the index itself, not a copy.

        Examples:
            >>> KeyedIndex(['a', 'b']).keys()
            dict_keys(['a', 'b'])
        """
        return self._index.keys()

    def __iter__(self) -> Iterator[TKey]:
        """
        Loop over the keys, in the order of their positions.

        Returns:
            An iterator of the keys.
        """
        return iter(self._index)

    # --- mutation (append only) ---

    def append(self, key: TKey) -> None:
        """
        Add a key at the end. Its position is the number of keys before it.

        This changes every KeyedList and KeyedIndex that shares this index. A KeyedList
        that adds a key through `row[key] = value` copies a shared index first, so it does not.

        Args:
            key: The new key.

        Raises:
            ValueError: The key is already in the index.

        Examples:
            >>> kidx = KeyedIndex(['a', 'b'])
            >>> kidx.append('c')
            >>> kidx['c']
            2
            >>> kidx.append('a')
            Traceback (most recent call last):
                ...
            ValueError: Duplicate key: a
        """
        if key in self._index:
            raise ValueError(f"Duplicate key: {key}")
        self._index[key] = len(self._index)

    # --- utilities ---

    def to_dict(self) -> Dict[TKey, int]:
        """
        Make a dict of each key and its position.

        Returns:
            A new dict. Changing it does not change the index.

        Examples:
            >>> KeyedIndex(['a', 'b']).to_dict()
            {'a': 0, 'b': 1}
        """
        return dict(self._index)

    def __repr__(self) -> str:
        """
        Show the keys and their positions as a dict.

        Returns:
            The text of the dict.
        """
        return repr(self._index)

