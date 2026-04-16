# keyedlist.py

from typing import List, Dict, Any, Optional, Union, Iterable, Iterator, \
                    Callable, KeysView, Tuple

from collections.abc import Hashable                    

TKey = Hashable
T_la = List[Any]

import json


class KeyedList:
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

    def __init__(self, 
            arg1: Optional[Union[Dict[Any, Any], List[Any], 'KeyedList', 'KeyedIndex']] = None, 
            arg2: Optional[List[Any]] = None,
            default: Optional[Union[int, str, float]] = None,
            ):
            
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
            
        elif isinstance(arg1, KeyedIndex) and isinstance(arg2, list):
            # Case: hd + row (critical for reference semantics)
            if len(arg1) != len(arg2):
                raise ValueError("hd and values must have the same length")
            self.hd = arg1              # reuse, DO NOT rebuild
            self._values = arg2         # direct reference
            return

        elif arg1 is None and arg2 is None:
            # Case 6, Empty - return a functional empty keyedlist, like {}
            self.hd = KeyedIndex()
            self._values = []
            return
        
        breakpoint() #perm logic error
        pass
        raise ValueError("Must provide either a dict, keys and values, hd and list, or KeyedList")
    
    def __getitem__(self, key):
        """
            indexing can use a scalar key or a list of keys, which returns a list.
        """
        
        if isinstance(key, list):
            return [self._values[self.hd[onekey]] for onekey in key if onekey in self.hd]
            
        elif isinstance(key, Hashable):
            return self._values[self.hd[key]]

        else:
            raise ValueError
    
    def __setitem__(self, key, value):
        if key not in self.hd:
            # extend hd
            self.hd.append(key)
            # self.hd[key] = len(self.hd)
            
            self._values.append(value)
        else:    
            self._values[self.hd[key]] = value

    
    def __delitem__(self, key):
        # index = self.hd.pop(key)
        # del self._values[index]
                
        # # it is necessary to rebuild hd whenever it is changed.
        # self.hd = type(self)._build_hd(self.hd.keys())        
        index = self.hd[key]
        del self._values[index]

        new_keys = list(self.hd)
        del new_keys[index]

        self.hd = KeyedIndex(new_keys)


    def __len__(self):
        return len(self._values)
    
    def __iter__(self):
        return iter(self.hd)

    def keys(self):
        return self.hd.keys()

    def set_values(self, new_values: List[Any]) -> None:
        if not isinstance(new_values, list):
            raise TypeError

        if len(new_values) != len(self.hd):
            raise ValueError("values length must match keys")

        self._values = new_values


    def values(self, astype: Optional[Union[Callable, str, type]] = None) -> List[Any]:
        # fast path: return underlying list
        if astype is None or astype is list:
            return self._values

        return astype_la(self._values, astype)


    def items(self):
        return zip(self.hd, self._values)


    def get(self, key, default=None):
        try:
            return self._values[self.hd[key]]
        except KeyError:
            return default


    def update(self, other):
        # this could allow direct updating.
        for key, value in other.items():
            self[key] = value
    
    def to_dict(self):
        return dict(self.items())
    
    def __repr__(self):
        return repr(dict(self.items()))
        
    def __bool__(self):
        """ return true if there is something in the values list. """
        return bool(self._values)
        
   
    # @staticmethod
    # def _build_hd(keys: Iterator):
    #     # it is necessary to rebuild hd whenever it is changed.
        
    #     # this is equivalent to:
        
    #     #   {col: idx for idx, col in enumerate(keys)}
        
    #     # but this is substantially faster

    #     return dict(zip(keys, range(len(keys))))
        

    def to_json(self):
        # Serialize KeyedList object to a JSON-compatible dictionary
        return json.dumps({"__KeyedList__": True, "hd": self.hd, "values": self._values})

    @classmethod
    def from_json(cls, json_str) -> 'KeyedList':
        # Deserialize JSON string into a KeyedList object
        obj_dict = json.loads(json_str)
        if "__KeyedList__" in obj_dict and obj_dict["__KeyedList__"]:
            return cls(hd=obj_dict.get("hd", {}), values=obj_dict.get("values", []))
        else:
            raise ValueError("Invalid JSON string for KeyedList")

def astype_la(la: T_la, astype: Optional[Union[Callable, str, type]]=None) -> T_la:
    """ fix the type according to astype spec if it is not None 
            this function current duplicated in daf_utils
    """        

    if astype is None:
        return la
        
    if callable(astype) and not isinstance(astype, type): 
        return [astype(val) for val in la]
        
    # Type provided (e.g., int, str, float, bool)
    if isinstance(astype, type):
        return [astype(val) for val in la]

    if isinstance(astype, str):
        if astype == 'int':
            return [int(val) for val in la]
        elif astype == 'str':
            return [str(val) for val in la]
        elif astype == 'float':
            return [float(val) for val in la]
        elif astype == 'bool':
            return [bool(val) for val in la]
        else:
            raise ValueError (f"astype not supported: {astype}")
            
    raise ValueError (f"astype not supported: {astype}")

        
        
class KeyedListEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, KeyedList):
            return {"__KeyedList__": True, "hd": obj.hd, "values": obj._values}
        return super().default(obj)



class KeyedIndex:
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
    >>> kidx.get("x")
    None
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
        keys: Optional[
            Union[
                List[TKey],
                Tuple[TKey, ...],
                Dict[TKey, Any],
                KeysView[TKey],
            ]
        ] = None,
    ) -> None:
        # normalize input → list
        if keys is None:
            keys_list: List[TKey] = []

        elif isinstance(keys, KeyedIndex):
            self._index = keys._index
            return

        if isinstance(keys, KeyedList):
            self._index = keys.keys()
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
        return self._index[key]

    def __contains__(self, key: object) -> bool:
        return key in self._index

    def get(self, key: TKey, default: Optional[int] = None) -> Optional[int]:
        return self._index.get(key, default)

    def index(self, key: TKey) -> int:
        return self._index[key]

    # --- size / truth ---

    def __len__(self) -> int:
        return len(self._index)

    def __bool__(self) -> bool:
        return bool(self._index)

    def __eq__(self, other):
        if isinstance(other, KeyedIndex):
            return self._index == other._index
            
        return NotImplemented
    
    # --- key access ---

    def keys(self) -> KeysView[TKey]:
        return self._index.keys()

    def __iter__(self) -> Iterator[TKey]:
        return iter(self._index)

    # --- mutation (append only) ---

    def append(self, key: TKey) -> None:
        if key in self._index:
            raise ValueError(f"Duplicate key: {key}")
        self._index[key] = len(self._index)

    # --- utilities ---

    def to_dict(self) -> Dict[TKey, int]:
        return dict(self._index)

    def __repr__(self) -> str:
        return repr(self._index)

