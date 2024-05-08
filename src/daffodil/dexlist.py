# dexlist.py

from typing import List, Dict, Any, Tuple, Optional, Union #


class DexList:
    """
    DexList is a custom data structure in Python that combines the functionality of a dictionary and a list,
    optimized for efficient indexing and manipulation of data. It maintains an index (dex) for fast key-based
    access to values stored in a list.
    
    For data that is already stored as a list, a dexlist can be produced instead of a conventional dict, and 
    the list can be adopted without copying. An important attribute of this approach is that the parent list is
    modified if values in the dexlist is modified. The code should make a copy if the values in the source list
    need to remain unaltered, or convert to a conventional dict which will inherently make a copy.

    Usage:
        - DexList can be initialized from either a list of keys and values, a dictionary, or an existing dex and
          list of values.
        - It supports standard dictionary operations such as __getitem__, __setitem__, __delitem__, __len__,
          __iter__, keys, values, items, get, update, and conversion to a conventional dictionary using to_dict().
        - DexList instances provide fast key-based access to values similar to dictionaries, while also allowing
          list-like operations for efficient value manipulation.

    Initialization:
        - DexList(keys: List, values: List) -> DexList: Initialize DexList from a list of keys and values.
        - DexList(da: Dict) -> DexList: Initialize DexList from a dictionary.
        - DexList(dex: Dict, values: List) -> DexList: Initialize DexList from an existing index (dex) and a list
          of values.

    Example:
        keys = ['a', 'b', 'c']
        values = [1, 2, 3]
        dexlist = DexList(keys, values)
        print(dexlist['a'])  # Output: 1
        dexlist['b'] = 5
        print(dexlist)  # Output: {'a': 1, 'b': 5, 'c': 3}
    """

    def __init__(self, arg1: Union[Dict[Any, Any], List[Any]], arg2: Optional[List[Any]] = None):
        if isinstance(arg1, dict):
            if arg2 is None:
                # Case 1: from_dict
                self.dex = dict(zip(arg1.keys(), range(len(arg1.keys()))))
                self.values = list(arg1.values())
                return
                
            if isinstance(arg2, list):
                # Case 2: from_dex plus values
                if len(arg1) != len(arg2):
                    raise ValueError("keys and values must have the same length")
                self.dex = arg1
                self.values = arg2
                return
            
        if isinstance(arg1, list) and isinstance(arg2, list):
            # Case 3: from list of keys and values
            if len(arg1) != len(arg2):
                raise ValueError("keys and values must have the same length")
            self.dex = dict(zip(arg1, range(len(arg1))))
            self.values = arg2
            return
        
        raise ValueError("Must provide either a dict, keys and values, or dex and list")
    
    def __getitem__(self, key):
        return self.values[self.dex[key]]
    
    def __setitem__(self, key, value):
        self.values[self.dex[key]] = value
    
    def __delitem__(self, key):
        index = self.dex.pop(key)
        del self.values[index]
        # Update indices in dex for keys after the deleted key
        for k, v in self.dex.items():
            if v > index:
                self.dex[k] -= 1
    
    def __len__(self):
        return len(self.dex)
    
    def __iter__(self):
        return iter(self.dex)
    
    def keys(self):
        return self.dex.keys()
    
    def values(self):
        return self.values
    
    def items(self):
        return ((key, self.values[index]) for key, index in self.dex.items())
    
    def get(self, key, default=None):
        return self.values[self.dex.get(key, default)]
    
    def update(self, other):
        for key, value in other.items():
            self[key] = value
    
    def to_dict(self):
        return dict(self.items())
    
    def __repr__(self):
        return repr(dict(self.items()))
        
    def __bool__(self):
        return bool(self.values)
