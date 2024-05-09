# keyedlist.py

from typing import List, Dict, Any, Tuple, Optional, Union #


class KeyedList:
    """
    KeyedList is a custom data structure in Python that combines the functionality of a dictionary and a list,
    optimized for efficient indexing and manipulation of data. It maintains an index for fast key-based
    access to values stored in a list. This is similar to a conventional dictionary, but the list items are
    not distributed to each item in the dict, but can be an existing list, used without copying.
    
    The keys are implemented as a dictionary of indexes, i.e. {'key0': 0, 'key1', 1, ... 'keyn'} where the keys
    are just examples here. For convenience, we call this structure a dict of index or 'dex'.
    
    For data that is already stored as a list, the list can be adopted without copying. An important attribute 
    of this approach is that the parent list is modified if values in the keyedlist is modified. The code should 
    make a copy if the values in the source list need to remain unaltered, or convert to a conventional dict 
    which will inherently make a copy, using .to_dict()
    
    Similarly, if the dex already exists, it can be reused on many instances of keyed list. Further, the dex can 
    be used on a list-of-list structure as the column indexes of all rows. If related to such an array, then the
    keys are frequently called 'cols'. See the daffodil package for a full implementation of such an array.

    Usage:
        - KeyedList can be initialized from either a list of keys and values, a dictionary, or an existing dex and
          list of values.
          
        - It supports standard dictionary operations such as __getitem__, __setitem__, __delitem__, __len__,
          __iter__, keys, values, items, get, update, and conversion to a conventional dictionary using .to_dict().
          
        - KeyedList instances provide fast key-based access to values similar to dictionaries, while also allowing
          list-like operations for efficient value manipulation.
          
        - Most importantly, creation of a KeyedList instance is much faster than creating a conventional dict, because
          the list can be adopted as a reference without copying.

    Initialization:
        - KeyedList(keys_iter, values_list) -> KeyedList: Initialize KeyedList from a list of keys and values_list. 
            Like dict(keys, values_list) but the values_list is adopted without copying, if it is a list.
            
        - KeyedList(full_dict) -> KeyedList: Initialize DexList from a full_dict, a conventional dictionary.
            This method of initialization will copy the values and is relatively expensive.
            
        - KeyedList(dex, values_list) -> KeyedList: Initialize KeyedList from an existing dex and values_list.
            This is very fast because no copying occurs. 
    
    Operation
        - A KeyedList instance acts just like a conventional dictionary, but it can be much less expensive to use,
            becuase the dex portion can be reused, and the values can be adopted without copying.
            
        - If a klist is created from an associated 'record' in a list-of-list (lol) with an associated dex header, then
            the klist can be created by reference to the dex header and a list in the lol array. Changes to the 
            klist item will update the lol array, because the list item is actually the same list a the one in the array.
            This behavior is not possible with dictionaries.
            
        - As a result, a KeyedList instance is more 'dangerous' for beginner Python programmers because they are constructed
            always by copying in the keys and values and do not maintain the connection to the prior source of the list
            portion. 
            
        - similar to a dictionary, a KeyedList can provide the keys and values as iterators or lists. The difference is that
            a KeyedList allows assignment to the dex and the list.
            
            klist = KeyedList({'a': 5, 'b' 8})
            print(klist)            # output: {'a': 5, 'b' 8}
            print(klist.keys())     # output: dict_keys(['a', 'b'])
            print(klist.values())   # output: [5, 8]
            alist = klist.values    # grab a reference to the values.
            alist[1] = 10           # assign a value to the list.
            print(alist)            # [5, 10]
            print(klist)            # {'a': 5, 'b' 10}
            
            klist.values = [2,3]
            print(klist)            # {'a': 2, 'b' 3}
            print(alist)            # [2,3]


        Example2:
            keys = ['a', 'b', 'c']
            values = [1, 2, 3]
            klist = KeyedList(keys, values)
            print(klist['a'])  # Output: 1
            klist['b'] = 5
            print(klist)  # Output: {'a': 1, 'b': 5, 'c': 3}
            
        See also:
            https://peps.python.org/pep-0412/#alternative-implementation
            
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
        if key not in self.dex:
            self.dex[key] = len(self.dex)
            self.values.append(value)
        else:    
            self.values[self.dex[key]] = value

    
    def __delitem__(self, key):
        index = self.dex.pop(key)
        del self.values[index]
        # Update indices in dex for keys after the deleted key
        # probably faster to rebuild it.
        # for k, v in self.dex.items():
            # if v > index:
                # self.dex[k] -= 1
                
        self.dex = dict(zip(self.dex.keys(), range(len(self.dex.keys()))))        
    
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
