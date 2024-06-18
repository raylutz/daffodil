# keyedlist.py

from typing import List, Dict, Any, Optional, Union, Iterator #, Tuple

import json


class KeyedList:
    """
    KeyedList is a custom data structure in Python that combines the functionality of a dictionary and a list,
    optimized for efficient indexing and manipulation of data. 
    
    It maintains an index for fast key-based access to values stored in a list. 
    This is similar to a conventional dictionary, but the list items are
    not distributed to each item in the dict, but can be an existing list, used without copying.
    
    hd (header dict)
    
    The keys are implemented as a dictionary of indexes, i.e. {'key0': 0, 'key1', 1, ... 'keyn': n} where the keys
    are just examples here. For convenience, we call this structure a "header dict" or 'hd'. 
    To create this the following can be used:
    
        hd = dict(zip(keys, range(len(keys))))
        
    Which is more performant and equivalent to:
    
        hd = {col: idx for idx, col in enumerate(keys)
        
    But this likely can be improved if a concise standard library function is created, since the current nature
    of a dict is that it is ordered, and thus has an implied index.
    
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
            
        - KeyedList(hd, values_list) -> KeyedList: Initialize KeyedList from an existing hd and values_list.
            This is very fast because no copying occurs. 
    
    Operation
        - A KeyedList instance acts just like a conventional dictionary, but it can be much less expensive to use,
            because the hd portion can be reused, and the values can be adopted without copying. However, there is 
            a slight penalty in access because of the additional indirection.
            
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
            print(klist)            # output: {'a': 5, 'b' 8}
            print(klist.keys())     # output: dict_keys(['a', 'b'])
            print(klist.values())   # output: [5, 8]
            alist = klist.values    # grab a reference to the values (no copying)
            alist[1] = 10           # assign a value to the list.
            print(alist)            # output: [5, 10]
            print(klist)            # output: {'a': 5, 'b' 10}  <-- note that klist changes too!
            
            klist.values = [2,3]    # assign new values to the list portion. Assignment like this is supported for dicts.
            print(klist)            # output: {'a': 2, 'b' 3}
            print(alist)            # output: [2, 3]    # note that the list that is a reference to the list portion changes.


        Example2:
            keys = ['a', 'b', 'c']
            values = [1, 2, 3]
            klist = KeyedList(keys, values)     # initialize like you would a dict using dict(zip(keys, values))
            print(klist)                        # Output: {'a': 1, 'b': 2, 'c': 3}
            print(klist['a'])                   # Output: 1
            klist['b'] = 5                      # overwrite a value
            print(klist)                        # Output: {'a': 1, 'b': 5, 'c': 3}
            
        See also:
            https://peps.python.org/pep-0412/#alternative-implementation
            
    """

    def __init__(self, 
            arg1: Optional[Union[Dict[Any, Any], List[Any], 'KeyedList']] = None, 
            arg2: Optional[List[Any]] = None,
            ):
            
        if isinstance(arg1, dict):
            if arg2 is None:
                # Case 1: from_dict
                self.hd = type(self)._build_hd(arg1.keys())
                self.values = list(arg1.values())
                return
                
            if isinstance(arg2, list):
                # Case 2: from hd plus values
                if len(arg1) != len(arg2):
                    raise ValueError("keys and values must have the same length")
                self.hd = arg1
                self.values = arg2
                return
            
        if isinstance(arg1, list) and isinstance(arg2, list):
            # Case 3: from list of keys and values
            if len(arg1) != len(arg2):
                raise ValueError("keys and values must have the same length")
            self.hd = type(self)._build_hd(arg1)
            self.values = arg2
            return
            
        if isinstance(arg1, type(self)) and arg2 is None:
            self.hd = arg1.hd
            self.values = arg1.values
            return
            
        if arg1 is None and arg2 is None:
            # return a functional empty keyedlist, like {}
            self.hd = {}
            self.values = []
            return
        
        raise ValueError("Must provide either a dict, keys and values, hd and list, or KeyedList")
    
    def __getitem__(self, key):
        return self.values[self.hd[key]]
    
    def __setitem__(self, key, value):
        if key not in self.hd:
            # extend hd
            self.hd[key] = len(self.hd)
            
            self.values.append(value)
        else:    
            self.values[self.hd[key]] = value

    
    def __delitem__(self, key):
        index = self.hd.pop(key)
        del self.values[index]
                
        # it is necessary to rebuild hd whenever it is changed.
        self.hd = type(self)._build_hd(self.hd.keys())        
    
    def __len__(self):
        return len(self.values)
    
    def __iter__(self):
        return iter(self.hd)
    
    def keys(self):
        return self.hd.keys()
    
    def values(self):
        return self.values
    
    def items(self):
        return ((key, self.values[idx]) for key, idx in self.hd.items())
    
    def get(self, key, default=None):
        try:
            return self.values[self.hd[key]]
        except Exception:
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
        return bool(self.values)
        
   
    @staticmethod
    def _build_hd(keys: Iterator):
        # it is necessary to rebuild hd whenever it is changed.
        
        # this is equivalent to:
        
        #   {col: idx for idx, col in enumerate(keys)}
        
        # but this is substantially faster

        return dict(zip(keys, range(len(keys))))
        

    def to_json(self):
        # Serialize KeyedList object to a JSON-compatible dictionary
        return json.dumps({"__KeyedList__": True, "hd": self.hd, "values": self.values})

    @classmethod
    def from_json(cls, json_str):
        # Deserialize JSON string into a KeyedList object
        obj_dict = json.loads(json_str)
        if "__KeyedList__" in obj_dict and obj_dict["__KeyedList__"]:
            return cls(hd=obj_dict.get("hd", {}), values=obj_dict.get("values", []))
        else:
            raise ValueError("Invalid JSON string for KeyedList")
        
        
        
class KeyedListEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, KeyedList):
            return {"__KeyedList__": True, "hd": obj.hd, "values": obj.values}
        return super().default(obj)
        