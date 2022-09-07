# Index objects

## Index

### Properties

#### `pd.Index.name`


- <code><apihead>pandas.Index.<apiname>name</apiname></apihead></code>
<br><br>

    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.name
    
    >>> I = pd.Index([1,2,3], name = "hello world")
    >>> f(I)
    "hello world"
    ```
    
#### `pd.Index.names`



- <code><apihead>pandas.Index.<apiname>names</apiname></apihead></code>
<br><br>

    !!! Important
        Bodo returns a tuple instead of a FrozenList.


    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.names
    
    >>> I = pd.MultiIndex.from_product([[1, 2], ["A", "B"]], names=["C1", "C2"])
    >>> f(I)
    ('C1', 'C2')
    ```

    
#### `pd.Index.shape`



- <code><apihead>pandas.Index.<apiname>shape</apiname></apihead></code>
<br><br>
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.shape
    
    >>> I = pd.Index([1,2,3])
    >>> f(I)
    (3,)
    ```


#### `pd.Index.size`



- <code><apihead>pandas.Index.<apiname>size</apiname></apihead></code>
<br><br>

    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.size
    
    >>> I = pd.Index([1,7,8,6])
    >>> f(I)
    4
    ```


#### `pd.Index.empty`



- <code><apihead>pandas.Index.<apiname>empty</apiname></apihead></code>
<br><br>

    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.empty
    
    >>> I = pd.Index(["A", "B", "C"])
    >>> f(I)
    False
    ```


#### `pd.Index.is_monotonic_increasing`


- <code><apihead>pandas.Index.<apiname>is_monotonic_increasing</apiname></apihead> and <apihead>pandas.<apiname>Index.is_monotonic</apiname></apihead></code>
<br><br>

    ***Unsupported Index Types***
    
      - StringIndex
      - BinaryIndex
      - IntervalIndex
      - CategoricalIndex
      - PeriodIndex
      - MultiIndex
    
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_monotonic_increasing
    
    >>> I = pd.Index([1,2,3])
    >>> f(I)
    True
    
    >>> @bodo.jit
    ... def g(I):
    ...   return I.is_monotonic
    
    >>> I = pd.Index(1,2,3])
    >>> g(I)
    True
    ```

#### `pd.Index.is_monotonic_decreasing`



- <code><apihead>pandas.Index.<apiname>is_monotonic_decreasing</apiname></apihead></code>
<br><br>

    ***Unsupported Index Types***
    
      - StringIndex
      - BinaryIndex
      - IntervalIndex
      - CategoricalIndex
      - PeriodIndex
      - MultiIndex
    
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_monotonic_decreasing
    
    >>> I = pd.Index([1,2,3])
    >>> f(I)
    False
    ```


#### `pd.Index.values`


- <code><apihead>pandas.Index.<apiname>values</apiname></apihead></code>
<br><br>

    ***Unsupported Index Types***
    
      - MultiIndex
      - IntervalIndex
    
    ***Example Usage***
    
    ```py
    
    >>> @bodo.jit
    ... def f(I):
    ...   return I.values
    
    >>> I = pd.Index([1,2,3])
    >>> f(I)
    [1 2 3]
    ```
    
    
#### `pd.Index.nbytes`


- <code><apihead>pandas.Index.<apiname>nbytes</apiname></apihead></code>
<br><br>

    ***Unsupported Index Types***
    
      - MultiIndex
      - IntervalIndex
    
    !!! important
        Currently, Bodo upcasts all numeric index data types to 64 bitwidth.
    
    ***Example Usage***
    
    ```py
    
    >>> @bodo.jit
    ... def f(I):
    ...   return I.nbytes
    
    >>> I1 = pd.Index([1,2,3,4,5,6], dtype = np.int64)
    >>> f(I1)
    48
    >>> I2 = pd.Index([1,2,3], dtype = np.int64)
    >>> f(I2)
    24
    >>> I3 = pd.Index([1,2,3], dtype = np.int32)
    >>> f(I3)
    24
    ```

#### `pd.Index.ndim`



- <code><apihead>pandas.Index.<apiname>ndim</apiname></apihead></code>
<br><br>

    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.ndim
    
    >>> I = pd.Index([1,2,3,4])
    >>> f(I)
    1
    ```    


#### `pd.Index.nlevels`



- <code><apihead>pandas.Index.<apiname>nlevels</apiname></apihead></code>
<br><br>

    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.nlevels
    
    >>> I = pd.MultiIndex.from_arrays([[1, 2, 3, 4],["A", "A", "B", "B"]])
    >>> f(I)
    2
    ```


 #### `pd.Index.dtype`



- <code><apihead>pandas.Index.<apiname>dtype</apiname></apihead></code>
<br><br>

    ***Unsupported Index Types***
    
      - PeriodIndex
      - IntervalIndex

    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.dtype
    
    >>> I = pd.Index([1,2,3,4])
    >>> f(I)
    dtype('int64')
    ``` 


 #### `pd.Index.inferred_type`



- <code><apihead>pandas.Index.<apiname>inferred_type</apiname></apihead></code>
<br><br>

    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.dtype
    
    >>> I = pd.Index(["A", "E", "I", "O", "U"])
    >>> f(I)
    'string'
    ```  


 #### `pd.Index.is_all_dates`



- <code><apihead>pandas.Index.<apiname>is_all_dates</apiname></apihead></code>
<br><br>

    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_all_dates
    
    >>> I = pd.date_range("2018-01-01", "2018-01-06")
    >>> f(I)
    True
    ```      


 #### `pd.Index.T`



- <code><apihead>pandas.Index.<apiname>T</apiname></apihead></code>
<br><br>

    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.T
    
    >>> I = pd.Index(["A", "E", "I", "O", "U"])
    >>> f(I)
    Index(["A", "E", "I", "O", "U"], dtype='object')
    ```  

### Type information

#### `pd.Index.is_numeric`


- <code><apihead>pandas.Index.<apiname>is_numeric</apiname>()</apihead></code>
<br><br>    
    ***Example Usage***
    
    ```py
    
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_numeric()
    
    >>> I = pd.Index([1, 2, 3])
    >>> f(I)
    True
    ```

#### `pd.Index.is_integer`


- <code><apihead>pandas.Index.<apiname>is_integer</apiname>()</apihead></code>
<br><br>    
    ***Example Usage***
    
    ```py
    
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_integer()
    
    >>> I = pd.Index([1, 2, 3])
    >>> f(I)
    True
    ```

#### `pd.Index.is_floating`


- <code><apihead>pandas.Index.<apiname>is_floating</apiname>()</apihead></code>
<br><br>    
    ***Example Usage***
    
    ```py
    
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_floating()
    
    >>> I = pd.Index([1, 2, 3])
    >>> f(I)
    False
    ```

#### `pd.Index.is_boolean`


- <code><apihead>pandas.Index.<apiname>is_boolean</apiname>()</apihead></code>
<br><br>    
    ***Example Usage***
    
    ```py
    
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_boolean()
    
    >>> I = pd.Index([1, 2, 3])
    >>> f(I)
    False
    ```

#### `pd.Index.is_categorical`


- <code><apihead>pandas.Index.<apiname>is_categorical</apiname>()</apihead></code>
<br><br>    
    ***Example Usage***
    
    ```py
    
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_categorical()
    
    >>> I = pd.Index([1, 2, 3])
    >>> f(I)
    False
    ```

#### `pd.Index.is_interval`


- <code><apihead>pandas.Index.<apiname>is_interval</apiname>()</apihead></code>
<br><br>    
    ***Example Usage***
    
    ```py
    
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_interval()
    
    >>> I = pd.Index([1, 2, 3])
    >>> f(I)
    False
    ```

#### `pd.Index.is_object`


- <code><apihead>pandas.Index.<apiname>is_object</apiname>()</apihead></code>
<br><br>
    
    !!! important
        Currently, Bodo diverges from the Pandas API for Indices of boolean
        values. Bodo always returns True, whereas Pandas returns False if
        the index was constructed from a pd.array of booleans.
    
    ***Example Usage***
    
    ```py
    
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_object()
    
    >>> I = pd.Index([1, 2, 3])
    >>> f(I)
    False
    ```

### Modifications and computations

#### `pd.Index.copy`



- <code><apihead>pandas.Index.<apiname>copy</apiname>(name=None, deep=False, dtype=None, names=None)</apihead></code>
<br><br>
    ***Unsupported Index Types***
    
    - MultiIndex
    - IntervalIndex
    
    ***Supported arguments***
 
    - `name`
    
    ***Example Usage***
    
    ```py
    
    >>> @bodo.jit
    ... def f(I):
    ...   return I.copy(name="new_name")
    
    >>> I = pd.Index([1,2,3], name = "origial_name")
    >>> f(I)
    Int64Index([1, 2, 3], dtype='int64', name='new_name')
    ```
    
    
#### `pd.Index.get_loc`


- <code><apihead>pandas.Index.<apiname>get_loc</apiname>(key, method=None, tolerance=None)</apihead></code>
<br><br>

    !!! note
        Should be about as fast as standard python, maybe slightly slower.
    
    ***Unsupported Index Types***
    
    - CategoricalIndex
    - MultiIndex
    - IntervalIndex
    
    ***Supported Arguments***

     - `key`: must be of same type as the index

    !!! important
    
        - Only works for index with unique values (scalar return).
        - Only works with replicated Index


    ***Example Usage***
    
    ```py
       
    >>> @bodo.jit
    ... def f(I):
    ...   return I.get_loc(2)
    
    >>> I = pd.Index([1,2,3])
    >>> f(I)
    1
    ```

#### `pd.Index.take`


- <code><apihead>pandas.Index.<apiname>take</apiname>(indices, axis=0, allow_fill=True, fill_value=None, **kwargs)</apihead></code>
<br><br>

    ***Supported Arguments***
    
    - `indices`:  can be boolean Array like, integer Array like, or slice
    
    ***Unsupported Index Types***
    
    - MultiIndex
    - IntervalIndex
    
    !!! important
         Bodo **Does Not** support `kwargs`, even for compatibility.

#### `pd.Index.min`


- <code><apihead>pandas.Index.<apiname>min</apiname>(axis=None, skipna=True, <em>args, </em>*kwargs)</apihead></code>
<br><br>

    ***Supported Arguments***: None
    
    ***Supported Index Types***
    
    - NumericIndex
    - RangeIndex
    - CategoricalIndex
    - TimedeltaIndex
    - DatetimeIndex
    
    !!! important
    
        - Bodo **Does Not** support `args` and `kwargs`, even for compatibility.
        - For DatetimeIndex, will throw an error if all values in the index are null.

    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.min()

    >>> I = pd.Index(pd.date_range(start="2018-04-24", end="2018-04-25", periods=5))
    >>> f(I)
    2018-04-24 00:00:00
    ```
    
    
    
#### `pd.Index.max`


- <code><apihead>pandas.Index.<apiname>max</apiname>(axis=None, skipna=True, <em>args, </em>*kwargs)</apihead></code>
<br><br>

    ***Supported Arguments***: None
    
    ***Supported Index Types***
    
    - NumericIndex
    - RangeIndex
    - CategoricalIndex
    - TimedeltaIndex
    - DatetimeIndex
    
    !!! important
        - Bodo **Does Not** support `args` and `kwargs`, even for compatibility.
        - For DatetimeIndex, will throw an error if all values in the index are null.


    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.min()
    
    >>> I = pd.Index(pd.date_range(start="2018-04-24", end="2018-04-25", periods=5))
    >>> f(I)
    2018-04-25 00:00:00
    ```

#### `pd.Index.rename`


- <code><apihead>pandas.Index.<apiname>rename</apiname>(name, inplace=False)</apihead></code>
<br><br>

    ***Supported Arguments***
    
    - `name`: label or list of labels
    
    ***Unsupported Index Types***
    
    - MultiIndex
    
    ***Example Usage***
    
    ```py 
    >>> @bodo.jit
    ... def f(I, name):
    ...   return I.rename(name)
    
    >>> I = pd.Index(["a", "b", "c"])
    >>> f(I, "new_name")
    Index(['a', 'b', 'c'], dtype='object', name='new_name')
    ```
    
    
    
#### `pd.Index.duplicated`



- <code><apihead>pandas.Index.<apiname>duplicated</apiname>(keep='first')</apihead></code>
<br><br>
    ***Supported Arguments***: None
    
    ***Example Usage*** 
    
    ```py
      
    >>> @bodo.jit
    ... def f(I):
    ...   return I.duplicated()
    
    >>> idx = pd.Index(['a', 'b', None, 'a', 'c', None, 'd', 'b'])
    >>> f(idx)
    array([False, False, False,  True, False,  True, False,  True])
    ```
    
    
#### `pd.Index.drop_duplicates`



- <code><apihead>pandas.Index.<apiname>drop_duplicates</apiname>(keep='first')</apihead></code>
<br><br>
    ***Supported Arguments***: None
    
    ***Unsupported Index Types***
    
    - MultiIndex
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.drop_duplicates()
    
    >>> I = pd.Index(["a", "b", "c", "a", "b", "c"])
    >>> f(I)
    Index(['a', 'b', 'c'], dtype='object')
    ```
    
#### `pd.Index.isin`


- <code><apihead>pandas.Index.<apiname>isin</apiname>(values)</apihead></code>
<br><br>
    ***Supported Arguments***
    
    - `values`: list-like or array-like of values

    ***Unsupported Index Types***
    
      - MultiIndex
      - IntervalIndex
      - PeriodIndex
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.isin([0, 2, 4])

    >>> I = pd.Index([2, 4, 3, 4, 0, 3, 3, 5])
    >>> f(I)
    array([ True,  True, False,  True,  True, False, False, False])
    ```
    
#### `pd.Index.unique`


- <code><apihead>pandas.Index.<apiname>unique</apiname>()</apihead></code>
<br><br>
    ***Unsupported Index Types***
    
      - IntervalIndex
      - PeriodIndex
      - MultiIndex
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.unique()

    >>> I = pd.Index([1, 5, 2, 1, 0, 1, 5, 2, 1, 3])
    >>> f(I)
    Int64Index([1, 5, 2, 0, 3], dtype='int64')
    ```
  
    
#### `pd.Index.nunique`


- <code><apihead>pandas.Index.<apiname>nunique</apiname>(dropna=True)</apihead></code>
<br><br>
    ***Supported Arguments***:
    
    - `dropna`: can be True or False

    ***Unsupported Index Types***
    
      - IntervalIndex
      - MultiIndex
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.nunique()

    >>> I = pd.Index([1, 5, 2, 1, 0, 1, 5, 2, 1])
    >>> f(I)
    4
    ```
    
#### `pd.Index.sort_values`


- <code><apihead>pandas.Index.<apiname>sort_values</apiname>(return_indexer=False, ascending=True, na_position="last", key=None)</apihead></code>
<br><br>

    ***Supported Arguments***:
    
    - `ascending`: can be True or False
    - `na_position`: can be "first" or "last"
    
    ***Unsupported Index Types***
    
    - IntervalIndex
    - PeriodIndex
    - MultiIndex
    

    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.sort_values()

    >>> I = pd.Index([0, -1, 1, -5, 8, -13, -2, 3])
    >>> f(I)
    Int64Index([-13, -5, -2, -1, 0, 1, 3, 8], dtype='int64')
    ```
    
#### `pd.Index.argsort`


- <code><apihead>pandas.Index.<apiname>argsort</apiname>(<em>args, </em>*kwargs)</apihead></code>
<br><br>

    ***Supported Arguments***: None
    
    ***Unsupported Index Types***
    
    - IntervalIndex
    - MultiIndex
    

    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.argsort()

    >>> I = pd.Index(["A", "L", "P", "H", "A"])
    >>> f(I)
    array([0, 4, 3, 1, 2])
    ```
    
#### `pd.Index.all`


- <code><apihead>pandas.Index.<apiname>all</apiname>(<em>args, </em>*kwargs)</apihead></code>
<br><br>
    ***Supported Arguments***: None

    ***Supported Index Types***
    
      - NumericIndex (only Integers or Booleans)
      - RangeIndex
      - StringIndex
      - BinaryIndex

     !!! important
          Bodo diverges from the Pandas API for StringIndex and BinaryIndex by always returning a boolean instead of sometimes returning a string.
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.all()

    >>> I = pd.Index([1, 4, 9, 0, 3])
    >>> f(I)
    False
    ```

    
#### `pd.Index.any`


- <code><apihead>pandas.Index.<apiname>any</apiname>(<em>args, </em>*kwargs)</apihead></code>
<br><br>
    ***Supported Arguments***: None

    ***Supported Index Types***
    
      - NumericIndex (only Integers or Booleans)
      - RangeIndex
      - StringIndex
      - BinaryIndex

     !!! important
          Bodo diverges from the Pandas API for StringIndex and BinaryIndex by always returning a boolean instead of sometimes returning a string.
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.any()

    >>> I = pd.Index([1, 4, 9, 0, 3])
    >>> f(I)
    True
    ```
    
#### `pd.Index.argmax`


- <code><apihead>pandas.Index.<apiname>argmax</apiname>(axis=None, skipna=True, <em>args, </em>*kwargs)</apihead></code>
<br><br>
    ***Supported Arguments***: None

    ***Unsupported Index Types***
    
      - IntervalIndex
      - MultiIndex
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.argmax()

    >>> I = pd.Index([1, 4, 9, 0, 3])
    >>> f(I)
    2
    ```
    
#### `pd.Index.argmin`


- <code><apihead>pandas.Index.<apiname>argmin</apiname>(axis=None, skipna=True, <em>args, </em>*kwargs)</apihead></code>
<br><br>
    ***Supported Arguments***: None

    ***Unsupported Index Types***
    
      - IntervalIndex
      - MultiIndex
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.argmin()

    >>> I = pd.Index([1, 4, 9, 0, 3])
    >>> f(I)
    3
    ```
    
#### `pd.Index.where`


- <code><apihead>pandas.Index.<apiname>where</apiname>(cond, other=None)</apihead></code>
<br><br>
    ***Supported Arguments***:
    
      - `cond`: can be a Series or 1-dim array of booleans
      - `other`: can be a scalar, non-categorical Series, 1-dim numpy array or StringArray with a matching type for the Index

    ***Unsupported Index Types***
    
      - IntervalIndex
      - MultiIndex

    !!! Important
          Only supported for CategoricalIndex if the elements of other are the same as (or a subset of) the categories of the CategoricalIndex.
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I, C, O):
    ...   return I.where(C, O)

    >>> I = pd.Index(["A", "B", "C", "D", "E"])
    >>> C = pd.array([True, False, True, True, False])
    >>> O = pd.Series(["a", "e", "i", "o", "u")
    >>> f(I, C, O)
    Index(['A', 'e', 'C', 'D', 'u'], dtype='object')
    ```
    
#### `pd.Index.putmask`


- <code><apihead>pandas.Index.<apiname>putmask</apiname>(cond, other=None)</apihead></code>
<br><br>
    ***Supported Arguments***:
    
      - `cond`: can be a Series or 1-dim array of booleans
      - `other`: can be a scalar, non-categorical Series, 1-dim numpy array or StringArray with a matching type for the Index

    ***Unsupported Index Types***
    
      - IntervalIndex
      - MultiIndex

    !!! Important
          Only supported for CategoricalIndex if the elements of other are the same as (or a subset of) the categories of the CategoricalIndex.
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I, C, O):
    ...   return I.putmask(C, O)

    >>> I = pd.Index(["A", "B", "C", "D", "E"])
    >>> C = pd.array([True, False, True, True, False])
    >>> O = pd.Series(["a", "e", "i", "o", "u")
    >>> f(I, C, O)
    Index(['a', 'B', 'i', 'o', 'E'], dtype='object')
    ```
    
    
#### `pd.Index.union`


- <code><apihead>pandas.Index.<apiname>union</apiname>(other, sort=None)</apihead></code>
<br><br>
    ***Supported Arguments***:
    
      - `other`: can be an Index, Series, or 1-dim numpy array with a matching type for the Index

    ***Supported Index Types***
    
      - NumericIndex
      - StringIndex
      - BinaryIndex
      - RangeIndex
      - DatetimeIndex
      - TimedeltaIndex

    !!! Important
          Bodo diverges from the Pandas API for Index.union() in several ways: duplicates are removed, the order of elements may be different, the shortcuts for returning the same Index are removed, and a NumericIndex is always returned instead of a RangeIndex.
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit(distributed=["I", "J"])
    ... def f(I, J):
    ...    return I.union(J)

    >>> I = pd.Index([1, 2, 3, 4, 5])
    >>> J = pd.Index([2, 4, 6, 8, 10, 12])
    >>> f(I, J)
    Int64Index([1, 2, 3, 4, 5, 6, 8, 10, 12], dtype='int64')
    ```
    
    
#### `pd.Index.intersection`


- <code><apihead>pandas.Index.<apiname>intersection</apiname>(other, sort=None)</apihead></code>
<br><br>
    ***Supported Arguments***:
    
      - `other`: can be an Index, Series, or 1-dim numpy array with a matching type for the Index

    ***Supported Index Types***
    
      - NumericIndex
      - StringIndex
      - BinaryIndex
      - RangeIndex
      - DatetimeIndex
      - TimedeltaIndex

    !!! Important
          Bodo diverges from the Pandas API for Index.intersection() in several ways: the default is sort=None, and a NumericIndex is always returned instead of a RangeIndex.
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit(distributed=["I", "J"])
    ... def f(I, J):
    ...    return I.intersection(J)

    >>> I = pd.Index([1, 2, 3, 4, 5])
    >>> J = pd.Index([2, 4, 6, 8, 10, 12])
    >>> f(I, J)
    Int64Index([2, 4], dtype='int64')
    ```
    
    
#### `pd.Index.difference`


- <code><apihead>pandas.Index.<apiname>difference</apiname>(other, sort=None)</apihead></code>
<br><br>
    ***Supported Arguments***:
    
      - `other`: can be an Index, Series, or 1-dim numpy array with a matching type for the Index

    ***Supported Index Types***
    
      - NumericIndex
      - StringIndex
      - BinaryIndex
      - RangeIndex
      - DatetimeIndex
      - TimedeltaIndex

    !!! Important
          Bodo diverges from the Pandas API for Index.difference() in several ways: the order of elements may be different and a NumericIndex is always returned instead of a RangeIndex.
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit(distributed=["I", "J"])
    ... def f(I, J):
    ...    return I.difference(J)

    >>> I = pd.Index([1, 2, 3, 4, 5])
    >>> J = pd.Index([2, 4, 6, 8, 10, 12])
    >>> f(I, J)
    Int64Index([1, 3, 5], dtype='int64')
    ```
    
    
    
#### `pd.Index.symmetric_difference`


- <code><apihead>pandas.Index.<apiname>symmetric_difference</apiname>(other, sort=None)</apihead></code>
<br><br>
    ***Supported Arguments***:
    
      - `other`: can be an Index, Series, or 1-dim numpy array with a matching type for the Index

    ***Supported Index Types***
    
      - NumericIndex
      - StringIndex
      - BinaryIndex
      - RangeIndex
      - DatetimeIndex
      - TimedeltaIndex

    !!! Important
          Bodo diverges from the Pandas API for Index.symmetric_difference() in several ways: the order of elements may be different and a NumericIndex is always returned instead of a RangeIndex.
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit(distributed=["I", "J"])
    ... def f(I, J):
    ...    return I.difference(J)

    >>> I = pd.Index([1, 2, 3, 4, 5])
    >>> J = pd.Index([2, 4, 6, 8, 10, 12])
    >>> f(I, J)
    Int64Index([1, 3, 5, 6, 8, 10, 12], dtype='int64')
    ```
    
#### `pd.Index.repeat`


- <code><apihead>pandas.Index.<apiname>repeat</apiname>(repeats, axis=None)</apihead></code>
<br><br>
    ***Supported Arguments***:
    
      - `repeat`: can be a non-negative integer or array of non-negative integers

    ***Supported Index Types***
    
      - NumericIndex
      - StringIndex
      - RangeIndex
      - DatetimeIndex
      - TimedeltaIndex
      - CategoricalIndex

    !!! Important
          If repeats is an integer array but its size is not the same as the length of I, undefined behavior may occur.
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit(distributed=["I"])
    ... def f(I):
    ...    return I.repeat(3)

    >>> I = pd.Index(["A", "B", "C"])
    >>> f(I)
    Index(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'], dtype='object')
    ```
    
### Missing values


#### `pd.Index.isna`


- <code><apihead>pandas.Index.<apiname>isna</apiname>()</apihead></code>
<br><br>
    ***Unsupported Index Types***
    
      - MultiIndex
      - IntervalIndex
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.isna()
    
    >>> I = pd.Index([1,None,3])
    >>> f(I)
    [False  True False]
    ```

#### `pd.Index.isnull`


- <code><apihead>pandas.Index.<apiname>isnull</apiname>()</apihead></code>
<br><br>
    ***Unsupported Index Types***
    
      - MultiIndex
      - IntervalIndex
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.isnull()

    >>> I = pd.Index([1,None,3])
    >>> f(I)
    [False  True False]
    ```
    
    
### Conversion

#### `pd.Index.map`


- <code><apihead>pandas.Index.<apiname>map</apiname>(mapper, na_action=None)</apihead></code>
<br><br>
    ***Unsupported Index Types***
    
    - MultiIndex
    - IntervalIndex
    
    ***Supported Arguments***
   
    - `mapper`: must be a function, function cannot return tuple type
    
    ***Example Usage***
    
    ```py 
    >>> @bodo.jit
    ... def f(I):
    ...   return I.map(lambda x: x + 2)

    >>> I = pd.Index([1,None,3])
    >>> f(I)
    Float64Index([3.0, nan, 5.0], dtype='float64')
    ```
    
#### `pd.Index.to_series`


- <code><apihead>pandas.Index.<apiname>to_series</apiname>(index=None, name=None)</apihead></code>
<br><br>
    ***Supported Arguments***:
      
      - `index`: can be a Index, Series, 1-dim numpy array, list, or tuple
      - `name`: can be a string or int

    ***Unsupported Index Types***
    
      - IntervalIndex
      - PeriodIndex
      - MultiIndex
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I, J):
    ...   return I.to_series(index=J)

    >>> I = pd.Index([1, 4, 9, 0, 3])
    >>> J = pd.Index(["A", "B", "C", "D", "E"])
    >>> f(I, J)
    A    1
    B    4
    C    9
    D    0
    E    3
    dtype: int64
    ```

#### `pd.Index.to_frame`


- <code><apihead>pandas.Index.<apiname>to_frame</apiname>(index=True, name=None)</apihead></code>
<br><br>
    ***Supported Arguments***:
      
      - `index`: can be a True or False
      - `name`: can be a string or int

    ***Unsupported Index Types***
    
      - IntervalIndex
      - PeriodIndex
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.to_frame(index=False)

    >>> I = pd.Index(["A", "E", "I", "O", "U", "Y"], name="vowels")
    >>> f(I)
      vowels
    0      A
    1      E
    2      I
    3      O
    4      U
    5      Y
    ```
    
#### `pd.Index.to_numpy`


- <code><apihead>pandas.Index.<apiname>to_numpy</apiname>(dtype=None, copy=True, na_value=None)</apihead></code>
<br><br>
    ***Supported Arguments***:
      
      - `copy`: can be a True or False

    ***Unsupported Index Types***
    
      - PeriodIndex
      - MultiIndex

    !!! Important
        Sometimes Bodo returns a Pandas array instead of a np.ndarray. Cases
        include a NumericIndex of integers containing nulls, or a CategoricalIndex.
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.numpy()

    >>> I = pd.Index([1, 9, -1, 3, 0, 1, 6])
    >>> f(I)
    [ 1  9 -1  3  0  1  6]
    ```
    
#### `pd.Index.to_list`


- <code><apihead>pandas.Index.<apiname>to_list</apiname>()</apihead></code>
<br><br>
    ***Unsupported Index Types***
    
      - PeriodIndex
      - IntervalIndex
      - MultiIndex
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.to_list()

    >>> I = pd.RangeIndex(5, -1, -1)
    >>> f(I)
    [5, 4, 3, 2, 1, 0]
    ```

#### `pd.Index.tolist`


- <code><apihead>pandas.Index.<apiname>tolist</apiname>()</apihead></code>
<br><br>
    ***Unsupported Index Types***
    
      - PeriodIndex
      - IntervalIndex
      - DatetimeIndex
      - TimedeltaIndex
      - MultiIndex
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.tolist()

    >>> I = pd.RangeIndex(5, -1, -1)
    >>> f(I)
    [5, 4, 3, 2, 1, 0]
    ```

## Numeric Index


Numeric index objects `RangeIndex`, `Int64Index`, `UInt64Index` and
`Float64Index` are supported as index to dataframes and series.
Constructing them in Bodo functions, passing them to Bodo functions (unboxing),
and returning them from Bodo functions (boxing) are also supported.


#### `pd.RangeIndex`



- <code><apihead>pandas.<apiname>RangeIndex</apiname>(start=None, stop=None, step=None, dtype=None, copy=False, name=None)</apihead></code>
<br><br>

    ***Supported Arguments***
        
    - `start`: integer
    - `stop`: integer
    - `step`: integer
    - `name`: String
    
    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.RangeIndex(0, 10, 2)

    >>> f(I)
    RangeIndex(start=0, stop=10, step=2)
    ```

#### `pd.Int64Index`


- <code><apihead>pandas.<apiname>Int64Index</apiname>(data=None, dtype=None, copy=False, name=None)</apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ... return pd.Int64Index(np.arange(3))

    >>> f()
    Int64Index([0, 1, 2], dtype='int64')
    ```

#### `pd.UInt64Index`


- <code><apihead>pandas.<apiname>UInt64Index</apiname>(data=None, dtype=None, copy=False, name=None)</apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ... return pd.UInt64Index([1,2,3])

    >>> f()
    UInt64Index([0, 1, 2], dtype='uint64')
    ```

#### `pd.Float64Index`


- <code><apihead>pandas.<apiname>Float64Index</apiname>(data=None, dtype=None, copy=False, name=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    
    - `data`: list or array
    - `copy`: Boolean
    - `name`: String


    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ... return pd.Float64Index(np.arange(3))

    >>> f()
    Float64Index([0.0, 1.0, 2.0], dtype='float64')
    ```
    
     
     
     
## DatetimeIndex

`DatetimeIndex` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.


#### `pd.DateTimeIndex`



- <code><apihead>pandas.<apiname>DatetimeIndex</apiname></apihead></code>
<br><br>

    ***Supported Arguments***
    
    - `data`: array-like of datetime64, Integer, or strings

### Date fields 

#### `pd.DateTimeIndex.year`



- <code><apihead>pandas.DatetimeIndex.<apiname>year</apiname></apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.year

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([2019, 2019, 2019, 2020, 2020], dtype='int64')
    ```
    
    
#### `pd.DateTimeIndex.month`



- <code><apihead>pandas.DatetimeIndex.<apiname>month</apiname></apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.month

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([12, 12, 12, 1, 1], dtype='int64')
    ```
    
    
#### `pd.DateTimeIndex.day`


- <code><apihead>pandas.DatetimeIndex.<apiname>day</apiname></apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.day

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([31, 31, 31, 1, 1], dtype='int64')
    ```

#### `pd.DateTimeIndex.hour`



- <code><apihead>pandas.DatetimeIndex.<apiname>hour</apiname></apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.hour

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([2, 12, 22, 9, 19], dtype='int64')
    ```
    
    
    
    
#### `pd.DateTimeIndex.minute`



- <code><apihead>pandas.DatetimeIndex.<apiname>minute</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.minute

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([32, 42, 52, 2, 12], dtype='int64')
    ```

#### `pd.DateTimeIndex.second`



- <code><apihead>pandas.DatetimeIndex.<apiname>second</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.second

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([45, 35, 25, 15, 5], dtype='int64')
    ```
    
    
#### `pd.DateTimeIndex.microsecond`


- <code><apihead>pandas.DatetimeIndex.<apiname>microsecond</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.microsecond

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 01:01:01", end="2019-12-31 01:01:02", periods=5))
    >>> f(I)
    Int64Index([0, 250000, 500000, 750000, 0], dtype='int64')
    ```

#### `pd.DateTimeIndex.nanosecond`


- <code><apihead>pandas.DatetimeIndex.<apiname>nanosecond</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.nanosecond

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 01:01:01.0000001", end="2019-12-31 01:01:01.0000002", periods=5))
    >>> f(I)
    Int64Index([100, 125, 150, 175, 200], dtype='int64')
    ```
    
    
    
    
#### `pd.DateTimeIndex.date`



- <code><apihead>pandas.DatetimeIndex.<apiname>date</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.date

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    [datetime.date(2019, 12, 31) datetime.date(2019, 12, 31) datetime.date(2019, 12, 31) datetime.date(2020, 1, 1) datetime.date(2020, 1, 1)]
    ```
    
    
#### `pd.DateTimeIndex.dayofyear`



- <code><apihead>pandas.DatetimeIndex.<apiname>dayofyear</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.dayofyear

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([365, 365, 365, 1, 1], dtype='int64')
    ```

#### `pd.DateTimeIndex.day_of_year`


- <code><apihead>pandas.DatetimeIndex.<apiname>day_of_year</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.day_of_year

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([365, 365, 365, 1, 1], dtype='int64')
    ```

#### `pd.DateTimeIndex.dayofweek`



- <code><apihead>pandas.DatetimeIndex.<apiname>dayofweek</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.dayofweek

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 2, 2], dtype='int64')
    ```

#### `pd.DateTimeIndex.day_of_week`


- <code><apihead>pandas.DatetimeIndex.<apiname>day_of_week</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.day_of_week

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 2, 2], dtype='int64')
    ```

#### `pd.DateTimeIndex.is_leap_year`



- <code><apihead>pandas.DatetimeIndex.<apiname>is_leap_year</apiname></apihead></code>
<br><br>
    ***Example Usage***
    
    ```py

    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_leap_year

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    [Flase False False True True]
    ```
    
    
#### `pd.DateTimeIndex.is_month_start`



- <code><apihead>pandas.DatetimeIndex.<apiname>is_month_start</apiname></apihead></code>
<br><br>
    ***Example Usage***
    
    ```py

    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_month_start

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([0, 0, 0, 1, 1], dtype='int64')
    ```
    
    
#### `pd.DateTimeIndex.is_month_end`



- <code><apihead>pandas.DatetimeIndex.<apiname>is_month_end</apiname></apihead></code>
<br><br>
    ***Example Usage***
    
    ```py

    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_month_end

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 0, 0], dtype='int64')
    ```

#### `pd.DateTimeIndex.is_quarter_start`



- <code><apihead>pandas.DatetimeIndex.<apiname>is_quarter_start</apiname></apihead></code>
<br><br>
    ***Example Usage***
    
    ```py

    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_quarter_start

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([0, 0, 0, 1, 1], dtype='int64')
    ```

#### `pd.DateTimeIndex.is_quarter_end`



- <code><apihead>pandas.DatetimeIndex.<apiname>is_quarter_end</apiname></apihead></code>
<br><br>
    ***Example Usage***
    
    ```py

    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_quarter_end

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 0, 0], dtype='int64')
    ```
    
    
#### `pd.DateTimeIndex.is_year_start`


- <code><apihead>pandas.DatetimeIndex.<apiname>is_year_start</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_year_start

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([0, 0, 0, 1, 1], dtype='int64')
    ```
    
    
#### `pd.DateTimeIndex.is_year_end`


- <code><apihead>pandas.DatetimeIndex.<apiname>is_year_end</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_year_end

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 0, 0], dtype='int64')
    ```

#### `pd.DateTimeIndex.week`


- <code><apihead>pandas.DatetimeIndex.<apiname>week</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.week

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 1, 1], dtype='int64')
    ```

#### `pd.DateTimeIndex.weekday`


- <code><apihead>pandas.DatetimeIndex.<apiname>weekday</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.weekday

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 2, 2], dtype='int64')
    ```
    
    
    
    
#### `pd.DateTimeIndex.weekofyear`


- <code><apihead>pandas.DatetimeIndex.<apiname>weekofyear</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.weekofyear

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 1,1], dtype='int64')
    ```

#### `pd.DateTimeIndex.quarter`


- <code><apihead>pandas.DatetimeIndex.<apiname>quarter</apiname></apihead></code>
<br><br>    
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.quarter
    
    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([4, 4, 4, 1, 1], dtype='int64')
    ```


Subtraction of `Timestamp` from `DatetimeIndex` and vice versa
is supported.

Comparison operators `==`, `!=`, `>=`, `>`, `<=`, `<` between
`DatetimeIndex` and a string of datetime
are supported.

## TimedeltaIndex

`TimedeltaIndex` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.

#### `pd.TimedeltaIndex`


- <code><apihead>pandas.<apiname>TimedeltaIndex</apiname>(data=None, unit=None, freq=NoDefault.no_default, closed=None, dtype=dtype('&lt;m8[ns]'), copy=False, name=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    
    -`data`:  must be array-like of timedelta64ns or Ingetger.

#### `pd.TimedeltaIndex.days`


- <code><apihead>pandas.TimedeltaIndex.<apiname>days</apiname></apihead></code>
<br><br>
    ***Example Usage***

    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.days

    >>> I = pd.TimedeltaIndex([pd.Timedelta(3, unit="D"))])
    >>> f(I)
    Int64Index([3], dtype='int64')
    ```

#### `pd.TimedeltaIndex.seconds`


- <code><apihead>pandas.TimedeltaIndex.<apiname>seconds</apiname></apihead></code>
<br><br>
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.seconds

    >>> I = pd.TimedeltaIndex([pd.Timedelta(-2, unit="S"))])
    >>> f(I)
    Int64Index([-2], dtype='int64')
    ```

#### `pd.TimedeltaIndex.microseconds`


- <code><apihead>pandas.TimedeltaIndex.<apiname>microseconds</apiname></apihead></code>
<br><br>
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.microseconds

    >>> I = pd.TimedeltaIndex([pd.Timedelta(11, unit="micros"))])
    >>> f(I)
    Int64Index([11], dtype='int64')
    ```

#### `pd.TimedeltaIndex.nanoseconds`


- <code><apihead>pandas.TimedeltaIndex.<apiname>nanoseconds</apiname></apihead></code>
<br><br>
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.nanoseconds
    
    >>> I = pd.TimedeltaIndex([pd.Timedelta(7, unit="nanos"))])
    >>> f(I)
    Int64Index([7], dtype='int64')
    ```

## PeriodIndex

`PeriodIndex` objects can be
boxed/unboxed and set as index to dataframes and series.
Operations on them will be supported in upcoming releases.

## BinaryIndex


`BinaryIndex` objects can be
boxed/unboxed and set as index to dataframes and series.
Operations on them will be supported in upcoming releases.

## MultiIndex

#### `pd.MultiIndex.from_product`


- <code><apihead>pandas.<apiname>MultiIndex.from_product</apiname></apihead> </code>
<br><br>    (*iterables* and *names* supported as tuples, no parallel support yet)
