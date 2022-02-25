# Index objects

## Index

### Properties

#### `pd.Index.name`

-   ++pandas.Index.%%name%%++


    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.name
    
    >>> I = pd.Index([1,2,3], name = "hello world")
    >>> f(I)
    "hello world"
    ```
    
    
    
#### `pd.Index.shape`


-   ++pandas.Index.%%shape%%++


    ***Unsupported Index Types***
    
    - MultiIndex
    - IntervalIndex

    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f(I):
    ...   return I.shape
    
    >>> I = pd.Index([1,2,3])
    >>> f(I)
    (3,)
    ```

#### `pd.Index.is_monotonic_increasing`

-   ++pandas.Index.%%is_monotonic_increasing%%++ and ++pandas.%%Index.is_monotonic%%++


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


-   ++pandas.Index.%%is_monotonic_decreasing%%++


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

-   ++pandas.Index.%%values%%++


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

-   ++pandas.Index.%%nbytes%%++


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

### Modifications and computations

#### `pd.Index.copy`


-   ++pandas.Index.%%copy%%(name=None, deep=False, dtype=None, names=None)++

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

-  ++pandas.Index.%%get_loc%%(key, method=None, tolerance=None)++


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

-   ++pandas.Index.%%take%%(indices, axis=0, allow_fill=True, fill_value=None, **kwargs)++


    ***Supported Arguments***
    
    - `indices`:  can be boolean Array like, integer Array like, or slice
    
    ***Unsupported Index Types***
    
    - MultiIndex
    - IntervalIndex
    
    !!! important
         Bodo **Does Not** support `kwargs`, even for compatibility.

#### `pd.Index.min`

-   ++pandas.Index.%%min%%(axis=None, skipna=True, *args, **kwargs)++


    ***Supported Arguments***: None
    
    ***Supported Index Types***
    
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

-   ++pandas.Index.%%max%%(axis=None, skipna=True, *args, **kwargs)++


    ***Supported Arguments***: None
    
    ***Supported Index Types***
    
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

-   ++pandas.Index.%%rename%%(name, inplace=False)++


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


-   ++pandas.Index.%%duplicated%%(keep='first')++

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


-   ++pandas.Index.%%drop_duplicates%%(keep='first')++

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

### Missing values


#### `pd.Index.isna`

-   ++pandas.Index.%%isna%%()++

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

-   ++pandas.Index.%%isnull%%()++

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

-   ++pandas.Index.%%map%%(mapper, na_action=None)++

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

## Numeric Index


Numeric index objects `RangeIndex`, `Int64Index`, `UInt64Index` and
`Float64Index` are supported as index to dataframes and series.
Constructing them in Bodo functions, passing them to Bodo functions (unboxing),
and returning them from Bodo functions (boxing) are also supported.


#### `pd.RangeIndex`


- ++pandas.%%RangeIndex%%(start=None, stop=None, step=None, dtype=None, copy=False, name=None)++


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

- ++pandas.%%Int64Index%%(data=None, dtype=None, copy=False, name=None)++

    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ... return pd.Int64Index(np.arange(3))

    >>> f()
    Int64Index([0, 1, 2], dtype='int64')
    ```

#### `pd.UInt64Index`

- ++pandas.%%UInt64Index%%(data=None, dtype=None, copy=False, name=None)++

    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ... return pd.UInt64Index([1,2,3])

    >>> f()
    UInt64Index([0, 1, 2], dtype='uint64')
    ```

#### `pd.Float64Index`

- ++pandas.%%Float64Index%%(data=None, dtype=None, copy=False, name=None)++

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


- ++pandas.%%DatetimeIndex%%++


    ***Supported Arguments***
    
    - `data`: array-like of datetime64, Integer, or strings

### Date fields 

#### `pd.DateTimeIndex.year`


-   ++pandas.DatetimeIndex.%%year%%++

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


-   ++pandas.DatetimeIndex.%%month%%++

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

-   ++pandas.DatetimeIndex.%%day%%++

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


-   ++pandas.DatetimeIndex.%%hour%%++

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


-   ++pandas.DatetimeIndex.%%minute%%++

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


-   ++pandas.DatetimeIndex.%%second%%++

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

-   ++pandas.DatetimeIndex.%%microsecond%%++

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

-   ++pandas.DatetimeIndex.%%nanosecond%%++

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


-   ++pandas.DatetimeIndex.%%date%%++

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


-   ++pandas.DatetimeIndex.%%dayofyear%%++

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

-   ++pandas.DatetimeIndex.%%day_of_year%%++

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


-   ++pandas.DatetimeIndex.%%dayofweek%%++

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

-   ++pandas.DatetimeIndex.%%day_of_week%%++

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


-   ++pandas.DatetimeIndex.%%is_leap_year%%++

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


-   ++pandas.DatetimeIndex.%%is_month_start%%++

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


-   ++pandas.DatetimeIndex.%%is_month_end%%++

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


-   ++pandas.DatetimeIndex.%%is_quarter_start%%++

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


-   ++pandas.DatetimeIndex.%%is_quarter_end%%++

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

-   ++pandas.DatetimeIndex.%%is_year_start%%++

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

-   ++pandas.DatetimeIndex.%%is_year_end%%++

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

-   ++pandas.DatetimeIndex.%%week%%++

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

-   ++pandas.DatetimeIndex.%%weekday%%++

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

-   ++pandas.DatetimeIndex.%%weekofyear%%++

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

-   ++pandas.DatetimeIndex.%%quarter%%++
    
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

-   ++pandas.%%TimedeltaIndex%%(data=None, unit=None, freq=NoDefault.no_default, closed=None, dtype=dtype('<m8[ns]'), copy=False, name=None)++

    ***Supported Arguments***
    
    -`data`:  must be array-like of timedelta64ns or Ingetger.

#### `pd.TimedeltaIndex.days`

-   ++pandas.TimedeltaIndex.%%days%%++

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

-   ++pandas.TimedeltaIndex.%%seconds%%++

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

-   ++pandas.TimedeltaIndex.%%microseconds%%++

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

-   ++pandas.TimedeltaIndex.%%nanoseconds%%++

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

-   ++pandas.%%MultiIndex.from_product%%++ 
    (*iterables* and *names* supported as tuples, no parallel support yet)
