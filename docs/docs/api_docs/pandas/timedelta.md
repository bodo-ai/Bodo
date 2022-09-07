# Timedelta

Timedelta functionality is documented in [`pandas.Timedelta`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html){target=blank}.

#### `pd.Timedelta`


- <code><apihead>pandas.<apiname>Timedelta</apiname>(value=<object object\>, unit="ns", days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)</apihead></code>


    ***Supported Arguments***

    - `value`: Integer (with constant string unit argument), String, Pandas Timedelta, datetime Timedelta
    - `unit`: Constant String. Only has an effect when passing an integer `value`, see [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html) for allowed values.
    - `days`: Integer
    - `seconds`: Integer
    - `microseconds`: Integer
    - `milliseconds`: Integer
    - `minutes`: Integer
    - `hours`: Integer
    - `weeks`: Integer

    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   td1 = pd.Timedelta("10 Seconds")
    ...   td2 = pd.Timedelta(10, unit= "W")
    ...   td3 = pd.Timedelta(days= 10, hours=2, microseconds= 23)
    ...   return (td1, td2, td3)
    >>> f()
    (Timedelta('0 days 00:00:10'), Timedelta('70 days 00:00:00'), Timedelta('10 days 02:00:00.000023'))
    ```
    
#### `pd.Timedelta.components`



- <code><apihead>pandas.Timedelta.<apiname>components</apiname></apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).components
    >>> f()
    Components(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23, nanoseconds=0)
    ```
  
#### `pd.Timedelta.days`



- <code><apihead>pandas.Timedelta.<apiname>days</apiname></apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).days
    >>> f()
    10
    ```
   
#### `pd.Timedelta.delta`



- <code><apihead>pandas.Timedelta.<apiname>delta</apiname></apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(microseconds=23).delta
    >>> f()
    23000
    ```
  
#### `pd.Timedelta.microseconds`



- <code><apihead>pandas.Timedelta.<apiname>microseconds</apiname></apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).microseconds
    >>> f()
    23
    ```
  
#### `pd.Timedelta.nanoseconds`
                              


- <code><apihead>pandas.Timedelta.<apiname>nanoseconds</apiname></apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).nanoseconds
    >>> f()
    0
    ```

#### `pd.Timedelta.seconds`
                              


- <code><apihead>pandas.Timedelta.<apiname>seconds</apiname></apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta("10 nanoseconds").nanoseconds
    >>> f()
    10
    ```
  
#### `pd.Timedelta.value`
                              


- <code><apihead>pandas.Timedelta.<apiname>value</apiname></apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta("13 nanoseconds").value
    >>> f()
    13
    ```

#### `pd.Timedelta.ceil`
                              


- <code><apihead>pandas.Timedelta.<apiname>ceil(freq)</apiname></apihead></code>
<br><br>

    ***Supported Arguments***


    -  `freq`: String

    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).ceil("D")
    >>> f()
    11 days 00:00:00
    ```
    
#### `pd.Timedelta.floor`
                              


- <code><apihead>pandas.Timedelta.<apiname>floor</apiname></apihead></code>
<br><br>
    ***Supported Arguments***


    - `freq`: String

    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).floor("D")
    >>> f()
    10 days 00:00:00
    ```
    
#### `pd.Timedelta.round`
                              

- <code><apihead>pandas.Timedelta.<apiname>round</apiname></apihead></code>
<br><br>
    ***Supported Arguments***

    - `freq`: String

    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return (pd.Timedelta(days=10, hours=12).round("D"), pd.Timedelta(days=10, hours=13).round("D"))
    >>> f()
    (Timedelta('10 days 00:00:00'), Timedelta('11 days 00:00:00'))
    ```
    
#### `pd.Timedelta.to_numpy`
                              

- <code><apihead>pandas.Timedelta.<apiname>to_numpy</apiname>()</apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).to_numpy()
    >>> f()
    871623013023000 nanoseconds
    ```
  
#### `pd.Timedelta.to_pytimedelta`
                                


- <code><apihead>pandas.Timedelta.<apiname>to_pytimedelta</apiname>()</apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).to_pytimedelta()
    >>> f()
    10 days, 2:07:03.013023
    ```
   
#### `pd.Timedelta.to_timedelta64`
                                

- <code><apihead>pandas.Timedelta.<apiname>to_timedelta64</apiname>()</apihead></code>
<br><br>
    ***Example Usage***
    
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).to_timedelta64()
    >>> f()
    871623013023000 nanoseconds
    ```
   
#### `pd.Timedelta.total_seconds`


- <code><apihead>pandas.Timedelta.<apiname>total_seconds</apiname>()</apihead></code>
<br><br>
    ***Example Usage***
    ```py
    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).total_seconds()
    >>> f()
    871623.013023
    ```