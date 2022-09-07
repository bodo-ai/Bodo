Miscellaneous Supported Python API {#miscapi}
==================================

In this page, we will discuss some useful Bodo features and concepts.

## Nullable Integers in Pandas

DataFrame and Series objects with integer data need special care due to
[integer NA issues in Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#nan-integer-na-values-and-na-type-promotions){target="blank"}.
By default, Pandas dynamically converts integer columns to floating
point when missing values (NAs) are needed, which can result in loss of
precision as well as type instability.

<br>

Pandas introduced [a new nullable integer datatype](https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html#integer-na){target="blank"}
that can solve this issue, which is also supported by Bodo. For example,
this code reads column A into a nullable integer array (the capital "I"
denotes nullable integer type):

``` py
data = (
    "11,1.2\n"
    "-2,\n"
    ",3.1\n"
    "4,-0.1\n"
)

with open("data/data.csv", "w") as f:
    f.write(data)


@bodo.jit(distributed=["df"])
def f():
    dtype = {"A": "Int64", "B": "float64"}
    df = pd.read_csv("data/data.csv", dtype = dtype, names = dtype.keys())
    return df

f()
```

<br/> 

![dataframe with nullable integers](../img/dataframe_misc.svg#center)

## Checking NA Values

When an operation iterates over the values in a Series or Array, type
stability requires special handling for NAs using `pd.isna()`. For
example, `Series.map()` applies an operation to each element in the
series and failing to check for NAs can result in garbage values
propagating.

```py
S = pd.Series(pd.array([1, None, None, 3, 10], dtype="Int8"))

@bodo.jit
def map_copy(S):
    return S.map(lambda a: a if not pd.isna(a) else None)

print(map_copy(S))
```

```console
0       1
1     <NA>
2     <NA>
3       3
4      10
dtype: Int8
```

### Boxing/Unboxing Overheads

Bodo uses efficient native data structures which can be different than
Python. When Python values are passed to Bodo, they are *unboxed* to
native representation. On the other hand, returning Bodo values requires
*boxing* to Python objects. Boxing and unboxing can have significant
overhead depending on size and type of data. For example, passing string
column between Python/Bodo repeatedly can be expensive:

```py
@bodo.jit(distributed=["df"])
def gen_data():
    df = pd.read_parquet("data/cycling_dataset.pq")
    df["hr"] = df["hr"].astype(str)
    return df

@bodo.jit(distributed=["df", "x"])
def mean_power(df):
    x = df.hr.str[1:]
    return x

df = gen_data()
res = mean_power(df)
print(res)
```
Output:

```console
0        1
1        2
2        2
3        3
4        3
        ..
3897    00
3898    00
3899    00
3900    00
3901    00
Name: hr, Length: 3902, dtype: object
```

One can try to keep data in Bodo functions as much as possible to avoid
boxing/unboxing overheads:

```py
@bodo.jit(distributed=["df"])
def gen_data():
    df = pd.read_parquet("data/cycling_dataset.pq")
    df["hr"] = df["hr"].astype(str)
    return df

@bodo.jit(distributed=["df", "x"])
def mean_power(df):
    x = df.hr.str[1:]
    return x

@bodo.jit
def f():
    df = gen_data()
    res = mean_power(df)
    print(res)

f()
```

```console
0        1
1        2
2        2
3        3
4        3
        ..
3897    00
3898    00
3899    00
3900    00
3901    00
Name: hr, Length: 3902, dtype: object
```

### Iterating Over Columns

Iterating over columns in a dataframe can cause type stability issues,
since column types in each iteration can be different. Bodo supports
this usage for many practical cases by automatically unrolling loops
over dataframe columns when possible. For example, the example below
computes the sum of all data frame columns:

```py
@bodo.jit
def f():
    n = 20
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
    s = 0
    for c in df.columns:
     s += df[c].sum()
    return s

f()
```

```console
2680.0
```

For automatic unrolling, the loop needs to be a `for` loop over column
names that can be determined by Bodo at compile time.

## Regular Expressions using `re`

Bodo supports string processing using Pandas and the `re` standard
package, offering significant flexibility for string processing
applications. For example, `re` can be used in user-defined functions
(UDFs) applied to Series and DataFrame values:

```py
import re

@bodo.jit
def f(S):
    def g(a):
        res = 0
        if re.search(".*AB.*", a):
            res = 3
        if re.search(".*23.*", a):
            res = 5
        return res

    return S.map(g)

S = pd.Series(["AABCDE", "BBABCE", "1234"])
f(S)
```

```console
0    3
1    3
2    5
dtype: int64
```

Below is a reference list of supported functionality. Full functionality
is documented in [standard re
documentation](https://docs.python.org/3/library/re.html). All functions
except *finditer* are supported. Note that currently, Bodo
JIT uses Python's `re` package as backend and therefore the compute
speed of these functions is similar to Python.

####   re.A

- <code><apihead>re.<apiname>A</apiname></apihead></code>
<br><br>
####   re.ASCII

- <code><apihead>re.<apiname>ASCII</apiname></apihead></code>
<br><br>
####   re.DEBUG

- <code><apihead>re.<apiname>DEBUG</apiname></apihead></code>
<br><br>
####   re.I

- <code><apihead>re.<apiname>I</apiname></apihead></code>
<br><br>
####   re.IGNORECASE

- <code><apihead>re.<apiname>IGNORECASE</apiname></apihead></code>
<br><br>
####   re.L

- <code><apihead>re.<apiname>L</apiname></apihead></code>
<br><br>
####   re.LOCALE

- <code><apihead>re.<apiname>LOCALE</apiname></apihead></code>
<br><br>
####   re.M

- <code><apihead>re.<apiname>M</apiname></apihead></code>
<br><br>
####   re.MULTILINE

- <code><apihead>re.<apiname>MULTILINE</apiname></apihead></code>
<br><br>
####   re.S

- <code><apihead>re.<apiname>S</apiname></apihead></code>
<br><br>
####   re.DOTALL

- <code><apihead>re.<apiname>DOTALL</apiname></apihead></code>
<br><br>
####   re.X

- <code><apihead>re.<apiname>X</apiname></apihead></code>
<br><br>
####   re.VERBOSE

- <code><apihead>re.<apiname>VERBOSE</apiname></apihead></code>
<br><br>
####   re.search

- <code><apihead>re.<apiname>search</apiname>(pattern, string, flags=0)</apihead></code>
<br><br>
####   re.match

- <code><apihead>re.<apiname>match</apiname>(pattern, string, flags=0)</apihead></code>
<br><br>
####   re.fullmatch

- <code><apihead>re.<apiname>fullmatch</apiname>(pattern, string, flags=0)</apihead></code>
<br><br>
####   re.split

- <code><apihead>re.<apiname>split</apiname>(pattern, string, maxsplit=0, flags=0)</apihead></code>
<br><br>
####   re.findall

- <code><apihead>re.<apiname>findall</apiname>(pattern, string, flags=0)</apihead></code>
<br><br>
    The `pattern` argument should be a constant string for
    multi-group patterns (for Bodo to know the output will be a list of
    string tuples). An error is raised otherwise.
    
    ***Example Usage***:
    
    ```py
    >>> @bodo.jit
    ... def f(pat, in_str):
    ...     return re.findall(pat, in_str)
    ...
    >>> f(r"\w+", "Words, words, words.")
    ['Words', 'words', 'words']
    ```
    
    Constant multi-group pattern works:
    
    ```py
    >>> @bodo.jit
    ... def f2(in_str):
    ...     return re.findall(r"(\w+).*(\d+)", in_str)
    ...
    >>> f2("Words, 123")
    [('Words', '3')]
    ```
    
    Non-constant multi-group pattern throws an error:
    
    ```py
    >>> @bodo.jit
    ... def f(pat, in_str):
    ...     return re.findall(pat, in_str)
    ...
    >>> f(r"(\w+).*(\d+)", "Words, 123")
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "/Users/user/dev/bodo/bodo/libs/re_ext.py", line 338, in _pat_findall_impl
        raise ValueError(
    ValueError: pattern string should be constant for 'findall' with multiple groups
    ```

####  re.sub

- <code><apihead>re.<apiname>sub</apiname>(pattern, repl, string, count=0, flags=0)</apihead></code>
<br><br>
####  re.subn

- <code><apihead>re.<apiname>subn</apiname>(pattern, repl, string, count=0, flags=0)</apihead></code>
<br><br>
####  re.escape

- <code><apihead>re.<apiname>escape</apiname>(pattern)</apihead></code>
<br><br>
####  re.purge

- <code><apihead>re.<apiname>purge</apiname></apihead></code>
<br><br>
####  re.Pattern.search

- <code><apihead>re.Pattern.<apiname>search</apiname>(string[, pos[, endpos]])</apihead></code>
<br><br>
####  re.Pattern.match

- <code><apihead>re.Pattern.<apiname>match</apiname>(string[, pos[, endpos]])</apihead></code>
<br><br>
####  re.Pattern.fullmatch

- <code><apihead>re.Pattern.<apiname>fullmatch</apiname>(string[, pos[, endpos]])</apihead></code>
<br><br>
####  re.Pattern.split

- <code><apihead>re.Pattern.<apiname>split</apiname>(string, maxsplit=0)</apihead></code>
<br><br>
####  re.Pattern.findall

- <code><apihead>re.Pattern.<apiname>findall</apiname>(string[, pos[, endpos]])</apihead></code>
<br><br>
This has the same limitation as [`re.findall`](#refindall). 

####  re.Pattern.sub

- <code><apihead>re.Pattern.<apiname>sub</apiname>(repl, string, count=0)</apihead></code>
<br><br>
####   re.Pattern.subn

- <code><apihead>re.Pattern.<apiname>subn</apiname>(repl, string, count=0)</apihead></code>
<br><br>
####   re.Pattern.flags

- <code><apihead>re.Pattern.<apiname>flags</apiname></apihead></code>
<br><br>
####   re.Pattern.groups

- <code><apihead>re.Pattern.<apiname>groups</apiname></apihead></code>
<br><br>
####   re.Pattern.groupindex

- <code><apihead>re.Pattern.<apiname>groupindex</apiname></apihead></code>
<br><br>
####   re.Pattern.pattern

- <code><apihead>re.Pattern.<apiname>pattern</apiname></apihead></code>
<br><br>
####   re.Match.expand

- <code><apihead>re.Match.<apiname>expand</apiname>(template)</apihead></code>
<br><br>
####   re.Match.group

- <code><apihead>re.Match.<apiname>group</apiname>([group1, ...])</apihead></code>
<br><br>
####   re.Match.\_\_getitem\_\_
- <code><apihead>re.Match.<apiname>\_\_getitem\_\_</apiname>(g)</apihead></code>

####   re.Match.groups

- <code><apihead>re.Match.<apiname>groups</apiname>(default=None)</apihead></code>
<br><br>
####   re.Match.groupdict

- <code><apihead>re.Match.<apiname>groupdict</apiname>(default=None)</apihead></code>
<br><br>
(does not support default=None for groups that did not participate
    in the match)
    
####   re.Match.start

- <code><apihead>re.Match.<apiname>start</apiname>([group])</apihead></code>
<br><br>
####   re.Match.end

- <code><apihead>re.Match.<apiname>end</apiname>([group])</apihead></code>
<br><br>
####   re.Match.span

- <code><apihead>re.Match.<apiname>span</apiname>([group])</apihead></code>
<br><br>
####   re.Match.pos

- <code><apihead>re.Match.<apiname>pos</apiname></apihead></code>
<br><br>
####   re.Match.endpos

- <code><apihead>re.Match.<apiname>endpos</apiname></apihead></code>
<br><br>
####   re.Match.lastindex

- <code><apihead>re.Match.<apiname>lastindex</apiname></apihead></code>
<br><br>
####   re.Match.lastgroup

- <code><apihead>re.Match.<apiname>lastgroup</apiname></apihead></code>
<br><br>
####   re.Match.re

- <code><apihead>re.Match.<apiname>re</apiname></apihead></code>
<br><br>
####   re.Match.string

- <code><apihead>re.Match.<apiname>string</apiname></apihead></code>
<br><br>
## Class Support using `@jitclass`

Bodo supports Python classes using the `@bodo.jitclass` decorator. It
requires type annotation of the fields, as well as distributed
annotation where applicable. For example, the example class below holds
a distributed dataframe and a name filed. Types can either be specified
directly using the imports in the bodo package or can be inferred from
existing types using `bodo.typeof`. The `%%init%%` function is required,
and has to initialize the attributes. In addition, subclasses are not
supported in `jitclass` yet.

!!! warning
    Class support is currently experimental and therefore we recommend
    refactoring computation into regular JIT functions instead if possible.


```py
@bodo.jitclass(
    {
        "df": bodo.typeof(pd.DataFrame({"A": [1], "B": [0.1]})),
        "name": bodo.string_type,
    },
    distributed=["df"],
)

class MyClass:
    def %%init%%(self, n, name):
        self.df = pd.DataFrame({"A": np.arange(n), "B": np.ones(n)})
        self.name = name

    def sum(self):
        return self.df.A.sum()

    @property
    def sum_vals(self):
        return self.df.sum().sum()

    def get_name(self):
        return self.name

    @staticmethod
    def add_one(a):
        return a + 1
```

This JIT class can be used in regular Python code, as well as other Bodo
JIT code.

```py
# From a compiled function
@bodo.jit
def f():
    my_instance = MyClass(32, "my_name_jit")
    print(my_instance.sum())
    print(my_instance.sum_vals)
    print(my_instance.get_name())

f()
```

```console
496
528.0
my_name_jit
```

```py
# From regular Python
my_instance = MyClass(32, "my_name_python")
print(my_instance.sum())
print(my_instance.sum_vals)
print(my_instance.get_name())
print(MyClass.add_one(8))
```

```console
496
528.0
my_name_python
9
```

Bodo's `jitclass` is built on top of Numba's `jitclass` (see Numba
[jitclass](https://numba.pydata.org/numba-doc/dev/user/jitclass.html){target="blank"}
for more details).
