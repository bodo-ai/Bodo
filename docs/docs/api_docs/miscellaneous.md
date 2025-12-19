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
- `re.A`

####   re.ASCII
- `re.ASCII`

####   re.DEBUG
- `re.DEBUG`

####   re.I
- `re.I`

####   re.IGNORECASE
- `re.IGNORECASE`

####   re.L
- `re.L`

####   re.LOCALE
- `re.LOCALE`

####   re.M
- `re.M`

####   re.MULTILINE
- `re.MULTILINE`

####   re.S
- `re.S`

####   re.DOTALL
- `re.DOTALL`

####   re.X
- `re.X`

####   re.VERBOSE
- `re.VERBOSE`

####   re.search
- `re.search(pattern, string, flags=0)`

####   re.match
- `re.match(pattern, string, flags=0)`

####   re.fullmatch
- `re.fullmatch(pattern, string, flags=0)`

####   re.split
- `re.split(pattern, string, maxsplit=0, flags=0)`

####   re.findall
- `re.findall(pattern, string, flags=0)`

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
- `re.sub(pattern, repl, string, count=0, flags=0)`

####  re.subn
- `re.subn(pattern, repl, string, count=0, flags=0)`

####  re.escape
- `re.escape(pattern)`

####  re.purge
- `re.purge`

####  re.Pattern.search
- `re.Pattern.search(string[, pos[, endpos]])`

####  re.Pattern.match
- `re.Pattern.match(string[, pos[, endpos]])`

####  re.Pattern.fullmatch
- `re.Pattern.fullmatch(string[, pos[, endpos]])`

####  re.Pattern.split
- `re.Pattern.split(string, maxsplit=0)`

####  re.Pattern.findall
- `re.Pattern.findall(string[, pos[, endpos]])`

This has the same limitation as [`re.findall`](#refindall). 

####  re.Pattern.sub
- `re.Pattern.sub(repl, string, count=0)`

####   re.Pattern.subn
- `re.Pattern.subn(repl, string, count=0)`

####   re.Pattern.flags
- `re.Pattern.flags`

####   re.Pattern.groups
- `re.Pattern.groups`

####   re.Pattern.groupindex
- `re.Pattern.groupindex`

####   re.Pattern.pattern
- `re.Pattern.pattern`

####   re.Match.expand
- `re.Match.expand(template)`

####   re.Match.group
- `re.Match.group([group1, ...])`

####   re.Match.\_\_getitem\_\_
- <code><apihead>re.Match.<apiname>\_\_getitem\_\_</apiname>(g)</apihead></code>

####   re.Match.groups
- `re.Match.groups(default=None)`

####   re.Match.groupdict
- `re.Match.groupdict(default=None)`

(does not support default=None for groups that did not participate
    in the match)
    
####   re.Match.start
- `re.Match.start([group])`

####   re.Match.end
- `re.Match.end([group])`

####   re.Match.span
- `re.Match.span([group])`

####   re.Match.pos
- `re.Match.pos`

####   re.Match.endpos
- `re.Match.endpos`

####   re.Match.lastindex
- `re.Match.lastindex`

####   re.Match.lastgroup
- `re.Match.lastgroup`

####   re.Match.re
- `re.Match.re`

####   re.Match.string
- `re.Match.string`

## Class Support using `@jitclass`

Bodo supports Python classes using the `@bodo.jitclass` decorator. It
requires type annotation of the fields, as well as distributed
annotation where applicable. For example, the example class below holds
a distributed dataframe and a name filed. Types can either be specified
directly using the imports in the bodo package or can be inferred from
existing types using `bodo.typeof`. The `init` function is required,
and has to initialize the attributes. In addition, subclasses are not
supported in `jitclass` yet.

!!! warning
    Class support is currently experimental and therefore we recommend
    refactoring computation into regular JIT functions instead if possible.


```py
@bodo.jitclass(
    {
        "df": bodo.typeof(pd.DataFrame({"A": [1], "B": [0.1]})),
        "name": bodo.types.string_type,
    },
    distributed=["df"],
)

class MyClass:
    def init(self, n, name):
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
