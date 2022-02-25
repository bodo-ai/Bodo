# sklearn.preprocessing

## sklearn.preprocessing.LabelEncoder


- <code><apihead>sklearn.preprocessing.<apiname>LabelEncoder</apiname></apihead></code>
<br><br>

This class provides LabelEncoder support to encode target labels `y`
with values between 0 and n-classes-1.

### Methods

#### sklearn.preprocessing.LabelEncoder.fit


- <code><apihead>sklearn.preprocessing.LabelEncoder.<apiname>fit</apiname>(y)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `y`: 1d array-like.

#### sklearn.preprocessing.LabelEncoder.fit_transform


- <code><apihead>sklearn.preprocessing.LabelEncoder.<apiname>fit_transform</apiname>(y)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `y`: 1d array-like.


#### sklearn.preprocessing.LabelEncoder.transform


- <code><apihead>sklearn.preprocessing.LabelEncoder.<apiname>transform</apiname>(y)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `y`: 1d array-like.

### Example Usage

```py
>>> import bodo
>>> import numpy as np
>>> from sklearn.preprocessing import LabelEncoder
>>> @bodo.jit
... def test_le():
...   le = LabelEncoder()
...   le.fit([1, 2, 2, 6])
...   print(le.transform([1, 1, 2, 6]))
...
>>> test_le()
[0 0 1 2]
```


## sklearn.preprocessing.MinMaxScaler

<code><apihead>sklearn.preprocessing.<apiname>MinMaxScaler</apiname></apihead></code><br><br><br>

This class provides MinMax Scaler support to scale your data based on
the range of its features.
 
### Methods
 
#### sklearn.preprocessing.MinMaxScaler.fit


- <code><apihead>sklearn.preprocessing.MinMaxScaler.<apiname>fit</apiname>(X, y=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy array or Pandas Dataframes.

#### sklearn.preprocessing.MinMaxScaler.inverse_transform


- <code><apihead>sklearn.preprocessing.MinMaxScaler.<apiname>inverse_transform</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy array or Pandas Dataframes.

#### sklearn.preprocessing.MinMaxScaler.transform


- <code><apihead>sklearn.preprocessing.MinMaxScaler.<apiname>transform</apiname>(X)</apihead></code>
<br><br>    
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy array or Pandas Dataframes.

### Example Usage

```py
>>> import bodo
>>> import numpy as np
>>> from sklearn.preprocessing import MinMaxScaler
>>> data = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
>>> @bodo.jit
... def test_minmax(data):
...   scaler = MinMaxScaler()
...   scaler.fit(data)
...   print(scaler.transform(data))
...
>>> test_minmax(data)
[[0.   0.  ]
 [0.25 0.25]
 [0.5  0.5 ]
 [1.   1.  ]]
```   

## sklearn.preprocessing.StandardScaler

<code><apihead>sklearn.preprocessing.<apiname>StandardScaler</apiname></apihead></code><br><br><br>

This class provides Standard Scaler support to center your data and to
scale it to achieve unit variance.

### Methods

#### sklearn.preprocessing.StandardScaler.fit


- <code><apihead>sklearn.preprocessing.StandardScaler.<apiname>fit</apiname>(X, y=None, sample_weight=None)</apihead></code>
<br><br>

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
            distributed)

#### sklearn.preprocessing.StandardScaler.inverse_transform


- <code><apihead>sklearn.preprocessing.StandardScaler.<apiname>inverse_transform</apiname>(X, copy=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `copy`: bool or None.

#### sklearn.preprocessing.StandardScaler.transform


- <code><apihead>sklearn.preprocessing.StandardScaler.<apiname>transform</apiname>(X, copy=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `copy`: bool or None.

### Example Usage

```py
>>> import bodo
>>> import numpy as np
>>> from sklearn.svm import LinearSVC
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_features=4, random_state=0)
>>> @bodo.jit
... def test_linearsvc(X, y):
...   scaler = StandardScaler()
...   scaler.fit(X)
...   X = scaler.transform(X)
...   clf = LinearSVC()
...   clf.fit(X, y)
...   ans = clf.predict(np.array([[0, 0, 0, 0]]))
...   print(ans)
...
>>> test_linearsvc(X, y)
[1]
```                   