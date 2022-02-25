# sklearn.preprocessing

## sklearn.preprocessing.LabelEncoder

- ++sklearn.preprocessing.%%LabelEncoder%%++


This class provides LabelEncoder support to encode target labels `y`
with values between 0 and n-classes-1.

### Methods

#### sklearn.preprocessing.LabelEncoder.fit

- ++sklearn.preprocessing.LabelEncoder.%%fit%%(y)++

    ***Supported Arguments***
    <br>
    <br>
    -   `y`: 1d array-like.

#### sklearn.preprocessing.LabelEncoder.fit_transform

- ++sklearn.preprocessing.LabelEncoder.%%fit_transform%%(y)++

    ***Supported Arguments***
    <br>
    <br>
    -   `y`: 1d array-like.


#### sklearn.preprocessing.LabelEncoder.transform

- ++sklearn.preprocessing.LabelEncoder.%%transform%%(y)++

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

++sklearn.preprocessing.%%MinMaxScaler%%++


This class provides MinMax Scaler support to scale your data based on
the range of its features.
 
### Methods
 
#### sklearn.preprocessing.MinMaxScaler.fit

- ++sklearn.preprocessing.MinMaxScaler.%%fit%%(X, y=None)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy array or Pandas Dataframes.

#### sklearn.preprocessing.MinMaxScaler.inverse_transform

- ++sklearn.preprocessing.MinMaxScaler.%%inverse_transform%%(X)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy array or Pandas Dataframes.

#### sklearn.preprocessing.MinMaxScaler.transform

- ++sklearn.preprocessing.MinMaxScaler.%%transform%%(X)++
    
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

++sklearn.preprocessing.%%StandardScaler%%++


This class provides Standard Scaler support to center your data and to
scale it to achieve unit variance.

### Methods

#### sklearn.preprocessing.StandardScaler.fit

- ++sklearn.preprocessing.StandardScaler.%%fit%%(X, y=None, sample_weight=None)++


    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
            distributed)

#### sklearn.preprocessing.StandardScaler.inverse_transform

- ++sklearn.preprocessing.StandardScaler.%%inverse_transform%%(X, copy=None)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `copy`: bool or None.

#### sklearn.preprocessing.StandardScaler.transform

- ++sklearn.preprocessing.StandardScaler.%%transform%%(X, copy=None)++

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