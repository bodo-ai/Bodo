# sklearn.svm

## sklearn.svm.LinearSVC

<code><apihead>sklearn.svm.<apiname>LinearSVC</apiname></apihead></code><br><br><br>

This class provides Linear Support Vector Classification.

### Methods

#### sklearn.svm.LinearSVC.fit


- <code><apihead>sklearn.svm.LinearSVC.<apiname>fit</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
     distributed)

#### sklearn.svm.LinearSVC.predict


- <code><apihead>sklearn.svm.LinearSVC.<apiname>predict</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.svm.LinearSVC.score


- <code><apihead>sklearn.svm.LinearSVC.<apiname>score</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array or Pandas Dataframes.
    -   `sample_weight`: Numeric NumPy Array or Pandas Dataframes.

### Example Usage:

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

