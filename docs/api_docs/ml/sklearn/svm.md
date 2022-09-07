# sklearn.svm

## sklearn.svm.LinearSVC

++sklearn.svm.%%LinearSVC%%++


This class provides Linear Support Vector Classification.

### Methods

#### sklearn.svm.LinearSVC.fit

- ++sklearn.svm.LinearSVC.%%fit%%(X, y, sample_weight=None)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
     distributed)

#### sklearn.svm.LinearSVC.predict

- ++sklearn.svm.LinearSVC.%%predict%%(X)++

    ***Supported Arguments***
    <br>
    <br>    
    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.svm.LinearSVC.score

- ++sklearn.svm.LinearSVC.%%score%%(X, y, sample_weight=None)++

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

