# sklearn.ensemble 

## sklearn.ensemble.RandomForestClassifier

<code><apihead>sklearn.ensemble.<apiname>RandomForestClassifier</apiname></apihead></code><br><br><br>

This class provides Random Forest Classifier, an ensemble learning
model, for distributed large-scale learning.

!!! important
    `random_state` value is ignored when running on a multi-node cluster.

### Methods

#### sklearn.ensemble.RandomForestClassifier.fit 


- <code><apihead>sklearn.ensemble.RandomForestClassifier.<apiname>fit</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br> 
    -   `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    -   `y`: NumPy Array
    -   `sample_weight`: Numeric NumPy Array (only if data is not
        distributed)
    
    
#### sklearn.ensemble.RandomForestClassifier.predict

- <code><apihead>sklearn.ensemble.RandomForestClassifier.<apiname>predict</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br> 
    -   `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.

#### sklearn.ensemble.RandomForestClassifier.predict_log_proba


- <code><apihead>sklearn.ensemble.RandomForestClassifier.<apiname>predict_log_proba</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br> 
    -   `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.

#### sklearn.ensemble.RandomForestClassifier.predict_proba


- <code><apihead>sklearn.ensemble.RandomForestClassifier.<apiname>predict_proba</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br> 
    -   `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.

#### sklearn.ensemble.RandomForestClassifier.score


- <code><apihead>sklearn.ensemble.RandomForestClassifier.<apiname>score</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br> 
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array
    -   `sample_weight`: Numeric NumPy Array

### Example Usage

```py
>>> import bodo
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_samples=1000, n_features=4,
...                            n_informative=2, n_redundant=0,
...                            random_state=0, shuffle=False)
>>> @bodo.jit
>>> def test_random_forest_classifier(X, y):
...   clf = RandomForestClassifier(max_depth=2)
...   clf.fit(X, y)
...   ans = clf.predict(np.array([[0, 0, 0, 0]]))
...   print(ans)
...
>>> test_random_forest_classifier(X, y)
[1]
```  
     
     
     
## sklearn.ensemble.RandomForestRegressor

<code><apihead>sklearn.ensemble.<apiname>RandomForestRegressor</apiname></apihead></code><br><br><br>
This class provides Random Forest Regressor, an ensemble learning
model, for distributed large-scale learning.

!!! important
    `random_state` value is ignored when running on a multi-node cluster.

### Methods

#### sklearn.ensemble.RandomForestRegressor.fit


- <code><apihead>sklearn.ensemble.RandomForestRegressor.<apiname>fit</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br> 
    -   `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    -   `y`: NumPy Array
    -   `sample_weight`: Numeric NumPy Array (only if data is not
        distributed)

#### sklearn.ensemble.RandomForestRegressor.predict


- <code><apihead>sklearn.ensemble.RandomForestRegressor.<apiname>predict</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br> 
    -   `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.

#### sklearn.ensemble.RandomForestRegressor.score


- <code><apihead>sklearn.ensemble.RandomForestRegressor.<apiname>score</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br> 
    -   `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    -   `y`: NumPy Array
    -   `sample_weight`: Numeric NumPy Array

### Example Usage

```py
>>> import bodo
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(n_features=4, n_informative=2,
... random_state=0, shuffle=False)
>>> @bodo.jit
>>> def test_random_forest_regressor(X, y):
...   regr = RandomForestRegressor(max_depth=2)
...   regr.fit(X, y)
...   ans = regr.predict(np.array([[0, 0, 0, 0]]))
...   print(ans)
...
>>> test_random_forest_regressor(X, y)
[-6.7933243]
```
