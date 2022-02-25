# sklearn.linear_model

## sklearn.linear_model.Lasso

<code><apihead>sklearn.linear_model.<apiname>Lasso</apiname></apihead></code><br><br><br>

This class provides Lasso regression support.

### Methods

#### sklearn.linear_model.Lasso.fit


- <code><apihead>sklearn.linear_model.Lasso.<apiname>fit</apiname>(X, y, sample_weight=None, check_input=True)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br> 
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
        distributed)

#### sklearn.linear_model.Lasso.predict


- <code><apihead>sklearn.linear_model.Lasso.<apiname>predict</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br> 
    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.Lasso.score


- <code><apihead>sklearn.linear_model.Lasso.<apiname>score</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>     
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array or Pandas Dataframes.
    -   `sample_weight`: Numeric NumPy Array or Pandas Dataframes.

### Example Usage

```py 
>>> import bodo
>>> from sklearn.linear_model import Lasso
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(
... n_samples=10,
... n_features=10,
... n_informative=5,
... )
>>> @bodo.jit
... def test_lasso(X, y):
...   scaler = StandardScaler()
...   scaler.fit(X)
...   X = scaler.transform(X)
...   reg = Lasso(alpha=0.1)
...   reg.fit(X, y)
...   ans = reg.predict(X)
...   print(ans)
...   print("score: ", reg.score(X, y))
...
>>> test_lasso(X, y)
[-108.40717491  -92.14977392  -54.82835898  -52.81762142  291.33173703
60.60660979  128.64172956   30.42129155  110.20607814   58.05321319]
score:  0.9999971902794988
```

## sklearn.linear_model.LinearRegression

<code><apihead>sklearn.linear_model.<apiname>LinearRegression</apiname></apihead></code><br><br><br>


This class provides linear regression support.

!!! note

    Multilabel targets are not currently supported.

### Methods

#### sklearn.linear_model.LinearRegression.fit


- <code><apihead>sklearn.linear_model.LinearRegression.<apiname>fit</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>     
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
        distributed)

#### sklearn.linear_model.LinearRegression.predict


- <code><apihead>sklearn.linear_model.LinearRegression.<apiname>predict</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>     
    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.LinearRegression.score


- <code><apihead>sklearn.linear_model.LinearRegression.<apiname>score</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>     
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array or Pandas Dataframes.
    -   `sample_weight`: Numeric NumPy Array or Pandas Dataframes.

### Attributes

#### sklearn.linear_model.LinearRegression.coef_


- <code><apihead>sklearn.linear_model.LinearRegression.<apiname>coef_</apiname></apihead></code>
<br><br>
### Example Usage

```py 
>>> import bodo
>>> from sklearn.linear_model import LinearRegression
>>> import numpy as np
>>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
>>> y = np.dot(X, np.array([1, 2])) + 3
>>> @bodo.jit
... def test_linear_reg(X, y):
...   reg = LinearRegression()
...   reg.fit(X, y)
...   print("score: ", reg.score(X, y))
...   print("coef_: ", reg.coef_)
...   ans = reg.predict(np.array([[3, 5]]))
...   print(ans)
...
>>> test_linear_reg(X, y)
score:  1.0
coef_:  [1. 2.]
[16.]
```

## sklearn.linear_model.LogisticRegression

<code><apihead>sklearn.linear_model.<apiname>LogisticRegression</apiname></apihead></code><br><br><br>This class provides logistic regression classifier.

!!! note

    Bodo uses Stochastic Gradient Descent (SGD) to train linear
    models across multiple nodes in a distributed fashion. This produces
    models that have similar accuracy compared to their corresponding
    sequential version in most cases. To achieve that, it is highly
    recommended to scale your data using `StandardScaler` before training
    and/or testing the model. See scikit-learn for more tips on how to tune
    model parameters for SGD [here](https://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use){target="blank"}.
    

### Methods

#### sklearn.linear_model.LogisticRegression.fit


- <code><apihead>sklearn.linear_model.LogisticRegression.<apiname>fit</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
         distributed)

#### sklearn.linear_model.LogisticRegression.predict


- <code><apihead>sklearn.linear_model.LogisticRegression.<apiname>predict</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -  `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.LogisticRegression.predict_log_proba


- <code><apihead>sklearn.linear_model.LogisticRegression.<apiname>predict_log_proba</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -  `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.LogisticRegression.predict_proba


- <code><apihead>sklearn.linear_model.LogisticRegression.<apiname>predict_proba</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    
#### sklearn.linear_model.LogisticRegression.score


- <code><apihead>sklearn.linear_model.LogisticRegression.<apiname>score</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array or Pandas Dataframes.
    -   `sample_weight`: Numeric NumPy Array or Pandas Dataframes.

### Attributes

#### sklearn.linear_model.LogisticRegression.coef_


- <code><apihead>sklearn.linear_model.LogisticRegression.<apiname>coef_</apiname></apihead></code>
<br><br>
### Example Usage

```py 
>>> import bodo
>>> from sklearn.datasets import make_classification
>>> from sklearn.linear_model import LogisticRegression
>>> X, y = make_classification(
... n_samples=1000,
... n_features=10,
... n_informative=5,
... n_redundant=0,
... random_state=0,
... shuffle=0,
... n_classes=2,
... n_clusters_per_class=1
... )
>>> @bodo.jit
... def test_logistic(X, y):
...   clf = LogisticRegression()
...   clf.fit(X, y)
...   ans = clf.predict(X)
...   print("score: ", clf.score(X, y))
...
>>> test_logistic(X, y)
score:  0.997
```

## sklearn.linear_model.Ridge

<code><apihead>sklearn.linear_model.<apiname>Ridge</apiname></apihead></code><br><br><br>
This class provides ridge regression support.
    
### Methods
    
#### sklearn.linear_model.Ridge.fit


- <code><apihead>sklearn.linear_model.Ridge.<apiname>fit</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
     distributed)

#### sklearn.linear_model.Ridge.predict


- <code><apihead>sklearn.linear_model.Ridge.<apiname>predict</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -  `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.Ridge.score


- <code><apihead>sklearn.linear_model.Ridge.<apiname>score</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array or Pandas Dataframes.
    -   `sample_weight`: Numeric NumPy Array or Pandas Dataframes.

### Attributes

#### sklearn.linear_model.Ridge.coef_


- <code><apihead>sklearn.linear_model.Ridge.<apiname>coef_</apiname></apihead></code>
<br><br>
### Example Usage

```py
>>> import bodo
>>> from sklearn.linear_model import Ridge
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(
... n_samples=1000,
... n_features=10,
... n_informative=5,
... )
>>> @bodo.jit
... def test_ridge(X, y):
...   reg = Ridge(alpha=1.0)
...   reg.fit(X, y)
...   print("score: ", reg.score(X, y))
...   print("coef_: ", reg.coef_)
...
>>> test_ridge(X, y)
score:  0.999998857191076
coef_:  [ 1.07963671e-03  2.35051611e+01  9.46672751e+01  8.01581769e-03
3.66612234e+01  5.82527987e-03  2.60885671e+01 -3.49454103e-03
8.39573884e+01 -7.52605483e-03]
```

## sklearn.linear_model.SGDClassifier

<code><apihead>sklearn.linear_model.SGDClassifier</apihead></code><br><br><br>

This class provides linear classification models with SGD optimization
which allows distributed large-scale learning.

-  Supported loss functions `hinge` and `log`.
- `SGDClassifier(loss='hinge')` is equivalent to [SVM linear classifer](https://scikit-learn.org/0.24/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC){target="blank"}.
- `SGDClassifier(loss='log')` is equivalent to [logistic regression classifer](https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression){target="blank"}.
-  `early_stopping` is not supported yet.

### Methods

#### sklearn.linear_model.SGDClassifier.fit


- <code><apihead>sklearn.linear_model.SGDClassifier.<apiname>fit</apiname>(X, y, coef_init=None, intercept_init=None, sample_weight=None)</apihead></code>
<br><br>

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
     distributed)

#### sklearn.linear_model.SGDClassifier.predict


- <code><apihead>sklearn.linear_model.SGDClassifier.<apiname>predict</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.SGDClassifier.predict_log_proba


- <code><apihead>sklearn.linear_model.SGDClassifier.<apiname>predict_log_proba</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.SGDClassifier.predict_proba


- <code><apihead>sklearn.linear_model.SGDClassifier.<apiname>predict_proba</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.SGDClassifier.score


- <code><apihead>sklearn.linear_model.SGDClassifier.score(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array or Pandas Dataframes.
    -   `sample_weight`: Numeric NumPy Array or Pandas Dataframes.

### Attributes

#### sklearn.linear_model.SGDClassifier.coef_


- <code><apihead>sklearn.linear_model.SGDClassifier.<apiname>coef_<apiname></apihead></code>
<br><br>
### Example Usage

```py 
>>> import bodo
>>> from sklearn.linear_model import SGDClassifier
>>> from sklearn.preprocessing import StandardScaler
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
>>> y = np.array([1, 1, 2, 2])
>>> @bodo.jit
... def test_sgdclassifier(X, y):
...   scaler = StandardScaler()
...   scaler.fit(X)
...   X = scaler.transform(X)
...   clf = SGDClassifier(loss="hinge", penalty="l2")
...   clf.fit(X, y)
...   ans = clf.predict(np.array([[-0.8, -1]]))
...   print(ans)
...   print("coef_: ", clf.coef_)
...
>>> test_sgdclassifier(X, y)
[1]
coef_:  [[6.18236102 9.77517107]]
```

## sklearn.linear_model.SGDRegressor

<code><apihead>sklearn.linear_model.<apiname>SGDRegressor</apiname></apihead></code><br><br><br>

This class provides linear regression models with SGD optimization
which allows distributed large-scale learning.

-   Supported loss function is `squared_error`. 
-  `early_stopping` is not supported yet.

- `SGDRegressor(loss='squared_error', penalty='None')` is equivalent to
[linear regression](https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression){target="blank"}.

- `SGDRegressor(loss='squared_error', penalty='l2')` is equivalent to
[Ridge regression](https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge){target="blank"}.

- `SGDRegressor(loss='squared_error', penalty='l1')` is equivalent to
[Lasso regression](https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso){target="blank"}.

### Methods

#### sklearn.linear_model.SGDRegressor.fit


- <code><apihead>sklearn.linear_model.SGDRegressor.<apiname>fit</apiname>(X, y, coef_init=None, intercept_init=None, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
                         distributed)
 
 
#### sklearn.linear_model.SGDRegressor.predict


- <code><apihead>sklearn.linear_model.SGDRegressor.<apiname>predict</apiname>(X)</apihead></code>
<br><br>
***Supported Arguments***

    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.SGDRegressor.score


- <code><apihead>sklearn.linear_model.SGDRegressor.<apiname>score</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    - `X`: NumPy Array or Pandas Dataframes.
    - `y`: NumPy Array or Pandas Dataframes.
    - `sample_weight`: Numeric NumPy Array or Pandas Dataframes.

### Example Usage

```py
>>> import bodo
>>> from sklearn.linear_model import SGDRegressor
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(
... n_samples=1000,
... n_features=10,
... n_informative=5,
... )
>>> @bodo.jit
... def test_sgd_reg(X, y):
...   scaler = StandardScaler()
...   scaler.fit(X)
...   X = scaler.transform(X)
...   reg = SGDRegressor()
...   reg.fit(X, y)
...   print("score: ", reg.score(X, y))
...
>>> test_sgd_reg(X, y)
0.9999999836265652
```

