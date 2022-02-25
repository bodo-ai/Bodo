# sklearn.linear_model

## sklearn.linear_model.Lasso

++sklearn.linear_model.%%Lasso%%++


This class provides Lasso regression support.

### Methods

#### sklearn.linear_model.Lasso.fit

- ++sklearn.linear_model.Lasso.%%fit%%(X, y, sample_weight=None, check_input=True)++

    ***Supported Arguments***
    <br>
    <br> 
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
        distributed)

#### sklearn.linear_model.Lasso.predict

- ++sklearn.linear_model.Lasso.%%predict%%(X)++

    ***Supported Arguments***
    <br>
    <br> 
    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.Lasso.score

- ++sklearn.linear_model.Lasso.%%score%%(X, y, sample_weight=None)++

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

++sklearn.linear_model.%%LinearRegression%%++



This class provides linear regression support.

!!! note

    Multilabel targets are not currently supported.

### Methods

#### sklearn.linear_model.LinearRegression.fit

- ++sklearn.linear_model.LinearRegression.%%fit%%(X, y, sample_weight=None)++

    ***Supported Arguments***
    <br>
    <br>     
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
        distributed)

#### sklearn.linear_model.LinearRegression.predict

- ++sklearn.linear_model.LinearRegression.%%predict%%(X)++

    ***Supported Arguments***
    <br>
    <br>     
    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.LinearRegression.score

- ++sklearn.linear_model.LinearRegression.%%score%%(X, y, sample_weight=None)++

    ***Supported Arguments***
    <br>
    <br>     
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array or Pandas Dataframes.
    -   `sample_weight`: Numeric NumPy Array or Pandas Dataframes.

### Attributes

#### sklearn.linear_model.LinearRegression.coef_

- ++sklearn.linear_model.LinearRegression.<apiname>coef\_</apiname>++

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

++sklearn.linear_model.%%LogisticRegression%%++
This class provides logistic regression classifier.

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

- ++sklearn.linear_model.LogisticRegression.%%fit%%(X, y, sample_weight=None)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
         distributed)

#### sklearn.linear_model.LogisticRegression.predict

- ++sklearn.linear_model.LogisticRegression.%%predict%%(X)++

    ***Supported Arguments***
    <br>
    <br>
    -  `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.LogisticRegression.predict_log_proba

- ++sklearn.linear_model.LogisticRegression.%%predict_log_proba%%(X)++

    ***Supported Arguments***
    <br>
    <br>    
    -  `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.LogisticRegression.predict_proba

- ++sklearn.linear_model.LogisticRegression.%%predict_proba%%(X)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    
#### sklearn.linear_model.LogisticRegression.score

- ++sklearn.linear_model.LogisticRegression.%%score%%(X, y, sample_weight=None)++

    ***Supported Arguments***
    <br>
    <br>    
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array or Pandas Dataframes.
    -   `sample_weight`: Numeric NumPy Array or Pandas Dataframes.

### Attributes

#### sklearn.linear_model.LogisticRegression.coef_

- ++sklearn.linear_model.LogisticRegression.<apiname>coef\_</apiname>++

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

++sklearn.linear_model.%%Ridge%%++

This class provides ridge regression support.
    
### Methods
    
#### sklearn.linear_model.Ridge.fit

- ++sklearn.linear_model.Ridge.%%fit%%(X, y, sample_weight=None)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
     distributed)

#### sklearn.linear_model.Ridge.predict

- ++sklearn.linear_model.Ridge.%%predict%%(X)++

    ***Supported Arguments***
    <br>
    <br>
    -  `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.Ridge.score

- ++sklearn.linear_model.Ridge.%%score%%(X, y, sample_weight=None)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array or Pandas Dataframes.
    -   `sample_weight`: Numeric NumPy Array or Pandas Dataframes.

### Attributes

#### sklearn.linear_model.Ridge.coef_

- ++sklearn.linear_model.Ridge.<apiname>coef\_</apiname>++

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

++sklearn.linear_model.SGDClassifier++


This class provides linear classification models with SGD optimization
which allows distributed large-scale learning.

-  Supported loss functions `hinge` and `log`.
- `SGDClassifier(loss='hinge')` is equivalent to [SVM linear classifer](https://scikit-learn.org/0.24/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC){target="blank"}.
- `SGDClassifier(loss='log')` is equivalent to [logistic regression classifer](https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression){target="blank"}.
-  `early_stopping` is not supported yet.

### Methods

#### sklearn.linear_model.SGDClassifier.fit

- ++sklearn.linear_model.SGDClassifier.%%fit%%(X, y, coef_init=None, intercept_init=None, sample_weight=None)++


    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
     distributed)

#### sklearn.linear_model.SGDClassifier.predict

- ++sklearn.linear_model.SGDClassifier.%%predict%%(X)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.SGDClassifier.predict_log_proba

- ++sklearn.linear_model.SGDClassifier.%%predict_log_proba%%(X)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.SGDClassifier.predict_proba

- ++sklearn.linear_model.SGDClassifier.%%predict_proba%%(X)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.SGDClassifier.score

- ++sklearn.linear_model.SGDClassifier.score(X, y, sample_weight=None)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array or Pandas Dataframes.
    -   `sample_weight`: Numeric NumPy Array or Pandas Dataframes.

### Attributes

#### sklearn.linear_model.SGDClassifier.coef_

- ++sklearn.linear_model.SGDClassifier.<apiname>coef\_<apiname>++

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

++sklearn.linear_model.%%SGDRegressor%%++


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

- ++sklearn.linear_model.SGDRegressor.%%fit%%(X, y, coef_init=None, intercept_init=None, sample_weight=None)++

    ***Supported Arguments***
    <br>
    <br>
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array.
    -   `sample_weight`: Numeric NumPy Array (only if data is not
                         distributed)
 
 
#### sklearn.linear_model.SGDRegressor.predict

- ++sklearn.linear_model.SGDRegressor.%%predict%%(X)++

***Supported Arguments***

    -   `X`: NumPy Array or Pandas Dataframes.

#### sklearn.linear_model.SGDRegressor.score

- ++sklearn.linear_model.SGDRegressor.%%score%%(X, y, sample_weight=None)++

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

