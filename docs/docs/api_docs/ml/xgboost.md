# XGBoost

This page lists the XGBoost (using the Scikit-Learn-like API) classes and functions that Bodo supports natively inside JIT functions.


## Installing XGBoost

You will need to build XGBoost with MPI support from source.
XGBoost version must be ``<= 1.5.1``. Refer to [XGBoost instructions about building requirements](https://xgboost.readthedocs.io/en/stable/build.html#id4){target="blank"} for more details.
Then, build XGBoost with MPI support from source and install it in your Bodo environment as follows:

```shell
git clone --recursive https://github.com/dmlc/xgboost --branch v1.5.1
cd xgboost
mkdir build
cd build
cmake -DRABIT_BUILD_MPI=ON ..
make -j4
cd ../python-package
python setup.py install
```

## `xgboost.XGBClassifier`

This class provides implementation of the scikit-learn API for XGBoost classification with distributed large-scale learning.

###  Methods

#### `xgboost.XGBClassifier.fit`

- <code><apihead>xgboost.XGBClassifier.<apiname>fit</apiname>(X, y, sample_weight=None, base_margin=None, eval_set=None, eval_metric=None, early_stopping_rounds=None, verbose=True, xgb_model=None, sample_weight_eval_set=None, feature_weights=None, callbacks=None)</apihead></code>
<br><br>
  ***Supported Arguments***

    +-----------------------------+-----------------------------------------+
    | argument                    | datatypes                               |
    +=============================+=========================================+
    | ``X``                       |  NumPy Array or Pandas Dataframes       |
    +-----------------------------+-----------------------------------------+
    | ``y``                       |  NumPy Array or Pandas Dataframes       |
    +-----------------------------+-----------------------------------------+

#### `xgboost.XGBClassifier.predict`

- <code><apihead>xgboost.XGBClassifier.<apiname>predict</apiname>(X, output_margin=False, ntree_limit=None, validate_features=True, base_margin=None)</apihead></code>
<br><br>

    ***Supported Arguments***

    +-----------------------------+-----------------------------------------+
    | argument                    | datatypes                               |
    +=============================+=========================================+
    | ``X``                       |  NumPy Array or Pandas Dataframes       |
    +-----------------------------+-----------------------------------------+


#### `xgboost.XGBClassifier.predict_proba`

- <code><apihead>xgboost.XGBClassifier.<apiname>predict_proba</apiname>(X, ntree_limit=None, validate_features=True, base_margin=None)</apihead></code>
<br><br>
    ***Supported Arguments***

    +-----------------------------+-----------------------------------------+
    | argument                    | datatypes                               |
    +=============================+=========================================+
    | ``X``                       |  NumPy Array or Pandas Dataframes       |
    +-----------------------------+-----------------------------------------+


### Attributes

#### `xgboost.XGBClassifier.feature_importances_`

- <code><apihead>xgboost.XGBClassifier.<apiname>feature_importances_</apiname></apihead></code>
<br><br>
###  Example Usage:
```py
>>> import bodo
>>> import xgboost as xgb
>>> import numpy as np
>>> @bodo.jit
>>> def test_xgbc():
...   X = np.random.rand(5, 10)
...   y = np.random.randint(0, 2, 5)
...   clf = xgb.XGBClassifier(
...   booster="gbtree",
...   random_state=0,
...   tree_method="hist",
...   )
...   clf.fit(X, y)
...   print(clf.predict([[1, 2, 3, 4, 5, 6]]))
...   print(clf.feature_importances_)
...
>>> test_xgbc(X, y)
[1]
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```

## `xgboost.XGBRegressor`

This class provides implementation of the scikit-learn API for XGBoost regression with distributed large-scale learning.

###  Methods

#### `xgboost.XGBRegressor.fit`

- <code><apihead>xgboost.XGBRegressor.<apiname>fit</apiname>(X, y, sample_weight=None, base_margin=None, eval_set=None, eval_metric=None, early_stopping_rounds=None, verbose=True, xgb_model=None, sample_weight_eval_set=None, feature_weights=None, callbacks=None)</apihead></code>
<br><br>
  ***Supported Arguments***

    +-----------------------------+-----------------------------------------+
    | argument                    | datatypes                               |
    +=============================+=========================================+
    |``X``                        | NumPy Array                             |
    +-----------------------------+-----------------------------------------+
    |``y``                        | NumPy Array                             |
    +-----------------------------+-----------------------------------------+

#### `xgboost.XGBRegressor.predict`


- <code><apihead> xgboost.XGBRegressor.<apiname>predict</apiname>(X, output_margin=False, ntree_limit=None, validate_features=True, base_margin=None)</apihead></code>
<br><br>
  ***Supported Arguments***

    +-----------------------------+-----------------------------------------+
    | argument                    | datatypes                               |
    +=============================+=========================================+
    |``X``                        | NumPy Array                             |
    +-----------------------------+-----------------------------------------+

###  Attributes

#### `xgboost.XGBRegressor.feature_importances_`


- <code><apihead>xgboost.XGBRegressor.<apiname>feature_importances_</apiname></apihead></code>
<br><br>
###  Example Usage

```py
>>> import bodo
>>> import xgboost as xgb
>>> import numpy as np
>>> np.random.seed(42)
>>> @bodo.jit
>>> def test_xgbc():
...   X = np.random.rand(5, 10)
...   y = np.random.rand(5)
...   clf = xgb.XGBRegressor()
...   clf.fit(X, y)
...   print(clf.predict([[1, 2, 3, 4, 5, 6]]))
...   print(clf.feature_importances_)
...
>>> test_xgbc(X, y)
[0.84368145]
[5.7460850e-01 1.2052832e-04 0.0000000e+00 4.2441860e-01 1.5441242e-04
 6.9795933e-04 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]
```