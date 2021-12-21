.. _ml:

Machine Learning
================

Bodo natively supports use of scikit-learn and XGBoost libraries with large-scale distributed data inside ``bodo.jit`` decorated functions.

Scikit-learn
------------

Prerequisites
#############

Bodo supports ``scikit-learn`` versions ``0.24.2`` and below.

Installing scikit-learn
#######################

Install scikit-learn in your Bodo environment::

   conda install scikit-learn=0.24.2 -c conda-forge

JIT Supported
#############

Below is the list of scikit-learn classes and functions that Bodo supports natively inside JIT functions.

sklearn.cluster: Clustering
***************************

:class:`sklearn.cluster.KMeans`

  This class provides K-Means clustering model.


  .. note::
    Currently, this model works by gathering all the data in a single node and then generating K-Means model.
    Make sure you have enough memory on the first node in your `hostfile`.

**Methods:**

  * :any:`sklearn.cluster.KMeans.fit` ``(X, y=None, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    * ``sample_weight``: Numeric NumPy Array

    .. note::
      Bodo ignores ``y``, which is consistent with scikit-learn.

  * :any:`sklearn.cluster.KMeans.predict` ``(X, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    * ``sample_weight``: Numeric NumPy Array

  * :any:`sklearn.cluster.KMeans.score` ``X, y=None, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    * ``sample_weight``: Numeric NumPy Array

    .. note::
      Bodo ignores ``y``, which is consistent with scikit-learn.

  * :any:`sklearn.cluster.KMeans.transform` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array, Pandas Dataframes, or CSR sparse matrix.


Example Usage:
    >>> import bodo
    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> @bodo.jit
    >>> def test_kmeans(X):
    ...   kmeans = KMeans(n_clusters=2)
    ...   kmeans.fit(X)
    ...   ans = kmeans.predict([[0, 0], [12, 3]])
    ...   print(ans)
    ... 
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> test_kmeans(X)
    [1 0]


sklearn.ensemble: Ensemble Methods
**********************************

:class:`sklearn.ensemble.RandomForestClassifier`

  This class provides Random Forest Classifier, an ensemble learning model, for distributed large-scale learning.

  .. note::
    ``random_state`` value is ignored when running on a multi-node cluster.

**Methods:**

  * :any:`sklearn.ensemble.RandomForestClassifier.fit` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    * ``y``: NumPy Array
    * ``sample_weight``: Numeric NumPy Array

  * :any:`sklearn.ensemble.RandomForestClassifier.predict` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array, Pandas Dataframes, or CSR sparse matrix.

  * :any:`sklearn.ensemble.RandomForestClassifier.predict_proba` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array, Pandas Dataframes, or CSR sparse matrix.

  * :any:`sklearn.ensemble.RandomForestClassifier.predict_log_proba` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array, Pandas Dataframes, or CSR sparse matrix.

  * :any:`sklearn.ensemble.RandomForestClassifier.score` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``: NumPy Array
    * ``sample_weight``: Numeric NumPy Array

Example Usage:
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

:class:`sklearn.ensemble.RandomForestRegressor`

  This class provides Random Forest Regressor, an ensemble learning model, for distributed large-scale learning.
    
  .. note::
    ``random_state`` value is ignored when running on a multi-node cluster.

**Methods:**

  * :any:`sklearn.ensemble.RandomForestRegressor.fit` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    * ``y``: NumPy Array
    * ``sample_weight``: Numeric NumPy Array

  * :any:`sklearn.ensemble.RandomForestRegressor.predict` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array, Pandas Dataframes, or CSR sparse matrix.

  * :any:`sklearn.ensemble.RandomForestRegressor.score` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    * ``y``: NumPy Array
    * ``sample_weight``: Numeric NumPy Array

Example Usage:
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

sklearn.feature_extraction: Feature Extraction
**********************************************

From text
~~~~~~~~~

:class:`sklearn.feature_extraction.text.CountVectorizer`

  This class provides CountVectorizer support to convert a collection of text documents to a matrix of token counts.

**Methods:**
  * :any:`sklearn.feature_extraction.text.CountVectorizer.fit_transform` ``(raw_documents, y=None)``

  Supported Arguments:
    * ``X``: iterables ( list, tuple, or NumPy Array, or Pandas Series that contains string)

    .. note::
      Bodo ignores ``y``, which is consistent with scikit-learn.

  * :any:`sklearn.feature_extraction.text.CountVectorizer.get_feature_names` ``()``

Example Usage:
    >>> import bodo
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> corpus = [
    ... 'This is the first document.',
    ... 'This document is the second document.',
    ... 'And this is the third one.',
    ... 'Is this the first document?',
    ... ]
    >>> @bodo.jit
    >>> def test_count_vectorizer(corpus):
    >>>   vectorizer = CountVectorizer()
    >>>   X = vectorizer.fit_transform(corpus)
    >>>   print(vectorizer.get_feature_names())
    ... 
    >>> test_count_vectorizer(corpus)
    ['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']


:class:`sklearn.feature_extraction.text.HashingVectorizer`

  This class provides HashingVectorizer support to convert a collection of text documents to a matrix of token occurrences.

**Methods:**
  * :any:`sklearn.feature_extraction.text.HashingVectorizer.fit_transform` ``(X, y=None)``

  Supported Arguments:
    * ``X``: iterables ( list, tuple, or NumPy Array, or Pandas Series that contains string)

    .. note::
      Bodo ignores ``y``, which is consistent with scikit-learn.

Example Usage:
    >>> import bodo
    >>> from sklearn.feature_extraction.text import HashingVectorizer 
    >>> corpus = [
    ... 'This is the first document.',
    ... 'This document is the second document.',
    ... 'And this is the third one.',
    ... 'Is this the first document?',
    ... ]
    >>> @bodo.jit
    >>> def test_hashing_vectorizer(corpus):
    >>>   vectorizer = HashingVectorizer(n_features=2**4)
    >>>   X = vectorizer.fit_transform(corpus)
    >>>   print(X.shape)
    ... 
    >>> test_hashing_vectorizer(corpus)
    (4, 16)

sklearn.linear_model: Linear Models
***********************************

Linear Classifiers
~~~~~~~~~~~~~~~~~~

:class:`sklearn.linear_model.LogisticRegression` :sup:`*`

  This class provides logistic regression classifier.

**Methods:**

  * :any:`sklearn.linear_model.LogisticRegression.fit` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array.

  * :any:`sklearn.linear_model.LogisticRegression.predict` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.linear_model.LogisticRegression.predict_proba` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.linear_model.LogisticRegression.predict_log_proba` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.linear_model.LogisticRegression.score` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array or Pandas Dataframes.
    * ``sample_weight``:  Numeric NumPy Array or Pandas Dataframes.


**Attributes:**

* :attr:`sklearn.linear_model.LogisticRegression.coef_`

Example Usage:
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

:class:`sklearn.linear_model.SGDClassifier`

  This class provides linear classification models with SGD optimization which allows distributed large-scale learning.

  ``SGDClassifier(loss='hinge')`` is equivalent to `SVM linear classifer <https://scikit-learn.org/0.24/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC>`_. 

  ``SGDClassifier(loss='log')`` is equivalent to `logistic regression classifer <https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression>`_.

    * Supported loss functions ``hinge`` and ``log``.
    * ``early_stopping`` is not supported yet.

**Methods:**

  * :any:`sklearn.linear_model.SGDClassifier.fit` ``(X, y, coef_init=None, intercept_init=None, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array.

  * :any:`sklearn.linear_model.SGDClassifier.predict` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.linear_model.SGDClassifier.predict_proba` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.linear_model.SGDClassifier.predict_log_proba` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.linear_model.SGDClassifier.score` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array or Pandas Dataframes.
    * ``sample_weight``:  Numeric NumPy Array or Pandas Dataframes.

**Attributes:**

* :attr:`sklearn.linear_model.SGDClassifier.coef_`

Example Usage:
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

Classical Linear Regressors 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`sklearn.linear_model.LinearRegression` :sup:`*`

  This class provides linear regression support.

  .. note:: Multilabel targets are not currently supported.

**Methods:**

  * :any:`sklearn.linear_model.LinearRegression.fit` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array.

  * :any:`sklearn.linear_model.LinearRegression.predict` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.linear_model.LinearRegression.score` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array or Pandas Dataframes.
    * ``sample_weight``:  Numeric NumPy Array or Pandas Dataframes.

**Attributes:**

* :attr:`sklearn.linear_model.LinearRegression.coef_`

Example Usage:
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

:class:`sklearn.linear_model.Ridge` :sup:`*`

  This class provides ridge regression support.

**Methods:**

  * :any:`sklearn.linear_model.Ridge.fit` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array.

  * :any:`sklearn.linear_model.Ridge.predict` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.linear_model.Ridge.score` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array or Pandas Dataframes.
    * ``sample_weight``:  Numeric NumPy Array or Pandas Dataframes.


**Attributes:**

* :attr:`sklearn.linear_model.Ridge.coef_`

Example Usage:
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

:class:`sklearn.linear_model.SGDRegressor`

  This class provides linear regression models with SGD optimization which allows distributed large-scale learning.

  ``SGDRegressor(loss='squared_loss', penalty='None')`` is equivalent to `linear regression <https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression>`_. 

  ``SGDRegressor(loss='squared_loss', penalty='l2')`` is equivalent to `Ridge regression <https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge>`_. 

  ``SGDRegressor(loss='squared_loss', penalty='l1')`` is equivalent to `Lasso regression <https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso>`_. 

    * Supported loss function is ``squared_loss``
    * ``early_stopping`` is not supported yet.

**Methods:**

  * :any:`sklearn.linear_model.SGDRegressor.fit` ``(X, y, coef_init=None, intercept_init=None, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array.

  * :any:`sklearn.linear_model.SGDRegressor.predict` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.linear_model.SGDRegressor.score` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array or Pandas Dataframes.
    * ``sample_weight``:  Numeric NumPy Array or Pandas Dataframes.

Example Usage:
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

Regressors with variable selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`sklearn.linear_model.Lasso` :sup:`*`

  This class provides Lasso regression support.

**Methods:**

  * :any:`sklearn.linear_model.Lasso.fit` ``(X, y, sample_weight=None, check_input=True)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array.

  * :any:`sklearn.linear_model.Lasso.predict` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.linear_model.Lasso.score` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array or Pandas Dataframes.
    * ``sample_weight``:  Numeric NumPy Array or Pandas Dataframes.


Example Usage:
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

.. note::
  **\*** Bodo uses Stochastic Gradient Descent (SGD) to train linear models across multiple nodes in a distributed fashion. 
  This produces models that have similar accuracy compared to their corresponding sequential version in most cases.
  To achieve that, it is highly recommended to scale your data using ``StandardScaler`` before training and/or testing the model.
  See scikit-learn for more tips on how to tune model parameters for SGD `here <https://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use>`_.

sklearn.metrics: Metrics
************************

Classification metrics
~~~~~~~~~~~~~~~~~~~~~~

* :func:`sklearn.metrics.accuracy_score` ``(y_true, y_pred, normalize=True, sample_weight=None)``

Supported Arguments:
    * ``y_true``:  1d array-like.
    * ``y_pred``:  1d array-like.
    * ``normalize``:  bool.
    * ``sample_weight``: 1d numeric array-like or None.

    ``y_true``, ``y_pred``, and ``sample_weight`` (if provided) must be of same length.

Example Usage:
    >>> import bodo
    >>> import numpy as np
    >>> from sklearn.metrics import accuracy_score
    >>> y_pred = np.array([0, 2, 1, 3])
    >>> y_true = np.array([0, 1, 2, 3])
    >>> @bodo.jit
    >>> def test_accuracy_score(y_true, y_pred):
    ...   print(accuracy_score(y_true, y_pred))
    >>> test_accuracy_score(y_true, y_pred)
    0.5

* :func:`sklearn.metrics.confusion_matrix` ``(y_true, y_pred, labels=None, sample_weight=None, normalize=None)``

  Supported Arguments:
    * ``y_true``:  1d array-like.
    * ``y_pred``:  1d array-like.
    * ``labels``:  1d array-like.
    * ``sample_weight``: 1d numeric array-like or None. 
    * ``normalize``:  Must be one of ``'true'``, ``'pred'``, ``'all'``,  or None

    ``y_true``, ``y_pred``, and ``sample_weight`` (if provided) must be of same length.

  Example Usage:
    >>> import bodo
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> @bodo.jit
    >>> def test_confusion_matirx(y_true, y_pred):
    ...   print(confusion_matrix(y_true, y_pred))
    >>> test_confusion_matrix(y_true, y_pred)
    [[2 0 0]
    [0 0 1]
    [1 0 2]]

* :func:`sklearn.metrics.f1_score` ``(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')``

  Supported Arguments:
    * ``y_true``:  1d array-like.
    * ``y_pred``:  1d array-like.
    * ``average``: Must be one of ``'micro'``, ``'macro'``, ``'samples'``, ``'weighted'``, ``'binary'``, or None.

    ``y_true`` and  ``y_pred`` must be of same length.

  Example Usage:
    >>> import bodo
    >>> from sklearn.metrics import f1_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> @bodo.jit
    >>> def test_f1_score(y_true, y_pred):
    ...   print(f1_score(y_true, y_pred, average='macro'))
    >>> test_f1_score(y_true, y_pred)
    0.26666666666666666
* :func:`sklearn.metrics.precision_score` ``(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')``

  Supported Arguments:
    * ``y_true``:  1d array-like.
    * ``y_pred``:  1d array-like.
    * ``average``: Must be one of ``'micro'``, ``'macro'``, ``'samples'``, ``'weighted'``, ``'binary'``, or None.

    ``y_true`` and  ``y_pred`` must be of same length.

  Example Usage:
    >>> import bodo
    >>> from sklearn.metrics import precision_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> @bodo.jit
    >>> def test_precision_score(y_true, y_pred):
    ...   print(precision_score(y_true, y_pred, average='macro'))
    >>> test_precision_score(y_true, y_pred)
    0.2222222222222222

* :func:`sklearn.metrics.recall_score` ``(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')``

  Supported Arguments:
    * ``y_true``:  1d array-like.
    * ``y_pred``:  1d array-like.
    * ``average``: Must be one of ``'micro'``, ``'macro'``, ``'samples'``, ``'weighted'``, ``'binary'``, or None.

    ``y_true`` and  ``y_pred`` must be of same length.

  Example Usage:
    >>> import bodo
    >>> from sklearn.metrics import recall_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> @bodo.jit
    >>> def test_recall_score(y_true, y_pred):
    ...   print(recall_score(y_true, y_pred, average='macro'))
    >>> test_recall_score(y_true, y_pred)
    0.3333333333333333



Regression metrics
~~~~~~~~~~~~~~~~~~

* :func:`sklearn.metrics.mean_absolute_error` ``(y_true, y_pred, sample_weight=None, multioutput='uniform_average')``

  Supported Arguments:
    * ``y_true``:  NumPy array.
    * ``y_pred``:  NumPy array.
    * ``sample_weight``:  Numeric NumPy array or None.
    * ``multioutput``: Must be one of ``'raw_values'``, ``'uniform_average'``, or array-like.

    ``y_true``, ``y_pred``, and ``sample_weight`` (if provided) must be of same length.

  Example Usage:
    >>> import bodo
    >>> import numpy as np
    >>> from sklearn.metrics import mean_absolute_error
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> @bodo.jit
    >>> def test_mean_absolute_error(y_true, y_pred):
    ...   print(mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7]))
    >>> test_mean_absolute_error(y_true, y_pred)
    0.85

* :func:`sklearn.metrics.mean_squared_error` ``(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True)``

  Supported Arguments:
    * ``y_true``:  NumPy array.
    * ``y_pred``:  NumPy array.
    * ``sample_weight``:  Numeric NumPy array or None.
    * ``multioutput``: Must be one of ``'raw_values'``, ``'uniform_average'``, or array-like.

    ``y_true``, ``y_pred``, and ``sample_weight`` (if provided) must be of same length.

  Example Usage:
    >>> import bodo
    >>> import numpy as np
    >>> from sklearn.metrics import mean_squared_error 
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> @bodo.jit
    >>> def test_mean_squared_error(y_true, y_pred):
    ...   print(mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7]))
    >>> test_mean_squared_error(y_true, y_pred)
    0.825

* :func:`sklearn.metrics.r2_score` ``(y_true, y_pred, sample_weight=None, multioutput='uniform_average')``

  Supported Arguments:
    * ``y_true``:  NumPy array.
    * ``y_pred``:  NumPy array.
    * ``sample_weight``:  Numeric NumPy array or None.
    * ``multioutput``: Must be one of ``'raw_values'``, ``'uniform_average'``, ``'variance_weighted'``, None, or array-like.

    ``y_true``, ``y_pred``, and ``sample_weight`` (if provided) must be of same length.

  Example Usage:
    >>> import bodo
    >>> import numpy as np
    >>> from sklearn.metrics import r2_score
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> @bodo.jit
    >>> def test_r2_score(y_true, y_pred):
    ...   print(r2_score(y_true, y_pred, multioutput=[0.3, 0.7]))
    >>> test_r2_score(y_true, y_pred)
    0.9253456221198156

sklearn.model_selection: Model Selection
****************************************

Splitter Functions
~~~~~~~~~~~~~~~~~~

* :func:`sklearn.model_selection.train_test_split` ``(X, y, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)``

  Supported Arguments:
    * ``X``: NumPy array or Pandas Dataframes. 
    * ``y``: NumPy array or Pandas Dataframes. 
    * ``train_size``: float between 0.0 and 1.0 or ``None`` only.
    * ``test_size``: float between 0.0 and 1.0 or ``None`` only.
    * ``random_state``: int, RandomState, or None.
    * ``shuffle``: bool.

  Example Usage:
    >>> import bodo
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> @bodo.jit
    >>> def test_split():
    ...   X, y = np.arange(10).reshape(5, 2), np.arange(5)
    ...   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
    ...   print(X_train)
    ...   print(y_train)
    X_train:  [[4 5]
    [6 7]
    [8 9]]
    y_train:  [2 3 4]
    X_test:  [[2 3]
    [0 1]]
    y_test:  [1 0]
    

sklearn.naive_bayes: Naive Bayes
********************************

:class:`sklearn.naive_bayes.MultinomialNB`

  This class provides Naive Bayes classifier for multinomial models with distributed large-scale learning.

**Methods:**

  * :any:`sklearn.naive_bayes.MultinomialNB.fit` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.naive_bayes.MultinomialNB.predict` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.naive_bayes.MultinomialNB.score` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array or Pandas Dataframes.
    * ``sample_weight``:  Numeric NumPy Array or Pandas Dataframes.

Example Usage:
    >>> import bodo
    >>> import numpy as np
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> X_test = rng.randint(5, size=(1, 100))
    >>> @bodo.jit
    ... def test_mnb(X, y, X_test):
    ...   clf = MultinomialNB()
    ...   clf.fit(X, y)
    ...   ans = clf.predict(X_test)
    ...   print(ans)
    ... 
    >>> test_mnb(X, y, X_test)
    [5]

sklearn.preprocessing: Preprocessing and Normalization
******************************************************

:class:`sklearn.preprocessing.LabelEncoder`

  This class provides LabelEncoder support to encode target labels (y) with values between 0 and n-classes-1.

**Methods:**

  * :any:`sklearn.preprocessing.LabelEncoder.fit` ``(y)``

  Supported Arguments:
    * ``y``:  1d array-like.

  * :any:`sklearn.preprocessing.LabelEncoder.fit_transform` ``(y)``

  Supported Arguments:
    * ``y``:  1d array-like.

  * :any:`sklearn.preprocessing.LabelEncoder.transform` ``(y)``

  Supported Arguments:
    * ``y``:  1d array-like.

Example Usage:
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

:class:`sklearn.preprocessing.MinMaxScaler`

  This class provides MinMax Scaler support to scale your data based on the range of its features.

**Methods:**

  * :any:`sklearn.preprocessing.MinMaxScaler.fit` ``(X, y=None)``

  Supported Arguments:
    * ``X``:  NumPy array or Pandas Dataframes.

  * :any:`sklearn.preprocessing.MinMaxScaler.inverse_transform` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy array or Pandas Dataframes.

  * :any:`sklearn.preprocessing.MinMaxScaler.transform` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy array or Pandas Dataframes.

Example Usage:
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

:class:`sklearn.preprocessing.StandardScaler`

  This class provides Standard Scaler support to center your data and to scale it to achieve unit variance.

**Methods:**

  * :any:`sklearn.preprocessing.StandardScaler.fit` ``(X, y=None, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array.

  * :any:`sklearn.preprocessing.StandardScaler.inverse_transform` ``(X, copy=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``copy``: bool or None.

  * :any:`sklearn.preprocessing.StandardScaler.transform` ``(X, copy=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``copy``: bool or None.

Example Usage:
    >>> import bodo
    >>> import numpy as np
    >>> from sklearn.preprocessing import StandardScaler
    >>> data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    >>> @bodo.jit
    ... def test_sscaler(data):
    ...   scaler = StandardScaler()
    ...   scaler.fit(data)
    ...   print(scaler.transform(data))
    ... 
    >>> test_sscaler(data)
    [[-1. -1.]
     [-1. -1.]
     [ 1.  1.]
     [ 1.  1.]]

sklearn.svm: Support Vector Machines
************************************

Estimators
~~~~~~~~~~

:class:`sklearn.svm.LinearSVC` :sup:`*`

  This class provides Linear Support Vector Classification.

**Methods:**

  * :any:`sklearn.svm.LinearSVC.fit` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array.

  * :any:`sklearn.svm.LinearSVC.predict` ``(X)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`sklearn.svm.LinearSVC.score` ``(X, y, sample_weight=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array or Pandas Dataframes.
    * ``sample_weight``:  Numeric NumPy Array or Pandas Dataframes.

Example Usage:
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


XGBoost
-------

Prerequisites
#############

You will need to build XGBoost with MPI support from source.
XGBoost version must be ``<= 1.5.1``.


Installing xgboost
##################

Refer to `XGBoost instructions about building requirement <https://xgboost.readthedocs.io/en/stable/build.html#id4>`_. 

Then, build XGBoost with MPI support from source and install it in your Bodo environment as follows::

  git clone --recursive https://github.com/dmlc/xgboost --branch v1.5.1
  cd xgboost
  mkdir build
  cd build
  cmake -DRABIT_BUILD_MPI=ON ..
  make -j4
  cd ../python-package
  python setup.py install

JIT Supported
#############

Below is the list of XGBoost (using the Scikit-Learn-like API) classes and functions that Bodo supports natively inside JIT functions.

XGBClassifier
*****************

:class:`xgboost.XGBClassifier`

  This class provides implementation of the scikit-learn API for XGBoost classification with distributed large-scale learning.

**Methods:**

  * :any:`xgboost.XGBClassifier.fit` ``(X, y, sample_weight=None, base_margin=None, eval_set=None, eval_metric=None, early_stopping_rounds=None, verbose=True, xgb_model=None, sample_weight_eval_set=None, feature_weights=None, callbacks=None)``

  ..
    COMMENT: In theory, we support all but to be on the safe side, until more testing is done. 

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.
    * ``y``:  NumPy Array or Pandas Dataframes.

  * :any:`xgboost.XGBClassifier.predict` ``(X, output_margin=False, ntree_limit=None, validate_features=True, base_margin=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

  * :any:`xgboost.XGBClassifier.predict_proba` ``(X, ntree_limit=None, validate_features=True, base_margin=None)``

  Supported Arguments:
    * ``X``:  NumPy Array or Pandas Dataframes.

**Attributes:**

* :attr:`xgboost.XGBClassifier.feature_importances_`

Example Usage:
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

XGBRegressor
*****************

:class:`xgboost.XGBRegressor`

  This class provides implementation of the scikit-learn API for XGBoost regression with distributed large-scale learning.

**Methods:**

  * :any:`xgboost.XGBRegressor.fit` ``(X, y, sample_weight=None, base_margin=None, eval_set=None, eval_metric=None, early_stopping_rounds=None, verbose=True, xgb_model=None, sample_weight_eval_set=None, feature_weights=None, callbacks=None)``

  .. COMMENT: In theory, we support all but to be on the safe side, until more testing is done.

  Supported Arguments:
    * ``X``:  NumPy Array.
    * ``y``:  NumPy Array.

  * :any:`xgboost.XGBRegressor.predict` ``(X, output_margin=False, ntree_limit=None, validate_features=True, base_margin=None)``

  Supported Arguments:
    * ``X``:  NumPy Array.

**Attributes:**

* :attr:`xgboost.XGBRegressor.feature_importances_`

Example Usage:
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