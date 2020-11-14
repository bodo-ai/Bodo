.. _sklearn:

Supported Scikit-learn
----------------------

Below is the list of scikit-learn classes and functions that Bodo supports natively inside JIT functions.
This list will expand regularly as we add support for more APIs.
Optional arguments are not supported unless specified.

Linear Classifiers
~~~~~~~~~~~~~~~~~~

:class:`sklearn.linear_model.LogisticRegression`

  This class provides logistic regression classifier.

Methods:

  * :any:`sklearn.linear_model.LogisticRegression.fit`
  * :any:`sklearn.linear_model.LogisticRegression.predict`
  * :any:`sklearn.linear_model.LogisticRegression.score`

:class:`sklearn.linear_model.SGDClassifier`

  This class provides linear classification models with SGD optimization which allows distributed large-scale learning.

  ``SGDClassifier(loss='hinge')`` is equivalent to SVM linear classifer. 

  ``SGDClassifier(loss='log')`` is equivalent to logistic regression classifer. 

    * Supported loss functions ``hinge`` and ``log``.
    * ``early_stopping`` is not supported yet.

Methods:

  * :any:`sklearn.linear_model.SGDClassifier.fit`
  * :any:`sklearn.linear_model.SGDClassifier.predict`
  * :any:`sklearn.linear_model.SGDClassifier.score`

:class:`sklearn.svm.LinearSVC`

  This class provides Linear Support Vector Classification.

Methods:

  * :any:`sklearn.svm.LinearSVC.fit`
  * :any:`sklearn.svm.LinearSVC.predict`
  * :any:`sklearn.svm.LinearSVC.score`

Linear Regressors 
~~~~~~~~~~~~~~~~~

:class:`sklearn.linear_model.LinearRegression`

  This class provides linear regression support.
  Note: Multilabel targets are not currently supported.

Methods:

  * :any:`sklearn.linear_model.LinearRegression.fit`
  * :any:`sklearn.linear_model.LinearRegression.predict`
  * :any:`sklearn.linear_model.LinearRegression.score`

:class:`sklearn.linear_model.Ridge`

  This class provides ridge regression support.

Methods:

  * :any:`sklearn.linear_model.Ridge.fit`
  * :any:`sklearn.linear_model.Ridge.predict`
  * :any:`sklearn.linear_model.Ridge.score`

:class:`sklearn.linear_model.SGDRegressor`

  This class provides linear regression models with SGD optimization which allows distributed large-scale learning.

  ``SGDRegressor(loss='squared_loss', penalty='None')`` is equivalent to linear regression. 

  ``SGDRegressor(loss='squared_loss', penalty='l2')`` is equivalent to Ridge regression. 

  ``SGDRegressor(loss='squared_loss', penalty='l1')`` is equivalent to Lasso regression. 

    * Supported loss function is ``squared_loss``
    * ``early_stopping`` is not supported yet.

Methods:

  * :any:`sklearn.linear_model.SGDRegressor.fit`
  * :any:`sklearn.linear_model.SGDRegressor.predict`
  * :any:`sklearn.linear_model.SGDRegressor.score`


:class:`sklearn.linear_model.Lasso`

  This class provides Lasso regression support.

Methods:

  * :any:`sklearn.linear_model.Lasso.fit`
  * :any:`sklearn.linear_model.Lasso.predict`
  * :any:`sklearn.linear_model.Lasso.score`

Clustering
~~~~~~~~~~

:class:`sklearn.cluster.KMeans`

  This class provides K-Means clustering models which allows distributed large-scale unsupervised learning.

Methods:

  * :any:`sklearn.cluster.KMeans.fit`
  * :any:`sklearn.cluster.KMeans.predict`
  * :any:`sklearn.cluster.KMeans.score`
  * :any:`sklearn.cluster.KMeans.transform`

Ensemble Methods
~~~~~~~~~~~~~~~~

:class:`sklearn.ensemble.RandomForestClassifier`

  This class provides Random Forest Classifier, an ensemble learning model, for distributed large-scale learning.

Methods:

  * :any:`sklearn.ensemble.RandomForestClassifier.fit`
  * :any:`sklearn.ensemble.RandomForestClassifier.predict`
  * :any:`sklearn.ensemble.RandomForestClassifier.score`


Naive Bayes
~~~~~~~~~~~~~~~~

:class:`sklearn.naive_bayes.MultinomialNB`

  This class provides Naive Bayes classifier for multinomial models with distributed large-scale learning.

Methods:

  * :any:`sklearn.naive_bayes.MultinomialNB.fit`
  * :any:`sklearn.naive_bayes.MultinomialNB.predict`
  * :any:`sklearn.naive_bayes.MultinomialNB.score`

Classification metrics
~~~~~~~~~~~~~~~~~~~~~~

* :func:`sklearn.metrics.accuracy_score`
* :func:`sklearn.metrics.f1_score`
* :func:`sklearn.metrics.precision_score`
* :func:`sklearn.metrics.recall_score`

