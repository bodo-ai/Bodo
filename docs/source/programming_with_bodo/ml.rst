.. _ml:

Machine Learning
----------------

Scikit-learn
~~~~~~~~~~~~

Below is the list of scikit-learn classes and functions that Bodo supports natively inside JIT functions.
This list will expand regularly as we add support for more APIs.
Optional arguments are not supported unless specified.

Linear Classifiers
******************

:class:`sklearn.linear_model.LogisticRegression` :sup:`*`

  This class provides logistic regression classifier.

Methods:

  * :any:`sklearn.linear_model.LogisticRegression.fit`
  * :any:`sklearn.linear_model.LogisticRegression.predict`
  * :any:`sklearn.linear_model.LogisticRegression.predict_proba`
  * :any:`sklearn.linear_model.LogisticRegression.predict_log_proba`
  * :any:`sklearn.linear_model.LogisticRegression.score`

Attributes:

* :attr:`sklearn.linear_model.LogisticRegression.coef_`

:class:`sklearn.linear_model.SGDClassifier`

  This class provides linear classification models with SGD optimization which allows distributed large-scale learning.

  ``SGDClassifier(loss='hinge')`` is equivalent to SVM linear classifer. 

  ``SGDClassifier(loss='log')`` is equivalent to logistic regression classifer. 

    * Supported loss functions ``hinge`` and ``log``.
    * ``early_stopping`` is not supported yet.

Methods:

  * :any:`sklearn.linear_model.SGDClassifier.fit`
  * :any:`sklearn.linear_model.SGDClassifier.predict`
  * :any:`sklearn.linear_model.SGDClassifier.predict_proba`
  * :any:`sklearn.linear_model.SGDClassifier.predict_log_proba`
  * :any:`sklearn.linear_model.SGDClassifier.score`

:class:`sklearn.svm.LinearSVC`

  This class provides Linear Support Vector Classification.

Methods:

  * :any:`sklearn.svm.LinearSVC.fit`
  * :any:`sklearn.svm.LinearSVC.predict`
  * :any:`sklearn.svm.LinearSVC.score`

Attributes:

* :attr:`sklearn.linear_model.SGDClassifier.coef_`

Linear Regressors 
*****************

:class:`sklearn.linear_model.LinearRegression` :sup:`*`

  This class provides linear regression support.
  Note: Multilabel targets are not currently supported.

Methods:

  * :any:`sklearn.linear_model.LinearRegression.fit`
  * :any:`sklearn.linear_model.LinearRegression.predict`
  * :any:`sklearn.linear_model.LinearRegression.score`

Attributes:

* :attr:`sklearn.linear_model.LinearRegression.coef_`

:class:`sklearn.linear_model.Ridge` :sup:`*`

  This class provides ridge regression support.

Methods:

  * :any:`sklearn.linear_model.Ridge.fit`
  * :any:`sklearn.linear_model.Ridge.predict`
  * :any:`sklearn.linear_model.Ridge.score`

Attributes:

* :attr:`sklearn.linear_model.Ridge.coef_`

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


:class:`sklearn.linear_model.Lasso` :sup:`*`

  This class provides Lasso regression support.

Methods:

  * :any:`sklearn.linear_model.Lasso.fit`
  * :any:`sklearn.linear_model.Lasso.predict`
  * :any:`sklearn.linear_model.Lasso.score`


**\***
To enable distributed training across multiple nodes, Bodo uses Stochastic Gradient Descent (SGD) to train these model types. This provides a model that has similar performance as the corresponding sequential version.

  To achieve that, it is highly recommended to scale your data using `StandardScaler` before training and/or testing the model.

  See scikit-learn for more tips on how to tune model parameters for SGD `here <https://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use>`_.


Clustering
**********

:class:`sklearn.cluster.KMeans`

  This class provides K-Means clustering models which allows distributed large-scale unsupervised learning.

Methods:

  * :any:`sklearn.cluster.KMeans.fit`
  * :any:`sklearn.cluster.KMeans.predict`
  * :any:`sklearn.cluster.KMeans.score`
  * :any:`sklearn.cluster.KMeans.transform`

Ensemble Methods
****************

:class:`sklearn.ensemble.RandomForestClassifier`

  This class provides Random Forest Classifier, an ensemble learning model, for distributed large-scale learning.

  * ``random_state`` value is ignored when running on multi-node.

Methods:

  * :any:`sklearn.ensemble.RandomForestClassifier.fit`
  * :any:`sklearn.ensemble.RandomForestClassifier.predict`
  * :any:`sklearn.ensemble.RandomForestClassifier.predict_proba`
  * :any:`sklearn.ensemble.RandomForestClassifier.predict_log_proba`
  * :any:`sklearn.ensemble.RandomForestClassifier.score`

:class:`sklearn.ensemble.RandomForestRegressor`

  This class provides Random Forest Regressor, an ensemble learning model, for distributed large-scale learning.

  * ``random_state`` value is ignored when running on multi-node.

Methods:

  * :any:`sklearn.ensemble.RandomForestRegressor.fit`
  * :any:`sklearn.ensemble.RandomForestRegressor.predict`
  * :any:`sklearn.ensemble.RandomForestRegressor.score`

Naive Bayes
***********

:class:`sklearn.naive_bayes.MultinomialNB`

  This class provides Naive Bayes classifier for multinomial models with distributed large-scale learning.

Methods:

  * :any:`sklearn.naive_bayes.MultinomialNB.fit`
  * :any:`sklearn.naive_bayes.MultinomialNB.predict`
  * :any:`sklearn.naive_bayes.MultinomialNB.score`

Classification metrics
**********************

* :func:`sklearn.metrics.accuracy_score`
* :func:`sklearn.metrics.confusion_matrix`
* :func:`sklearn.metrics.f1_score`
* :func:`sklearn.metrics.precision_score`
* :func:`sklearn.metrics.recall_score`


Regression metrics
******************

* :func:`sklearn.metrics.mean_absolute_error`
* :func:`sklearn.metrics.mean_squared_error`
* :func:`sklearn.metrics.r2_score`


Data Preprocessing
******************

:class:`sklearn.preprocessing.StandardScaler`

  This class provides Standard Scaler support to center your data and to scale it to achieve unit variance.

Methods:

  * :any:`sklearn.preprocessing.StandardScaler.fit`
  * :any:`sklearn.preprocessing.StandardScaler.transform`
  * :any:`sklearn.preprocessing.StandardScaler.inverse_transform`

:class:`sklearn.preprocessing.MinMaxScaler`

  This class provides MinMax Scaler support to scale your data based on the range of its features.

Methods:

  * :any:`sklearn.preprocessing.MinMaxScaler.fit`
  * :any:`sklearn.preprocessing.MinMaxScaler.transform`
  * :any:`sklearn.preprocessing.MinMaxScaler.inverse_transform`

:class:`sklearn.preprocessing.LabelEncoder`

  This class provides LabelEncoder support to encode target labels (y) with values between 0 and n-classes-1.

Methods:

  * :any:`sklearn.preprocessing.LabelEncoder.fit`
  * :any:`sklearn.preprocessing.LabelEncoder.transform`
  * :any:`sklearn.preprocessing.LabelEncoder.fit_transform`

Feature Extraction
******************

:class:`sklearn.feature_extraction.text.HashingVectorizer`

  This class provides HashingVectorizer support to convert a collection of text documents to a matrix of token occurrences.

Methods:
  * :any:`sklearn.feature_extraction.text.HashingVectorizer.fit_transform`

:class:`sklearn.feature_extraction.text.CountVectorizer`

  This class provides CountVectorizer support to convert a collection of text documents to a matrix of token counts.

Methods:
  * :any:`sklearn.feature_extraction.text.CountVectorizer.fit_transform`
  * :any:`sklearn.feature_extraction.text.CountVectorizer.get_feature_names`

Model Selection
***************

* :func:`sklearn.model_selection.train_test_split`

  * Currently it only supports two inputs of type numpy arrays and/or pandas dataframes.
  * Arguments ``train_size`` and ``test_size`` accept float between 0.0 and 1.0 or ``None`` only.
  * Arguments ``random_state`` and ``shuffle`` are supported.
  * Argument ``stratify`` is not supported yet.



XGBoost
~~~~~~~

Below is the list of XGBoost (using the Scikit-Learn-like API) classes and functions that Bodo supports natively inside JIT functions.
This list will expand regularly as we add support for more APIs.

XGBClassifier
*****************

:class:`xgboost.XGBClassifier`

  This class provides implementation of the scikit-learn API for XGBoost classification with distributed large-scale learning.

Methods:

  * :any:`xgboost.XGBClassifier.fit`
  * :any:`xgboost.XGBClassifier.predict`
  * :any:`xgboost.XGBClassifier.predict_proba`

Attributes:

* :attr:`xgboost.XGBClassifier.feature_importances_`

XGBRegressor
*****************

:class:`xgboost.XGBRegressor`

  This class provides implementation of the scikit-learn API for XGBoost regression with distributed large-scale learning.

Methods:

  * :any:`xgboost.XGBRegressor.fit`
  * :any:`xgboost.XGBRegressor.predict`

Attributes:

* :attr:`xgboost.XGBRegressor.feature_importances_`
