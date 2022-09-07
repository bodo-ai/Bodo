# sklearn.naive_bayes

## sklearn.naive_bayes.MultinomialNB


- <code><apihead>sklearn.naive_bayes.<apiname>MultinomialNB</apiname></apihead></code>
<br><br>

This class provides Naive Bayes classifier for multinomial models with
distributed large-scale learning.

### Methods

#### sklearn.naive_bayes.MultinomialNB.fit


- <code><apihead>sklearn.naive_bayes.MultinomialNB.<apiname>fit</apiname>(X, y, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -   `X`: NumPy Array or Pandas Dataframes.
    -   `y`: NumPy Array or Pandas Dataframes.

#### sklearn.naive_bayes.MultinomialNB.predict


- <code><apihead>sklearn.naive_bayes.MultinomialNB.<apiname>predict</apiname>(X)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -  `X`: NumPy Array or Pandas Dataframes.

#### sklearn.naive_bayes.MultinomialNB.score


- <code><apihead>sklearn.naive_bayes.MultinomialNB.<apiname>score</apiname>(X, y, sample_weight=None)</apihead></code>
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
```