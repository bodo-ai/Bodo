# sklearn.model_selection

#### sklearn.model_selection.train_test_split

<code><apihead>sklearn.model_selection.<apiname>train_test_split</apiname>(X, y, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)</apihead></code><br><br><br>
***Supported Arguments***

-   `X`: NumPy array or Pandas Dataframes.
-   `y`: NumPy array or Pandas Dataframes.
-   `train_size`: float between 0.0 and 1.0 or `None` only.
-   `test_size`: float between 0.0 and 1.0 or `None` only.
-   `random_state`: int, RandomState, or None.
-   `shuffle`: bool.

***Example Usage***

```py 
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
```
