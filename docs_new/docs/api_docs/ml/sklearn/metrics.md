# sklearn.metrics 

#### sklearn.metrics.accuracy_score


- <code><apihead>sklearn.metrics.<apiname>accuracy_score</apiname>(y_true, y_pred, normalize=True, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    -   `y_true`: 1d array-like.
    -   `y_pred`: 1d array-like.
    -   `normalize`: bool.
    -   `sample_weight`: 1d numeric array-like or None.
    
    !!! note
        `y_true`, `y_pred`, and `sample_weight` (if provided) must be of
        same length.

    ***Example Usage***
    
    ```py 
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
    ```

#### sklearn.metrics.confusion_matrix


- <code><apihead>sklearn.metrics.<apiname>confusion_matrix</apiname>(y_true, y_pred, labels=None, sample_weight=None, normalize=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -   `y_true`: 1d array-like.
    -   `y_pred`: 1d array-like.
    -   `labels`: 1d array-like.
    -   `sample_weight`: 1d numeric array-like or `None`.
    -   `normalize`: Must be one of `'true'`, `'pred'`, `'all'`, or `None`
    
    !!! note
        `y_true`, `y_pred`, and `sample_weight` (if provided) must be of
        same length.
    
    ***Example Usage***
    
    ```py 
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
    ```      

#### sklearn.metrics.f1_score


- <code><apihead>sklearn.metrics.<apiname>f1_score</apiname>(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -   `y_true`: 1d array-like.
    -   `y_pred`: 1d array-like.
    -   `average`: Must be one of `'micro'`, `'macro'`, `'samples'`,
        `'weighted'`, `'binary'`, or None.
    
    !!! note
        `y_true` and `y_pred` must be of same length.
    
    ***Example Usage***
    
    ```py 
    >>> import bodo
    >>> from sklearn.metrics import f1_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> @bodo.jit
    >>> def test_f1_score(y_true, y_pred):
    ...   print(f1_score(y_true, y_pred, average='macro'))
    >>> test_f1_score(y_true, y_pred)
    0.26666666666666666
    ```  

#### sklearn.metrics.mean_absolute_error


- <code><apihead>sklearn.metrics.<apiname>mean_absolute_error</apiname>(y_true, y_pred, sample_weight=None, multioutput='uniform_average')</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -   `y_true`: NumPy array.
    -   `y_pred`: NumPy array.
    -   `sample_weight`: Numeric NumPy array or None.
    -   `multioutput`: Must be one of `'raw_values'`,
        `'uniform_average'`, or array-like.
    
    !!! note
        `y_true`, `y_pred`, and `sample_weight` (if provided) must be of
        same length.
    
    ***Example Usage***
    <br>
    <br>    
    ```py 
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
    ```

#### sklearn.metrics.mean_squared_error


- <code><apihead>sklearn.metrics.<apiname>mean_squared_error</apiname>(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -   `y_true`: NumPy array.
    -   `y_pred`: NumPy array.
    -   `sample_weight`: Numeric NumPy array or None.
    -   `multioutput`: Must be one of `'raw_values'`,
        `'uniform_average'`, or array-like.
    
    !!! note
        `y_true`, `y_pred`, and `sample_weight` (if provided) must be of
        same length.
    
    ***Example Usage***
    
    ```py 
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
    ```  

#### sklearn.metrics.precision_score


- <code><apihead>sklearn.metrics.<apiname>precision_score</apiname>(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -   `y_true`: 1d array-like.
    -   `y_pred`: 1d array-like.
    -   `average`: Must be one of `'micro'`, `'macro'`, `'samples'`,
        `'weighted'`, `'binary'`, or `None`.
    
    !!! note
        `y_true` and `y_pred` must be of same length.
    
    ***Example Usage***
    
    ```py 
    >>> import bodo
    >>> from sklearn.metrics import precision_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> @bodo.jit
    >>> def test_precision_score(y_true, y_pred):
    ...   print(precision_score(y_true, y_pred, average='macro'))
    >>> test_precision_score(y_true, y_pred)
    0.2222222222222222
    ```  

#### sklearn.metrics.r2_score


- <code><apihead>sklearn.metrics.<apiname>r2_score</apiname>(y_true, y_pred, sample_weight=None, multioutput='uniform_average')</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -   `y_true`: NumPy array.
    -   `y_pred`: NumPy array.
    -   `sample_weight`: Numeric NumPy array or `None`.
    -   `multioutput`: Must be one of `'raw_values'`,
        `'uniform_average'`, `'variance_weighted'`, `None`, or
         array-like.
    
    !!! note 
        `y_true`, `y_pred`, and `sample_weight` (if provided) must be of
        same length.
    
    ***Example Usage***
    
    ```py 
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
    ```

#### sklearn.metrics.recall_score


- <code><apihead>sklearn.metrics.<apiname>recall_score</apiname>(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>    
    -   `y_true`: 1d array-like.
    -   `y_pred`: 1d array-like.
    -   `average`: Must be one of `'micro'`, `'macro'`, `'samples'`,
        `'weighted'`, `'binary'`, or `None`.
    
    !!! note
        `y_true` and `y_pred` must be of same length.
    
    ***Example Usage***
    
    ```py 
    >>> import bodo
    >>> from sklearn.metrics import recall_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> @bodo.jit
    >>> def test_recall_score(y_true, y_pred):
    ...   print(recall_score(y_true, y_pred, average='macro'))
    >>> test_recall_score(y_true, y_pred)
    0.3333333333333333
    ```
