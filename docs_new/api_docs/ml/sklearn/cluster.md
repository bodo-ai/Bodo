# sklearn.cluster: Clustering¶

## sklearn.cluster.KMeans

++sklearn.cluster.%%KMeans%%++
This class provides K-Means clustering model.

!!! important

    Currently, this model works by gathering all the data in a single node and 
    then generating K-Means model. Make sure you have enough memory on the first 
    node in your hostfile.

### Methods

#### sklearn.cluster.KMeans.fit 
- ++sklearn.cluster.KMeans.%%fit%%(X, y=None, sample_weight=None)++

    ***Supported Arguments***
    <br>
    <br>
    * `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    * `sample_weight`: Numeric NumPy Array

    !!! note
        Bodo ignores `y`, which is consistent with scikit-learn.

#### sklearn.cluster.KMeans.predict

- ++ sklearn.cluster.KMeans.%%predict%%(X, sample_weight=None)++

    ***Supported Arguments***
    <br>
    <br>
    - `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    - `sample_weight`: Numeric NumPy Array

#### sklearn.cluster.KMeans.score
* ++ sklearn.cluster.KMeans.%%score%%(X, y=None, sample_weight=None)++

    ***Supported Arguments***
    <br>
    <br>
    - `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    - `sample_weight`: Numeric NumPy Array

    !!! note
        Bodo ignores y, which is consistent with scikit-learn.

#### sklearn.cluster.KMeans.transform
* ++ sklearn.cluster.KMeans.%%transform%%(X)++

     ***Supported Arguments***
    <br>
    <br> 
     - `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.

### Example Usage

```py
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
```