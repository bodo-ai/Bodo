# sklearn.cluster: Clustering

## sklearn.cluster.KMeans

<code><apihead>sklearn.cluster.<apiname>KMeans</apiname></apihead></code><br><br><br>This class provides K-Means clustering model.

!!! important

    Currently, this model works by gathering all the data in a single node and 
    then generating K-Means model. Make sure you have enough memory on the first 
    node in your hostfile.

### Methods

#### sklearn.cluster.KMeans.fit 

- <code><apihead>sklearn.cluster.KMeans.<apiname>fit</apiname>(X, y=None, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    * `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    * `sample_weight`: Numeric NumPy Array

    !!! note
        Bodo ignores `y`, which is consistent with scikit-learn.

#### sklearn.cluster.KMeans.predict


- <code><apihead> sklearn.cluster.KMeans.<apiname>predict</apiname>(X, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    - `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    - `sample_weight`: Numeric NumPy Array

#### sklearn.cluster.KMeans.score

- <code><apihead> sklearn.cluster.KMeans.<apiname>score</apiname>(X, y=None, sample_weight=None)</apihead></code>
<br><br>
    ***Supported Arguments***
    <br>
    <br>
    - `X`: NumPy Array, Pandas Dataframes, or CSR sparse matrix.
    - `sample_weight`: Numeric NumPy Array

    !!! note
        Bodo ignores y, which is consistent with scikit-learn.

#### sklearn.cluster.KMeans.transform

- <code><apihead> sklearn.cluster.KMeans.<apiname>transform</apiname>(X)</apihead></code>
<br><br>
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