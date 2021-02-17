.. _February_2021:

Bodo 2021.2 Release (Date: 2/16/2021)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This release includes many new features, bug fixes and usability improvements.
Overall, 70 code patches were merged since the last release.

New Features and Improvements
-----------------------------

- Bodo is updated to use pandas 1.2 and Arrow 3.0 (latest)

- Many improvements to error checking and reporting

- Several documentation improvements

- Support tuple return from Bodo functions where elements of the tuple have
  a mix of distributed and replicated distributions

- Improvements in automatic loop unrolling to support column names generated in loops, e.g.
  ``pd.DataFrame(X, columns=["y"] + ["x{}".format(i) for i in range(m)])``

- Improvements in caching to cover missing cases

- Pandas coverage:

    - Support column indices in ``read_csv()`` ``dtype`` argument.
      For example: ``df = pd.read_csv(fname, dtype={3: str})``
    - Support for ``df.to_string()``
    - Initial support for ``pd.Categorical()``
    - Support ``Series.min`` and ``Series.max`` for categorical data
    - Support ``pd.to_datetime()`` with categorical string input
    - Support ``pd.Series()`` constructor without ``data`` argument specified
    - Support ``dtype="str"`` in Series constructor
    - Support for ``Series.to_dict()``
    - Support for ``Series.between()``
    - Support ``Series.loc[]`` setitem with boolean array index, such as ``S.loc[idx] = val``
      where ``idx`` is a boolean array or Series
    - Support dictionary input in ``Series.map()``, such as ``S.map({1.0: "A", 4.0: "DD"})``
    - Support for ``pd.TimedeltaIndex`` min and max
    - Support for ``pd.tseries.offsets.Week``


- Numpy coverage:

    - Support ``axis=1`` in distributed ``np.concatenate``
    - Initial support for ``np.random.multivariate_normal``


- Scikit-learn:

    - Add ``coef_`` attribute to SGDClassifier model.
    - Add ``coef_`` attribute to LinearRegression model.
    - Support for ``sklearn.preprocessing.LabelEncoder`` inside jit functions.
    - Support for ``sklearn.metrics.r2_score`` inside jit functions.
