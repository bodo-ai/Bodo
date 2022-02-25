# sklearn.feature_extraction



## sklearn.feature_extraction.text.CountVectorizer

++sklearn.feature_extraction.text.%%CountVectorizer%%++


This class provides CountVectorizer support to convert a collection of
text documents to a matrix of token counts.

!!! note
    Arguments `max_df` and `min_df` are not supported yet.

### Methods

#### sklearn.feature_extraction.text.CountVectorizer.fit_transform

- ++sklearn.feature_extraction.text.CountVectorizer.%%fit_transform%% ( raw_documents, y=None ) ++

    ***Supported Arguments***
    <br>
    <br> 
    -   `raw_documents`: iterables ( list, tuple, or NumPy Array, or Pandas
    Series that contains string)

    !!! note
        Bodo ignores `y`, which is consistent with scikit-learn.
        
        
#### sklearn.feature_extraction.text.CountVectorizer.get_feature_names_out

- ++sklearn.feature_extraction.text.CountVectorizer. %%get_feature_names_out%%()++

### Example Usage

```py
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
>>>   print(vectorizer.get_feature_names_out())
...
>>> test_count_vectorizer(corpus)
['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']
```

## sklearn.feature_extraction.text.HashingVectorizer

++sklearn.feature_extraction.text.%%HashingVectorizer%%++


This class provides `HashingVectorizer` support to convert a collection
of text documents to a matrix of token occurrences.

### Methods

#### sklearn.feature_extraction.text.HashingVectorizer.fit_transform

- ++sklearn.feature_extraction.text.HashingVectorizer.%%fit_transform%%(X, y=None)++

    ***Supported Arguments***
    <br>
    <br> 
    - `X`: iterables ( list, tuple, or NumPy Array, or Pandas
           Series that contains string)

    !!! note
        Bodo ignores `y`, which is consistent with scikit-learn.

### Example Usage

```py
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
```