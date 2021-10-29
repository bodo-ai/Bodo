Testing
=======

Test Directives
---------------


.. important::

    Testing

.. danger::

    Testing

.. note::

    Testing

.. warning::

    Testing

.. versionadded:: 1.0.0

    Testing

.. versionchanged:: 1.0.1

    Testing

.. deprecated:: 1.0.1

    Testing

.. seealso::

    Testing

.. rubric:: Testing

.. centered:: Testing

.. hlist::
    :columns: 3

    * Testing1
    * Testing2
    * Testing3

Test Tables
-----------

+------------+------------+------------+
| Test1      | Test2      | Test3      |
+============+============+============+
| Testing1   | Testing2   | Testing3   |
+------------+------------+------------+
| Testing1   | Testing2   | Testing3   |
+------------+------------+------------+
| Testing1   | Testing2   | Testing3   |
+------------+------------+------------+
| Testing1   | Testing2   | Testing3   |
+------------+------------+------------+
| Testing1   | Testing2   | Testing3   |
+------------+------------+------------+
| Testing1   | Testing2   | Testing3   |
+------------+------------+------------+

Test Code Blocks
----------------

.. code-block:: python

    from content_filter import Filter

    filter = Filter()

    filter.check('message here')
    # returns a Check object

    filter.check('message here').as_bool
    # returns a bool of True if it found anything, False if not

    filter.add_words(['word1', 'word2', 'word2'])
    # adds words to default filter

    filter.add_exceptions(['word1', 'word2', 'word2'])
    # ignores words in default filter


.. code-block:: json

    {
        "mainFilter": [
            { "find": "find", "word": "word", "censored": "censored" },
            { "find": "helo", "word": "hello", "censored": "h3110" }
        ],
        "dontFilter": ["word"],
        "conditionFilter": [
            {
            "find": "find",
            "word": "word",
            "censored": "censored",
            "require_space": true
            }
        ]
    }
