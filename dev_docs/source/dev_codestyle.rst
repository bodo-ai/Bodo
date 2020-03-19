.. _dev_codestyle:

Code Style
----------

Bodo uses the PEP8 standard for Python code style.
We use `black <https://github.com/psf/black>`_ as formatter
and check format with `flake8 <http://flake8.pycqa.org/en/latest/>`_.

Currently our :code:`.flake8` config ignores a number of files, so whenever you are done working on a python file, run  `black <https://github.com/psf/black>`_, remove the file from :code:`.flake8`, and ensure `flake8 <http://flake8.pycqa.org/en/latest/>`_ does not raise any error.

We use the Google C++ code style guide
and enforce with `cpplint <https://github.com/cpplint/cpplint>`_.
We use `clang-format` as the formatter.
See `instructions in Pandas <https://pandas.pydata.org/pandas-docs/stable/development/contributing.html#c-cpplint>`_.

Removing Unused Imports
~~~~~~~~~~~~~~~~~~~~~~~~
When removing unused imports across all the files in the repository, `autoflake` can be used.

First install `autoflake`::

    pip install --upgrade autoflake

Following command remove unused import in a file. ::

    autoflake --in-place --remove-all-unused-imports <filename>

`-r` flag can be added to the above command to apply `autoflake` to all the files in a directory. 
More information can be found `here <https://github.com/myint/autoflake>`_.