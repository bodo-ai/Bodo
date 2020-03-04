.. _documentation:

User&Dev Documentation
---------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~~~

The `user (or dev) documentation <https://docs.bodo.ai>`_ is under the `docs (or dev_docs)` directory of the repository and uses
the reStructuredText format.
It is built with `Sphinx <http://www.sphinx-doc.org>`_ and the alabaster theme::

    conda install sphinx
    conda install alabaster

After updating documentation, run :code:`make html` in the `docs` folder to build.  
Open `index.html` to view the documentation.  
For the user documentation only, to update, use the :code:`gh-pages.py`
script under :code:`docs`::

    python gh-pages.py [bodo version(ex:2020.02.0)]  

Default tag `dev` will be used if no tag is provided. 
Then verify the repository under the :code:`gh-pages` directory and
:code:`git push` to `Bodo-doc <https://github.com/Bodo-inc/Bodo-doc>`_ repo :code:`gh-pages` branch.

The developer documentation is under the `dev_docs` directory
and can be built with :code:`make html` as well.


Updating User Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remember to update User Documentation as new `pandas`/`numpy` functions are added, and ensure external links to `pandas` are valid.
For each new release of Bodo version, user documentation should also be updated.  

    1. Create a new ``Month_Year.rst``(ex. Feb_2020.rst) file for the new release under `docs/source/release_notes directory <https://github.com/Bodo-inc/Bodo/tree/master/docs/source/release_notes>`_ , fill out the contents, and request every core developer to review it.
    2. Once the release is tagged, the new documentation version is built, and uploaded to Bodo-doc repository using commands above, update the hyper link on `Previous Documentation` category of the User Doc.  
    3. Next, the `latest` symbolic link from gh-pages branch of Bodo-doc repository should be updated to the new version. 

For more release related instruction, visit our `release checklist <https://github.com/Bodo-inc/Bodo/wiki/Release-Checklist>`_ Wiki page.
