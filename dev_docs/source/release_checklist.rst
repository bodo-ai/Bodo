.. _release_checklist:

Release Checklist
----------------------

1. Create a new ``Month_Year.rst`` (ex. Feb_2020.rst) file for the new release under `docs/source/release_notes directory <https://github.com/Bodo-inc/Bodo/tree/master/docs/source/release_notes>`_, fill out the contents, add it to the `release notes page <https://github.com/Bodo-inc/Bodo/blob/master/docs/source/releases.rst>`_. To prepare the release notes, it is useful to view the PRs closed since
the last release. For example, use the following filter on GitHub: ``is:pr is:closed merged:>=yyyy-mm-dd base:master``.

2. Change ``:caption:`` field in ``docs/index.rst`` to ``Version year.month`` (ex. 2020.04).

3. Go to `redirect <https://github.com/Bodo-inc/Bodo/tree/master/docs/_static/redirect>`_ , create a new ``Year_Month.html`` (ex. 2020_04.rst) file. Fill in redirect url like ``<meta http-equiv="refresh" content="0; url=/2020.04/index.html" />``.

4. Update `previous documentation list <https://github.com/Bodo-inc/Bodo/blob/master/docs/source/prev_doc_link.rst>`_

5. open a PR, and request every core developer to review it.

6. Run HDFS tests manually, use the `Bodo development for hdfs <https://github.com/Bodo-inc/Bodo/blob/master/dev_docs/source/docker_dev.rst#docker-images>`_ docker image. Follow setup steps #1, #2, and #5.

7. Once the PR opened in #4 is merged, tag release::

	   # git tag year.month
	   git tag -a 2020.04 -m "Bodo release version xxxx.xx"
	   git push --tags

   Once the tag is pushed, the `release pipeline <https://dev.azure.com/bodo-inc/Bodo/_build?definitionId=2&_a=summary>`_ is triggered. 

8. Update docs caption, and push the updated `documentation <https://docs.bodo.ai>`_ ::

	   cd docs
	   make html
	   python gh-pages.py [bodo version(ex:2020.04)]
	   cd gh-pages
	   git push

9. Change the `symlink <https://github.com/Bodo-inc/Bodo-doc/blob/gh-pages/latest>`_ ::

	   # inside gh-pages directory
	   rm latest
	   ln -s [bodo version(ex:2020.04)] latest
	   git add .
	   git commit -m 'update symlink to 2020.04'
	   git push


10. Clear your browser's cache and ensure `docs.bodo.ai <https://docs.bodo.ai>`_ points to the right path.

Ta-da! The binary artifact will be available in the `release pipeline <https://dev.azure.com/bodo-inc/Bodo/_build?definitionId=2&_a=summary>`_
