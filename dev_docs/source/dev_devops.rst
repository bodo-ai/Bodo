.. _dev_devops:

DevOps
----------

We currently have two build pipelines on `Azure DevOps <https://dev.azure.com/bodo-inc/Bodo/_build>`_:

1. Bodo-inc.Bodo: This pipeline is triggered whenever a pull request whose target branch is set to :code:`master` is created and following commits. This does not test on the full test suite in order to save time. A `codecov <https://codecov.io/gh/Bodo-inc/Bodo>`_ code coverage report is generated and uploaded for testing on Linux with one processor.

2. Bodo-build-binary: This pipeline is used for release and automatic nightly testing on full test suite, triggered by pushing tags. It has two stages. The first stage removes docstrings, builds the bodo binary and makes the artifact(:code:`bodo-inc.zip`) available for downloads. The second stage runs the full test suite with the binary we just built on Linux with 1, 2, and 3 processors. It is structured this way so that in case of emergency bug fix release, we can still download the binary without waiting for the tests to finish. 

The default :code:`TRIAL_PERIOD` is 14(days) set through Azure's UI, and this enviroment variable can be changed before manually triggering the build. 

:code:`MAX_CORE_COUNT` does not have a default value, it can be set through Azure's UI when manually triggering it.
