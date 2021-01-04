
.. _dev_getting_started:

Getting Started
---------------

*Important*: While most of this page will showup in Github, the internal links
in this document only work properly after rendering the RST file into a webpage.
You can do this by running ``make html`` inside `dev_docs` and then view the output
in `dev_docs/_build/html/source/getting_started.html`.

This document is split into sections to serve a reference when shifting
between Bodo projects. Please refer back to this document as a first step 
when working in a new area, even if your onboarding stage has passed.

Registration
~~~~~~~~~~~~
#. Please confirm you have access to all of the following on your first day.


   - Your Bodo email account.
   - The shared `Bodo Google Drive <https://drive.google.com/drive/folders/16p921BC7ZIWTyssE-QDqL3F1UTg4uuXQ?usp=sharing>`_.
   - An IAM role on the 427 AWS account. You will use this account to view the output of CI on AWS Codebuild.
   - Access to Azure Pipelines.
   - Access to 1Password.
   - Acesss to JIRA.

#. Register on the Platform. There are three namespaces on which you should register.
   ``Production`` contains the released version of the platform and is used by Bodo customers.
   ``Staging`` also contains a released version of the platform, but is not accessible to customers.
   If you need to do any testing or benchmarking and you don't need a development branch of Bodo,
   please use this namespace. Finally, ``development`` contains the most recently added features
   in the platform. Please only use this namespace if you are testing the platform or you need access
   to development features. To signup for each of these namespaces, follow the steps to **Create a new organization**,
   as described on `this wiki <https://github.com/Bodo-inc/bodo-platform-auth/wiki/Registration-and-Organization-Initialization>`_.
   Then, using your signup link, follow the signup instructions on the `platform documentation <https://docs.bodo.ai/latest/source/bodo_platform.html>`_.

Bodo Background
~~~~~~~~~~~~~~~
#. This `company slide deck <https://drive.google.com/file/d/1V5Kq1n-Ud1qk87TqiPNs7ePaZpZzTBgX/view?usp=sharing>`_ provides an overview of the company and technology.
#. Learn Numba usage and compiler architecture, see :ref:`numba-info`.
#. Make sure you have signed up for the platform. Complete the `getting started tutorial <https://github.com/Bodo-inc/Bodo-tutorial/blob/master/bodo_getting_started.ipynb>`_
   and the `extended tutorial <https://github.com/Bodo-inc/Bodo-tutorial/blob/master/bodo_tutorial.ipynb>`_
   on the platform. These notebooks should be included when you create a notebook automatically.


Before your First Issue
~~~~~~~~~~~~~~~~~~~~~~~
#. Install Bodo for development, see :ref:`build_bodo_source`.
#. Read through our github practices, see :ref:`github_practices_info`.
#. Go over Development Process page :ref:`dev_process_info`.

**Important 1**: When you are first starting please utilize the draft PR feature
of Github. Getting teammate feedback is very important for your personal growth,
so please submit to Github even when you don't believe your code is completely
ready.

**Important 2**: Please remember the 20 minute rule. If you are stuck for 20 minutes,
ask for help.

Bodo Engine References
~~~~~~~~~~~~~~~~~~~~~~
#. Go over `a basic Pandas tutorial <https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html>`_.
#. Go over Bodo Engine Development page :ref:`bodo_dev_info`.

BodoSQL References
~~~~~~~~~~~~~~~~~~
#. Go over `a basic SQL tutorial <https://mode.com/sql-tutorial/introduction-to-sql>`_.
#. Go over the `BodoSQL development documentation <https://github.com/Bodo-inc/BodoSQL/tree/master/dev_docs>`_.

Benchmarking References
~~~~~~~~~~~~~~~~~~~~~~~
#. Go over `a basic Spark tutorial <https://www.tutorialspoint.com/pyspark/index.htm>`_.
#. Go over  the Benchmarking Tips documentation :ref:`benchmarking_getting_started`.

Additional Learning Material
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#. Watch the `tutorial workshop video <https://drive.google.com/file/d/1X3X5iv0P5hbAkeb5mIrwBBEd7TJc6-ak/view?usp=sharing>`_.
#. Read the `development priorities document <https://docs.google.com/document/d/15RcReBidrJbrojJvXBBpEWpuAEGtLsX46RcQr60iHCI/edit#>`_.
#. Go over `Bodo user documentation <http://docs.bodo.ai/>`_.
