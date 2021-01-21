.. _dev_with_platform:

Develop using Platform
----------------------


Building Dev AMIs
~~~~~~~~~~~~~~~~~
.. _dev-amis:

To test your branch or latest master before release on the platform, you can create a Dev AMI and test it on the `Dev Platform <https://dev.bodo.ai>`_ .

Run a new `Bodo-build-binary-platform-obfuscated <https://dev.azure.com/bodo-inc/Bodo/_build?definitionId=8>`_ pipeline on `Azure Pipelines <https://dev.azure.com/bodo-inc/Bodo/_build>`_ by clicking on **Run pipeline** button
with the variable ``PLATFORM_DEV_RELEASE`` set to ``true`` on your branch as shown below. 

    .. figure:: ../figs/azure-platform-pipeline.png
        :alt: Azure Platform Pipeline

    .. figure:: ../figs/azure-platform-set-vars.png
        :alt: Azure Platform Pipeline set Variable

This will create a new platform-compatible Bodo binary and upload it to the `bodo.ai-platform artifactory channel <https://bodo.jfrog.io/ui/repos/tree/General/bodo.ai-platform>`_. 
It will then trigger the `bodo-ami repo CI <https://github.com/Bodo-inc/bodo-ami/actions?query=workflow%3Abuild_publish_amis>`_ to create a new AMI and make it available on the `Dev Platform <https://dev.bodo.ai>`_.
The Bodo Version to use will be the same as the Bodo Version as printed by the Azure Pipeline during the *Get Bodo Version* step.

    .. figure:: ../figs/azure-platform-get-bodo-version.png
        :alt: Azure Platform Pipeline Get Bodo Version

Once created, log onto the Dev Platform, and create a cluster with this Bodo Version.
Once you're done testing, ask one of the Platform Team members (Sahil, Nick, etc.) to remove the AMI from the Dev Platform.
There will be a pipeline to automate this in the future.
