.. aws_codebuild:

AWS Codebuild Overview
======================
When attempting to merge a pull request or evaluate the correctness of the master branch,
we use AWS Codebuild to manage our CI. The set of projects used to evaluate Bodo
are found on the AWS account starting with 427 and once logged in can be found in
the Ohio region at this `link <https://us-east-2.console.aws.amazon.com/codesuite/codebuild/projects?projects-meta=%7B%22f%22%3A%7B%22text%22%3A%22%22%2C%22shared%22%3Afalse%7D%2C%22s%22%3A%7B%22property%22%3A%22LAST_MODIFIED_TIME%22%2C%22direction%22%3A-1%7D%2C%22n%22%3A20%2C%22i%22%3A0%7D&region=us-east-2#>`_.
General CI occurs inside the project labeled ``Bodo-PR-Testing``, while other evaluation, such
as performance benchmarking, are found under different project names and are not run on PRs.

Debugging Your Pull Request
---------------------------
Most likely your only interaction with Codebuild will be through attempts to debug your pull request.
This section will attempt to explain each step of the process, starting from the github interface.

1. When you submit a pull request it will automatically run the tests across a series of different
   instances. From the github interface you will see two different check associated with AWS Codebuild. 
   You want to select the check with ``BuildBatch`` in the name, as this leads to all of the tests that run
   (the other is used to spawn the batch).

   .. figure:: ../figs/Codebuild_Github.png
    :alt: Codebuild Github Appearance

2. Select the details tab to head to AWS Codebuild. You may be prompted to login. Be sure and use the 
   account starting with ``427`` and to select ``us-east-2 (Ohio)``.

3. You should now see a list of builds that have executed. Select any of the tests that have 
   failed by clicking on the link in build run. 

4. From inside a particular build you can debug your tests in a couple ways:

    a. If you are making modifications to dependencies, you should consider checking ``phase details``. 
       Here you will find a list of all the stages, including which failed. Failing in ``post-build`` suggests
       an issue with the unit tests, whereas an issue with installations or compilation will appear in either 
       ``install`` or ``build``. 

    b. If your issue is with failed tests, you should check the information in the logs. You can see the most recent commands 
       or any in progress commands in the build logs. Alteratively you can view the full set of logs on either CloudWatch or 
       download the entire set of logs from an S3 bucket. 
       
       To navigate to CloudWatch select the ``View entire log`` option on the top of the build logs. 
       Here you can look scan the logs using a query language. Unfortunately CloudWatch limits operations to 10,000 lines 
       at a time, so you will not be able to download the entire log file directly. 
       
       To properly download the logs from the S3 bucket, first look at the Build ARN. You should see a name that looks 
       something like 
       ``arn:aws:codebuild:us-east-2:427443013497:build/Bodo-PR-Testing:65ea719c-6893-4318-af6c-cc5431b0dd1f``.

       .. figure:: ../figs/Codebuild_ARN.png
        :alt: Display of where to find the Build Name

       Copy the value after the last ``:`` and place it somewhere you can copy it. In this example, the value is 
       ``65ea719c-6893-4318-af6c-cc5431b0dd1f``. Next, go to ``Build details`` and scroll down to the Logs and click 
       on the link under ``S3 location``. 
       
       .. figure:: ../figs/Codebuild_Logs_Link.png
        :alt: Display of where to find the Logs Link
       
       This takes you to the bucket that contains all the logs. Type in the copied 
       value into the prefix search to find the log associated with your build, and then you can download the log file. 
       You will need to repeat this for all failed builds whose logs you need to check.

       .. figure:: ../figs/Codebuild_Logs_Search.png
        :alt: Display of how to properly search the logs


Codebuild Scripts
-----------------

This section is meant to explain the process by Codebuild operates and the role of each script inside the
``Bodo`` repository. It is not necessary to understand how to debug your tests.

Buildspec Overview
~~~~~~~~~~~~~~~~~~

The set of builds which execute inside of AWS Codebuild are orchestrated by the contents of ``buildspec.yml``.
More specifically this file contains a section called `batch` and inside `batch-graph` which describes the
various builds that should run and the relationship between them. Currently this file is setup to distribute
tests across a set of builds with 1 and 2 processes and then once all those builds finish to run Sonar
on the result of those builds. In addition to the relationship between builds, each build in the batch also explains
the buildspec file that will be executed (all of these are found in ``buildscripts/aws/buildspecs``),
the docker image providing the environment (for example Ubuntu), the compute type upon which to execute the build,
and the environment variables at runtime. To assist with rescaling the number of builds as tests increase, 
``buildspec.yml`` is a generated by ``buildscripts/aws/update_buildspec_batch.py``.

 
Partitioning Tests
~~~~~~~~~~~~~~~~~~
In addition to simply specifying separate builds, we reduce the time of our CI process by distributing our tests
across each of the various builds according to the time needed to run. This is executed with the following steps:

1. A log of tests times are created by running all of the tests within CI using the ``-vv --durations=0`` flag on pytest.

2. This log file is uploaded to an S3 bucket where the log will subsequently be downloaded. The most recent log can
   be found `here <https://s3.console.aws.amazon.com/s3/buckets/bodo-pr-testing-logs/splitting_logs/?region=us-east-2&tab=overview>`_.

3. Within each build on AWS Codebuild, the latest log is downloaded from within ``bodo/runtests.py`` using
   ``buildscripts/aws/download_s3paths_with_prefix.py``.

4. From ``buildscripts/aws/select_timing_from_logs.py``, the log file is scanned and the timing of each test is extracted.
   Then a greedy algorithm runs to distribute these tests within each of the specified builds. This output is then
   written to a json file where it will be read by ``pytest`` as part of the test collection process.

5. A step in ``bodo/tests/conftest.py`` assigns markers to each test based on the results of the distribution process.
   Any tests that could not be found are assigned to group 0. This will be eventually replaced with a more even distribution,
   but note this CANNOT rely on any randomness or else it will hang when ``NP != 1``.

6. The tests within each build are then are selected an executed based upon their group number.

Additionally, support for partitioned tests also requires support for a distributed coverage collection process.
This is currently accomplished with the following steps, which assume familiarity with the basic Sonar and coverage
collection process.

1. Coverage is gathered and then merged for all ``NP=1`` labeled builds. This places the result in a single
   ``.coverage`` file.

2. ``.coverage`` files are uploaded to s3 with a unique batch specific name. This process is accomplished using the script,
   ``buildscripts/aws/get_sonar_artifact_name.sh``, which calls ``buildscripts/aws/get_batch_prefix.sh`` to get the batch
   specific prefix for the s3 artifact. To allow reusing the same buildspec file, ``NP=2`` builds upload an empty ``.coverage`` file
   using the same prefix. This does not impact the final results.

3. Within the Sonar specific build, all artifacts sharing a common prefix are downloaded using the scripts
   ``buildscripts/aws/get_batch_prefix.sh`` and ``buildscripts/aws/download_s3paths_with_prefix.py``. This step should always
   succeed because Sonar is dependent on all ``NP=1`` builds finishing and uploading their artifacts.

4. ``buildscripts/aws/update_coverage_config.py`` modifies the configuration file used for coverage. This informs
   the coverage step that certain paths should be treated as identical for calculating coverage. This step must be
   performed dynamically because an absolute path on the instance running the coverage merge must be given and this cannot
   be determined until runtime.

5. Coverages are merged and a report is generated, which is uploaded to the server. 


Generating a New Log File
~~~~~~~~~~~~~~~~~~~~~~~~~
In our current configuration, tests are split between ``CodeBuild`` runs based upon a log file that has to be produced manually.
As all new tests get grouped into group 0, it is necessary to occasionally create a updated logfile to redistribute
the tests. To create a new logfile you should:

1. Create a new branch from master upon which to generate your log. In this branch, modify the ``pytest``
   arguments in ``buildscripts/aws/run_unittests.sh`` to contain ``-vv --durations=0`` instead of ``-v``.

2. Push your branch to github, but do not create a pull request. In the next step you will manually run
   Codebuild using your branch.

3. Navigate to the ``Bodo-PR-Testing`` build project on codebuild and select ``Start Build``. Next you
   want to select ``Advanced build overrides`` in the top right. At this point you need to specify your individual
   build.

         a. In ``Build Configuration`` select ``Single Build`` for the build type.
         b. In ``Source`` give the name of the branch that you created inside the box for ``Source version``.
         c. Within ``Environment`` navigate to ``Additional configuration.`` 
               i. Modify the timeout value so it is large enough to run all tests 
                  (4 hours should work).
               ii. | Add three environment variables:
                   | Name: ``NP``, Value: ``1``
                   | Name: ``NUMBER_GROUPS_SPLIT``, Value: ``1``
                   | Name: ``PYTEST_MARKER``, Value: ``not slow``
         d. Navigate to ``Buildspec`` and enter ``buildscripts/aws/buildspecs/CI_buildspec.yml`` 
            in the box for ``Buildspec name``.
         e. Select ``Start Build`` at the bottom. Save this URL so you can find the log easily.

4. Once the Build finishes, download the logfile from S3 as described in ``Debugging Your Pull Request``.

5. Upload your logfile `here <https://s3.console.aws.amazon.com/s3/buckets/bodo-pr-testing-logs/splitting_logs/?region=us-east-2&tab=overview>`_
   with a new name (e.g. log-{date}).

6. Submit a new PR which updates the value of ``logfile_name`` within ``bodo/runtests.py`` to use the
   path of your logfile. This ensures that all existing PRs will continue to divide tests using the
   old log file until your code is merged.


.. _Updating the Batch:

Updating the Batch
~~~~~~~~~~~~~~~~~~
If you want to change the overall batch configuration, for example to change the total number of parallel
builds, you need to modify the contents of ``buildspec.yml``. However this file is current autogenerated by
``buildscripts/aws/update_coverage_config.py``. If you need to modify the details within the builds, for example
the environment variables in each build, you will need to make modifications to this script.
If you only want to change the number of parallel builds (increase or decrease), but not the overall structure,
then you don't need to make any changes to this file.

Once you make your changes, run ``python buildscripts/aws/update_coverage_config.py CI <num_parallel_builds>``,
where ``<num_parallel_builds>`` is the number of builds you want to partition tests across in parallel. This updates the 
``buildspec.yml``file locally, which you then need to merge to master through a PR. 

**Important Note:** In codebuild, your PR is not automatically merged to master, until after the ``buildspec.yml``
file is used. As a result, if you update ``buildspec.yml`` and these changes are needed in others PRs, 
you will need people to manually rebase with master.


Updating Dependencies
~~~~~~~~~~~~~~~~~~~~~

CI runs on custom docker images to avoid undergoing installation time on every build. As a result,
if you every need to upgrade a dependencies to a newer version, you also need to update the docker
image on ECR. To do so:

1. Update the installation in the necessary buildscripts. This should either modify ``buildscripts/setup_conda.sh``
   or ``buildscripts/aws/test_installs.sh``.

2. Rebuild the docker image. This can be done by executing the command 
   ``docker build -f buildscripts/aws/docker/CI-dockerfile -t bodo-codebuild:latest``

3. Upload the container to ECR `here <https://us-east-2.console.aws.amazon.com/ecr/repositories/bodo-codebuild/?region=us-east-2>`_.
   Click ``View Push Commands`` for steps to upload. However, rather than uploading as latest, you should
   use a tag that is different than the existing image (e.g. if 1.0 is there now do 1.1). The exact tag is
   not important and we don't intend to keep images around long-term, but this should be different so the
   process of developing your PR doesn't break any existing PRs before merging.

4. Update the image name in ``buildscripts/aws/update_coverage_config.py`` in the ``generate_CI_buildspec``
   function to use your current tag.

5. Generate a new buildspec file by following the steps in :ref:`Updating the Batch`.

6. Once your PR is merged you may need people to rebase their in progress PRs with master as updates to the 
   outermost ``buildspec.yml`` will not merge with master.

7. Delete the previous ECR image as all builds now need the newest build. This will allow for quicker failure
   if someone hasn't rebased their PR and avoids any lingering resources.
