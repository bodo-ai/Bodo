.. _dev_benchmark:

Performance Benchmarking
-------------------------
We use AWS EC2 instance for performance benchmark on Bodo. 
This is essentially to check the performance variations based on commits to master branch.
Similar to our nightly build, benchmarking is set to run regularly. 
To set up this infrastructure there are 3 things that should be constructed. 

    1. AWS EC2 instance
    2. AWS CodePipeline
    3. AWS CloudFormat

CodePipeline performs 4 tasks.

    1. Download source code from github
    2. Build the source on AWS build server. Build script for AWS build server can be found `here <https://github.com/Bodo-inc/Bodo/blob/master/buildspec.yml>`_
    3. Deploy the build artifact to EC2 instance
    4. Run whatever the user provides with `scripts <https://github.com/Bodo-inc/Bodo/blob/master/appspec.yml>`_
        - Run Bodo Benchmarking
        - Run TPCH Benchmarking
        - Upload the result to Bodo/`Benchmark_log repository <https://github.com/Bodo-inc/benchmark_logs>`_

CloudFormat performs 3 tasks.

    1. It will turn on the EC2 instance based on the schedule we set to reduce the cost.
    2. After turning on EC2 instance, CloudFormat will also trigger the pipeline.
    3. Turn off the EC2 instance based on the schedule. Make sure to give enough time to allow the pipeline to finish its tasks.
