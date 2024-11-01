# BodoSQL

## BodoSQL: Bodo's Vectorized SQL execution engine for clusters

BodoSQL is Bodo's vectorized SQL execution engine, designed to run on both a single laptop
and across a cluster of machines. BodoSQL integrates with Bodo's Python JIT compiler to
enable high performance analytics split across Python and SQL boundaries.

BodoSQL Documentation: https://docs.bodo.ai/latest/api_docs/sql/

## Additional Requirements

BodoSQL depends on the Bodo package with the same version. If you are already using Bodo you will
need to update your Bodo package to the same version as BodoSQL.

BodoSQL also depends on having Java installed on your system. You will need to download either Java 11
or Java 17 from one of the main available distribution and ensure that the `JAVA_HOME` environment
variable is properly set. If you can run `java -version` in your terminal and see the correct Java version, you are good to go.
