This is a dummy folder created for setting HADOOP_HOME. This allows Spark to initialize on Windows when
Hadoop catalog is specified. Spark checks for HADOOP_HOME and $HADOOP_HOME\bin\winutils.exe
during initialization even if Hadoop is never used later.

$env:HADOOP_HOME = "C:\Users\ehsan\Bodo\buildscripts\local_utils\hadoop_dummy"
$env:Path += ';C:\Users\ehsan\Bodo\buildscripts\local_utils\hadoop_dummy\bin'


https://github.com/cdarlint/winutils/
https://github.com/globalmentor/hadoop-bare-naked-local-fs
