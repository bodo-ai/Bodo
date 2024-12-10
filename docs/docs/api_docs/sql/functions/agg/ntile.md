# NTILE
`#!sql NTILE(N)`

Divides the partitioned groups into N buckets based on
ordering. For example if N=3 and there are 30 rows in a
partition, the first 10 are assigned 1, the next 10 are
assigned 2, and the final 10 are assigned 3. In cases where
the number of rows cannot be evenly divided by the number
of buckets, the first buckets will have one more value
than the last bucket. For example, if N=4 and there are
22 rows in a partition, the first 6 are assigned 1, the
next 6 are assigned 2, the next 5 are assigned 3, and
the last 5 are assigned 4.


