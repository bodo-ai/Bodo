SELECT
  s_acctbal,
  s_name,
  n_name,
  p_partkey,
  p_mfgr,
  s_address,
  s_phone,
  s_comment
FROM supplier, nation, partsupp, part
WHERE s_suppkey = ps_suppkey
  AND n_nationkey = s_nationkey
  AND p_partkey = ps_partkey
  AND p_name LIKE '%green%'
  AND ps_supplycost = (
    SELECT MIN(ps2.ps_supplycost)
    FROM partsupp ps2
    WHERE ps2.ps_partkey = ps_partkey
      AND ps2.ps_suppkey IN (
        SELECT s2.s_suppkey FROM supplier s2 WHERE s2.s_nationkey = s_nationkey
      )
  )
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
LIMIT 100;
