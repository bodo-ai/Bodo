# Bodo SQL Service.

## DEV environment

### Requirements
Your virtual environment should have installed packages:
- bodo
- bodosql
- bodo_platform_utils


To install all requirements follow docs: [Building Bodo from source](https://bodo.atlassian.net/wiki/spaces/B/pages/1018986500/Building+Bodo+from+Source#Build-the-Environment)

### Validation
Start service using slurm:
```bash
srun -n X --label python /opt/bodo-sql/main.py
```

Send example job query to service:
```bash
curl \
  --header "Content-Type: application/json" \
  --request POST \
  --data '{"job_uuid":"46d6819e-d93e-407d-a96d-fea795cb1618", "catalog": "TPCHSF1", "result_dir": "/bodofs/bodo-sql/output", "query_filename": "/bodofs/bodo-sql/query.sql"}' \
  http://localhost:5000/api/v1/sql
```