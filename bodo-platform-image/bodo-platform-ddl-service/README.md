# Bodo DDL Service

Bodo DDL Service is a simple app which purpose is to validate if provided query is DDL and if it is, execute it.
It runs at port 8888 and has one endpoint `/api/v1/query` which takes `catalog`, `job_uuid` and `query` as an input.
If query is DDL it will be executed, if query is SELECT 1 it also will be executed, for all other queries service will
return `409` error with message that query should be executed via slurm.

## DEV environment

### Requirements
Your virtual environment should have installed packages:
- bodo
- bodosql
- bodo_platform_utils


To install all requirements follow docs: [Building Bodo from source](https://bodo.atlassian.net/wiki/spaces/B/pages/1018986500/Building+Bodo+from+Source#Build-the-Environment)

## Testing - Bodo Platform
Here we assume `bodo`, `bodosql` and `bodo_platform_utils` are installed in `bodo` user
environment.

### Start BodoDDL Service
To start `bodo-ddl.service` if is not started execute below commands:
```bash
sudo systemctl start bodo-ddl.service
```

### Validation
To check if service is running execute following command:
If you have catalog data defined
```bash
curl -X POST http://localhost:8888/api/v1/query -H "Content-Type: application/json" -d '{"query": "SELECT 1", "job_uuid": "job_uuid", "save_results": "True"}'
```
