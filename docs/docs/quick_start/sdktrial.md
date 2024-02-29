#Bodo SDK Trial {#bodosdktrial}

If you want to see how Bodo integrates with your existing data stack, you can run a SQL query on sample Snowflake data using an AWS cluster managed by Bodo. 


## Prerequisites

- Sign up for the Bodo SDK trial [here](https://www.bodo.ai/sdktrial). You will receive an email with a google drive link to a python script. Download the script. 
- If you don't have Python 3.7 or later installed, download and install it from [here](https://www.python.org/downloads/). You also need to have pip installed. You can install pip by running the following command in your terminal:
    ```bash
    python -m ensurepip --default-pip
    ```

!!! tip
    We recommend using a virtual environment for the bodo SDK trial, but it is not required.


## Running the script

1. Open a terminal and navigate to the directory where you downloaded the script. E.g. if you downloaded the script to `~/Downloads`, run the following command:
    ```shell
    cd ~/Downloads
    ```
   
2. Run the following command to install the Bodo SDK:
    ```bash
    pip install bodosdk --upgrade
    ```
   
3. Run the following command to run the script:
    ```bash
    python bodo_sdk_trial.py
    ```
   

## What the script does

The script will execute the following query on the [TPCH_SF1](https://docs.snowflake.com/en/user-guide/sample-data-tpch.html) table in Snowflake using Bodo. The query will be executed on a 1 node `r5.2xlarge` AWS cluster. 

```sql
select
    l_returnflag,
    l_linestatus,
    sum(l_quantity) as sum_qty,
    count(*) as count_order
from
    TPCH_SF1.lineitem
where
    l_shipdate <= date '1998-12-01'
group by
    l_returnflag,
    l_linestatus
order by
    l_returnflag,
    l_linestatus
```

Once the query is executed, the script will print the result of the query. Since Bodo executes the query in parallel across multiple cores, the output will be split across the different cores, e.g.:

```shell
Output:
3:   L_RETURNFLAG L_LINESTATUS     SUM_QTY  COUNT_ORDER
3: 3            R            F  37719753.0      1478870
0:   L_RETURNFLAG L_LINESTATUS     SUM_QTY  COUNT_ORDER
0: 0            A            F  37734107.0      1478493
0: 1            N            F    991417.0        38854
0: 2            N            O  76633518.0      3004998
```

In this example, the output is split across 2 cores, with the output from core 3 and core 0 being printed separately.

The query is submitted as a job to the bodo platform and the logs will be downloaded to the folder you are executing the script from.
   
   
   


