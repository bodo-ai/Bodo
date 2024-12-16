# bodo.get_rank 

`bodo.get_rank()`

Get the process number from Bodo (called `rank` in MPI terminology).

### Example Usage

    
Save following code in `get_rank.py` file and run with multiple cores.

```py
import bodo

@bodo.jit
def run_in_parallel():
    # some work only on rank 0
    if bodo.get_rank() == 0:
        print("rank 0 done")

    # some work on every process
    print("rank", bodo.get_rank(), "here")
run_in_parallel()
```

```console 
BODO_NUM_WORKERS=4 python get_rank.py
```

Output

```console
rank 0 done
rank 0 here
rank 1 here
rank 2 here
rank 3 here
```

