import bodo

def count_array_REPs():
    from bodo.distributed import Distribution
    vals = bodo.distributed.dist_analysis.array_dists.values()
    return sum([v==Distribution.REP for v in vals])

def count_parfor_REPs():
    from bodo.distributed import Distribution
    vals = bodo.distributed.dist_analysis.parfor_dists.values()
    return sum([v==Distribution.REP for v in vals])

def count_parfor_OneDs():
    from bodo.distributed import Distribution
    vals = bodo.distributed.dist_analysis.parfor_dists.values()
    return sum([v==Distribution.OneD for v in vals])

def count_array_OneDs():
    from bodo.distributed import Distribution
    vals = bodo.distributed.dist_analysis.array_dists.values()
    return sum([v==Distribution.OneD for v in vals])

def count_parfor_OneD_Vars():
    from bodo.distributed import Distribution
    vals = bodo.distributed.dist_analysis.parfor_dists.values()
    return sum([v==Distribution.OneD_Var for v in vals])

def count_array_OneD_Vars():
    from bodo.distributed import Distribution
    vals = bodo.distributed.dist_analysis.array_dists.values()
    return sum([v==Distribution.OneD_Var for v in vals])

def dist_IR_contains(*args):
    return sum([(s in bodo.distributed.fir_text) for s in args])

@bodo.jit
def get_rank():
    return bodo.distributed_api.get_rank()

@bodo.jit
def get_start_end(n):
    rank = bodo.distributed_api.get_rank()
    n_pes = bodo.distributed_api.get_size()
    start = bodo.distributed_api.get_start(n, n_pes, rank)
    end = bodo.distributed_api.get_end(n, n_pes, rank)
    return start, end
