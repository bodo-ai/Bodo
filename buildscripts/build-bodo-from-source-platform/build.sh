cd ~
source activate DEV
# This is necessary if rebuilding Bodo and mpich was previously
# removed, but if not it still succeeds and just does nothing
mamba install -y mpi mpich
rm -rf bodo Bodo
tar xf bodo.tar.gz
#mv patch.diff bodo/
cd Bodo
#git apply patch.diff
python setup.py develop
# Build BodoSQL
cd BodoSQL
python setup.py develop
conda remove mpi mpich --force -y  # all platform instances need to remove mpich to pick up Intel MPI
