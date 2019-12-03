# Bodo Getting Started Tutorial

## Setting up your environment
The easiest and most reliable way to setup Bodo is to create a python environment using conda:

#* Linux: `conda create -n bodo_tut -c ehsantn -c numba/label/dev -c defaults -c intel -c conda-forge bodo daal4py pandas=0.23 blas=*=mkl jupyter notebook`
#* Windows: `conda create -n bodo_tut -c ehsantn -c numba/label/dev -c defaults -c intel bodo daal4py pandas=0.23 blas=*=mkl jupyter notebook`
* Linux: `conda create -n bodo_tut -c defaults -c conda-forge -c file:///path/to/bodo/pkg bodo pandas=0.24 jupyter ipyparallel`
* Windows: `conda create -n bodo_tut -c defaults -c intel -c file:///path/to/bodo/pkg bodo pandas=0.24 jupyter ipyparallel`


Then activate the environment:

`conda activate bodo_tut`

The main material is provided as Juypter notebooks and requires
# TODO: add link
`ipyparallel` MPI setup according to Bodo's Jupyter documentation.

## Tutorial Notebook

Start Jupyter:

`jupyter notebook`

go to `IPython Clusters` tab. Select the
number of engines (i.e., cores) you'd like to use and click `Start` next to the
`mpi` profile. Open the `bodo_getting_started.ipynb` notebook.
