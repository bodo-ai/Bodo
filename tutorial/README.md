# HPAT_tutorial

## Setting up your environment
This HPAT tutorial makes use of bodo (latest), numpy, pandas, daal4py (latest) and their dependences.

The easiest and most reliable way is to create a python environment using conda:

* Linux: `conda create -n hpattut -c ehsantn -c numba/label/dev -c defaults -c intel -c conda-forge bodo daal4py pandas=0.23 blas=*=mkl jupyter notebook`
* Windows: `conda create -n hpattut -c ehsantn -c numba/label/dev -c defaults -c intel bodo daal4py pandas=0.23 blas=*=mkl jupyter notebook`

Then activate the environment

`conda activate hpattut`

and you are ready to start the tutorial!

## The Tutorial

The main material is provided as juypter notebooks. To get started simply type

`jupyter notebook`

The main HPAT tutorial is in the notebook `bodo.ipynb`.
An example of an advanced analytics code is provided in `intraday_mean.py`.
A more complete notebook about using daal4py with HPAT can be found in `	daal4py_data_science.ipynb`.
